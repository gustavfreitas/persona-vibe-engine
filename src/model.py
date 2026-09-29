"""Treino do K-Means para agrupar faixas por 'mood' (pipeline offline)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Constantes centralizadas (app.py e predict.py tambÃ©m as importam)
FEATURE_COLUMNS = [
    "valence", "energy", "acousticness", "instrumentalness",
    "danceability", "loudness", "tempo",
]
METADATA_COLUMNS = ["track_name", "artists"]
GENRE_COLUMN = "track_genre"  # opcional: se existir, Ã© preservada no output
N_CLUSTERS = 5
RANDOM_STATE = 42  # garante resultados reprodutÃ­veis
N_TRAITS_PER_NAME = 2  # nÂº de caracterÃ­sticas usadas para nomear cada cluster
DEFAULT_OUTPUT = Path("data/clustered_tracks.csv")
DEFAULT_MODEL_PATH = Path("models/mood_model.joblib")

# RÃ³tulos (valor alto, valor baixo) usados para descrever cada cluster
TRAIT_LABELS = {
    "valence": ("Alegre", "MelancÃ³lico"),
    "energy": ("EnÃ©rgico", "Calmo"),
    "acousticness": ("AcÃºstico", "Pouco acÃºstico"),
    "instrumentalness": ("Instrumental", "Vocal"),
    "danceability": ("DanÃ§Ã¡vel", "Pouco danÃ§Ã¡vel"),
    "loudness": ("Intenso", "Suave"),
    "tempo": ("RÃ¡pido", "Lento"),
}


@dataclass
class MoodModel:
    """Agrupa tudo o que Ã© preciso para classificar faixas: scaler, K-Means,
    nomes dos clusters e a lista (ordenada) de features usadas no treino."""

    scaler: StandardScaler
    kmeans: KMeans
    cluster_names: dict[int, str]
    feature_columns: list[str]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devolve uma cÃ³pia do DataFrame com 'cluster' e 'cluster_name'.

        Usa o scaler JÃ AJUSTADO no treino (transform, nunca fit): faixas novas
        tÃªm de ser escaladas com a mÃ©dia/desvio do treino, senÃ£o as distÃ¢ncias
        aos centrÃ³ides deixam de fazer sentido.
        """
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise KeyError(f"Colunas ausentes: {sorted(missing)}")

        features = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        if features.isna().any().any():
            raise ValueError("Existem nulos ou valores nÃ£o numÃ©ricos nas features.")

        result = df.copy()
        result["cluster"] = self.kmeans.predict(self.scaler.transform(features))
        result["cluster_name"] = result["cluster"].map(self.cluster_names)
        return result


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega o CSV e valida a presenÃ§a das colunas necessÃ¡rias."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nÃ£o encontrado: {path}")

    df = pd.read_csv(path)

    # Datasets exportados do Pandas costumam trazer um Ã­ndice como coluna extra
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    missing = set(FEATURE_COLUMNS + METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no CSV: {sorted(missing)}")

    logger.info("Dataset carregado: %d linhas, %d colunas", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trata nulos, valores invÃ¡lidos e faixas duplicadas."""
    df = df.copy()

    # Garante que as features sÃ£o numÃ©ricas (valores invÃ¡lidos viram NaN)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + METADATA_COLUMNS)

    # O mesmo track_id pode aparecer em vÃ¡rios gÃ©neros; mantÃ©m a primeira
    # ocorrÃªncia (e, portanto, o primeiro gÃ©nero associado).
    dedup_key = ["track_id"] if "track_id" in df.columns else METADATA_COLUMNS
    df = df.drop_duplicates(subset=dedup_key)

    if GENRE_COLUMN in df.columns:
        df[GENRE_COLUMN] = df[GENRE_COLUMN].fillna("desconhecido")

    logger.info("Limpeza: %d -> %d linhas", before, len(df))
    return df.reset_index(drop=True)


def name_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Gera um nome legÃ­vel para cada cluster.

    Compara a mÃ©dia de cada feature no cluster com a mÃ©dia global (em z-score)
    e usa as caracterÃ­sticas que mais se destacam. Ex.: "Calmo Â· AcÃºstico".
    """
    global_mean = df[FEATURE_COLUMNS].mean()
    global_std = df[FEATURE_COLUMNS].std().replace(0, 1)
    z_scores = (df.groupby("cluster")[FEATURE_COLUMNS].mean() - global_mean) / global_std

    names: dict[int, str] = {}
    for cluster_id, row in z_scores.iterrows():
        top = row.abs().nlargest(N_TRAITS_PER_NAME).index
        traits = [TRAIT_LABELS[f][0 if row[f] > 0 else 1] for f in top]
        names[int(cluster_id)] = " Â· ".join(traits)

    # Nomes repetidos ficariam ambÃ­guos: acrescenta o id do cluster
    counts = pd.Series(names).value_counts()
    return {
        cid: f"{name} (#{cid})" if counts[name] > 1 else name
        for cid, name in names.items()
    }


def fit_model(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> MoodModel:
    """Ajusta scaler + K-Means e devolve o modelo completo."""
    if len(df) < n_clusters:
        raise ValueError(f"NecessÃ¡rio ao menos {n_clusters} faixas; hÃ¡ {len(df)}.")

    # O K-Means usa distÃ¢ncia euclidiana: sem padronizaÃ§Ã£o, variÃ¡veis com
    # maior variÃ¢ncia (ex.: tempo) dominariam o agrupamento.
    scaler = StandardScaler().fit(df[FEATURE_COLUMNS])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    kmeans.fit(scaler.transform(df[FEATURE_COLUMNS]))
    logger.info("InÃ©rcia do modelo: %.2f", kmeans.inertia_)

    names = name_clusters(df.assign(cluster=kmeans.labels_))
    return MoodModel(scaler, kmeans, names, list(FEATURE_COLUMNS))


def cluster_tracks(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Atalho: treina o modelo e devolve o DataFrame com 'cluster' e 'cluster_name'."""
    return fit_model(df, n_clusters).predict(df)


def save_model(model: MoodModel, path: Path = DEFAULT_MODEL_PATH) -> None:
    """Guarda o modelo com joblib.

    Grava-se um dicionÃ¡rio simples, e nÃ£o o objeto MoodModel: quando o script
    corre com `python -m src.model`, a classe fica registada como
    `__main__.MoodModel` e o ficheiro nÃ£o poderia ser carregado noutro sÃ­tio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": model.scaler,
            "kmeans": model.kmeans,
            "cluster_names": model.cluster_names,
            "feature_columns": model.feature_columns,
        },
        path,
    )
    logger.info("Modelo guardado em %s", path)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> MoodModel:
    """Carrega um modelo guardado por `save_model` (apenas ficheiros de confianÃ§a)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo nÃ£o encontrado: {path}. Execute antes `python -m src.model`."
        )
    return MoodModel(**joblib.load(path))


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Exporta o DataFrame final, criando a pasta de destino se necessÃ¡rio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Resultado salvo em %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClusterizaÃ§Ã£o musical por mood.")
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Caminho do CSV de entrada (dataset do Kaggle).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"CSV de saÃ­da (padrÃ£o: {DEFAULT_OUTPUT}).")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Onde guardar o modelo (padrÃ£o: {DEFAULT_MODEL_PATH}).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    df = clean_data(load_dataset(args.input_csv))
    model = fit_model(df)
    clustered = model.predict(df)

    save_dataset(clustered, args.output)
    save_model(model, args.model_path)

    # Perfil mÃ©dio por cluster: ajuda a validar os nomes atribuÃ­dos
    profile = clustered.groupby(["cluster", "cluster_name"])[FEATURE_COLUMNS].mean().round(2)
    profile["n_faixas"] = clustered.groupby(["cluster", "cluster_name"]).size()
    print("\nPerfil mÃ©dio dos clusters:\n", profile.to_string())


if __name__ == "__main__":
    main()
