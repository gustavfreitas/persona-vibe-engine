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
from src.model import FEATURE_COLUMNS, GENRE_COLUMN, clean_data, cluster_tracks

logger = logging.getLogger(__name__)

# Constantes centralizadas (app.py e predict.py também as importam)
FEATURE_COLUMNS = [
    "valence", "energy", "acousticness", "instrumentalness",
    "danceability", "loudness", "tempo",
]
METADATA_COLUMNS = ["track_name", "artists"]
GENRE_COLUMN = "track_genre"  # opcional: se existir, é preservada no output
N_CLUSTERS = 5
RANDOM_STATE = 42  # garante resultados reprodutíveis
N_TRAITS_PER_NAME = 2  # nº de características usadas para nomear cada cluster
DEFAULT_OUTPUT = Path("data/clustered_tracks.csv")
DEFAULT_MODEL_PATH = Path("models/mood_model.joblib")

# Rótulos (valor alto, valor baixo) usados para descrever cada cluster
TRAIT_LABELS = {
    "valence": ("Alegre", "Melancólico"),
    "energy": ("Enérgico", "Calmo"),
    "acousticness": ("Acústico", "Pouco acústico"),
    "instrumentalness": ("Instrumental", "Vocal"),
    "danceability": ("Dançável", "Pouco dançável"),
    "loudness": ("Intenso", "Suave"),
    "tempo": ("Rápido", "Lento"),
}


@dataclass
class MoodModel:
    """Agrupa tudo o que é preciso para classificar faixas: scaler, K-Means,
    nomes dos clusters e a lista (ordenada) de features usadas no treino."""

    scaler: StandardScaler
    kmeans: KMeans
    cluster_names: dict[int, str]
    feature_columns: list[str]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devolve uma cópia do DataFrame com 'cluster' e 'cluster_name'.

        Usa o scaler JÁ AJUSTADO no treino (transform, nunca fit): faixas novas
        têm de ser escaladas com a média/desvio do treino, senão as distâncias
        aos centróides deixam de fazer sentido.
        """
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise KeyError(f"Colunas ausentes: {sorted(missing)}")

        features = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        if features.isna().any().any():
            raise ValueError("Existem nulos ou valores não numéricos nas features.")

        result = df.copy()
        result["cluster"] = self.kmeans.predict(self.scaler.transform(features))
        result["cluster_name"] = result["cluster"].map(self.cluster_names)
        return result


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega o CSV e valida a presença das colunas necessárias."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    # Datasets exportados do Pandas costumam trazer um índice como coluna extra
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    missing = set(FEATURE_COLUMNS + METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no CSV: {sorted(missing)}")

    logger.info("Dataset carregado: %d linhas, %d colunas", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trata nulos, valores inválidos e faixas duplicadas."""
    df = df.copy()

    # Garante que as features são numéricas (valores inválidos viram NaN)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + METADATA_COLUMNS)

    # O mesmo track_id pode aparecer em vários géneros; mantém a primeira
    # ocorrência (e, portanto, o primeiro género associado).
    dedup_key = ["track_id"] if "track_id" in df.columns else METADATA_COLUMNS
    df = df.drop_duplicates(subset=dedup_key)

    if GENRE_COLUMN in df.columns:
        df[GENRE_COLUMN] = df[GENRE_COLUMN].fillna("desconhecido")

    logger.info("Limpeza: %d -> %d linhas", before, len(df))
    return df.reset_index(drop=True)


def name_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Gera um nome legível para cada cluster.

    Compara a média de cada feature no cluster com a média global (em z-score)
    e usa as características que mais se destacam. Ex.: "Calmo · Acústico".
    """
    global_mean = df[FEATURE_COLUMNS].mean()
    global_std = df[FEATURE_COLUMNS].std().replace(0, 1)
    z_scores = (df.groupby("cluster")[FEATURE_COLUMNS].mean() - global_mean) / global_std

    names: dict[int, str] = {}
    for cluster_id, row in z_scores.iterrows():
        top = row.abs().nlargest(N_TRAITS_PER_NAME).index
        traits = [TRAIT_LABELS[f][0 if row[f] > 0 else 1] for f in top]
        names[int(cluster_id)] = " · ".join(traits)

    # Nomes repetidos ficariam ambíguos: acrescenta o id do cluster
    counts = pd.Series(names).value_counts()
    return {
        cid: f"{name} (#{cid})" if counts[name] > 1 else name
        for cid, name in names.items()
    }


def fit_model(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> MoodModel:
    """Ajusta scaler + K-Means e devolve o modelo completo."""
    if len(df) < n_clusters:
        raise ValueError(f"Necessário ao menos {n_clusters} faixas; há {len(df)}.")

    # O K-Means usa distância euclidiana: sem padronização, variáveis com
    # maior variância (ex.: tempo) dominariam o agrupamento.
    scaler = StandardScaler().fit(df[FEATURE_COLUMNS])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    kmeans.fit(scaler.transform(df[FEATURE_COLUMNS]))
    logger.info("Inércia do modelo: %.2f", kmeans.inertia_)

    names = name_clusters(df.assign(cluster=kmeans.labels_))
    return MoodModel(scaler, kmeans, names, list(FEATURE_COLUMNS))


def cluster_tracks(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Atalho: treina o modelo e devolve o DataFrame com 'cluster' e 'cluster_name'."""
    return fit_model(df, n_clusters).predict(df)


def save_model(model: MoodModel, path: Path = DEFAULT_MODEL_PATH) -> None:
    """Guarda o modelo com joblib.

    Grava-se um dicionário simples, e não o objeto MoodModel: quando o script
    corre com `python -m src.model`, a classe fica registada como
    `__main__.MoodModel` e o ficheiro não poderia ser carregado noutro sítio.
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
    """Carrega um modelo guardado por `save_model` (apenas ficheiros de confiança)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}. Execute antes `python -m src.model`."
        )
    return MoodModel(**joblib.load(path))


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Exporta o DataFrame final, criando a pasta de destino se necessário."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Resultado salvo em %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clusterização musical por mood.")
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Caminho do CSV de entrada (dataset do Kaggle).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"CSV de saída (padrão: {DEFAULT_OUTPUT}).")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Onde guardar o modelo (padrão: {DEFAULT_MODEL_PATH}).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    df = clean_data(load_dataset(args.input_csv))
    model = fit_model(df)
    clustered = model.predict(df)

    save_dataset(clustered, args.output)
    save_model(model, args.model_path)

    # Perfil médio por cluster: ajuda a validar os nomes atribuídos
    profile = clustered.groupby(["cluster", "cluster_name"])[FEATURE_COLUMNS].mean().round(2)
    profile["n_faixas"] = clustered.groupby(["cluster", "cluster_name"]).size()
    print("\nPerfil médio dos clusters:\n", profile.to_string())


if __name__ == "__main__":
    main()