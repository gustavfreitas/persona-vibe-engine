"""Treino do K-Means para agrupar faixas por 'mood' (pipeline offline)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Constantes centralizadas (o app.py também as importa, evitando duplicação)
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


def cluster_tracks(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Normaliza as features, treina o K-Means e adiciona 'cluster' e 'cluster_name'."""
    if len(df) < n_clusters:
        raise ValueError(f"Necessário ao menos {n_clusters} faixas; há {len(df)}.")

    # O K-Means usa distância euclidiana: sem padronização, variáveis com
    # maior variância (ex.: tempo) dominariam o agrupamento.
    scaled = StandardScaler().fit_transform(df[FEATURE_COLUMNS])

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    result = df.copy()
    result["cluster"] = model.fit_predict(scaled)
    result["cluster_name"] = result["cluster"].map(name_clusters(result))

    logger.info("Inércia do modelo: %.2f", model.inertia_)
    return result


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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    df = clean_data(load_dataset(args.input_csv))
    clustered = cluster_tracks(df)
    save_dataset(clustered, args.output)

    # Perfil médio por cluster: ajuda a validar os nomes atribuídos
    profile = clustered.groupby(["cluster", "cluster_name"])[FEATURE_COLUMNS].mean().round(2)
    profile["n_faixas"] = clustered.groupby(["cluster", "cluster_name"]).size()
    print("\nPerfil médio dos clusters:\n", profile.to_string())


if __name__ == "__main__":
    main()