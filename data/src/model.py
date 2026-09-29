"""Treino do modelo K-Means para agrupar faixas por 'mood'."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

N_CLUSTERS = 5
RANDOM_STATE = 42  # reprodutibilidade
DEFAULT_OUTPUT = Path("data/tracks_clustered.csv")

# Features usadas no clustering (podem ser ajustadas)
FEATURE_COLUMNS = [
    "valence", "energy", "acousticness", "instrumentalness",
    "danceability", "loudness", "tempo",
]


def train_mood_model(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    """Normaliza as features, treina o K-Means e devolve o DataFrame com
    uma coluna extra 'cluster'."""
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise KeyError(f"Colunas em falta no DataFrame: {sorted(missing)}")

    # Limpeza: sem nulos nas features e sem faixas repetidas
    data = df.dropna(subset=feature_columns).drop_duplicates().copy()
    if len(data) < n_clusters:
        raise ValueError(
            f"São necessárias pelo menos {n_clusters} faixas; existem {len(data)}."
        )

    # K-Means usa distâncias euclidianas, logo as escalas têm de ser comparáveis
    # (ex.: 'tempo' ~ 120 vs 'valence' ~ 0.5). O StandardScaler dá média 0 e
    # desvio-padrão 1 a cada coluna.
    scaled = StandardScaler().fit_transform(data[feature_columns])

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    data["cluster"] = model.fit_predict(scaled)

    logger.info("Inércia do modelo: %.2f", model.inertia_)
    return data


def save_results(df: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Guarda o DataFrame com clusters em CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Resultados guardados em %s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clustering musical por mood.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--playlists", nargs="+", help="IDs de playlists Spotify")
    source.add_argument("--input-csv", type=Path, help="CSV com audio features")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()

    if args.playlists:
        # Import local: evita exigir credenciais quando se usa --input-csv
        from src.spotify_client import SpotifyClient

        df = SpotifyClient().get_features_dataframe(args.playlists)
    else:
        df = pd.read_csv(args.input_csv)

    clustered = train_mood_model(df)
    save_results(clustered, args.output)

    # Perfil médio de cada cluster, útil para nomear os moods manualmente
    print(clustered.groupby("cluster")[FEATURE_COLUMNS].mean().round(2))


if __name__ == "__main__":
    main()
