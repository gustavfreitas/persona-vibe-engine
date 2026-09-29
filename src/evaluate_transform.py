"""Compara transformações das features antes do K-Means (k=5)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.model import FEATURE_COLUMNS, N_CLUSTERS, RANDOM_STATE, clean_data, load_dataset
from pathlib import Path

SAMPLE_SIZE = 10_000


def scale_baseline(df):
    return StandardScaler().fit_transform(df[FEATURE_COLUMNS])


def scale_binary_instrumental(df):
    """instrumentalness vira 0/1 (>0.5) antes de padronizar."""
    data = df[FEATURE_COLUMNS].copy()
    data["instrumentalness"] = (data["instrumentalness"] > 0.5).astype(float)
    return StandardScaler().fit_transform(data)


def scale_power(df):
    """Yeo-Johnson em todas as features (já padroniza no fim)."""
    return PowerTransformer(method="yeo-johnson").fit_transform(df[FEATURE_COLUMNS])


VARIANTS = {
    "baseline": scale_baseline,
    "instrumental binária": scale_binary_instrumental,
    "yeo-johnson": scale_power,
}


def main() -> None:
    df = clean_data(load_dataset(Path("data/dataset.csv")))
    baseline_labels = None

    print(f"{'variante':<24}{'silhouette':>11}{'ARI vs base':>13}{'menor':>8}{'maior':>8}")
    for name, transform in VARIANTS.items():
        scaled = transform(df)
        labels = KMeans(
            n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE
        ).fit_predict(scaled)
        if baseline_labels is None:
            baseline_labels = labels
        sil = silhouette_score(
            scaled, labels, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE
        )
        sizes = np.bincount(labels)
        ari = adjusted_rand_score(baseline_labels, labels)
        print(f"{name:<24}{sil:>11.3f}{ari:>13.3f}{sizes.min():>8}{sizes.max():>8}")


if __name__ == "__main__":
    main()