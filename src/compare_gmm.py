"""Compara K-Means e GMM (Gaussian Mixture) nas mesmas features padronizadas."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.model import FEATURE_COLUMNS, N_CLUSTERS, RANDOM_STATE, clean_data, load_dataset

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 10_000       # silhouette é O(n²): usa uma amostra
AMBIGUOUS_THRESHOLD = 0.6  # faixa "ambígua" se a maior probabilidade for < 60 %
K_RANGE = range(5, 6)


def fit_gmm(scaled: np.ndarray, k: int, covariance: str = "full") -> GaussianMixture:
    """Ajusta um GMM; n_init=3 reduz o risco de ficar num mínimo local mau."""
    return GaussianMixture(
        n_components=k, covariance_type=covariance, n_init=3, random_state=RANDOM_STATE
    ).fit(scaled)


def evaluate_gmm(
    scaled: np.ndarray, k: int, covariance: str = "full", sample_size: int = SAMPLE_SIZE
) -> dict:
    """Métricas de um GMM: BIC/AIC (menor é melhor), silhouette, e confiança
    média das atribuições (quão 'firmes' são os clusters)."""
    gmm = fit_gmm(scaled, k, covariance)
    labels = gmm.predict(scaled)
    confidence = gmm.predict_proba(scaled).max(axis=1)

    # silhouette exige >= 2 clusters efetivamente usados
    n_used = len(np.unique(labels))
    silhouette = (
        silhouette_score(
            scaled, labels,
            sample_size=min(sample_size, len(scaled)), random_state=RANDOM_STATE,
        )
        if n_used > 1 else float("nan")
    )
    return {
        "k": k,
        "bic": gmm.bic(scaled),
        "aic": gmm.aic(scaled),
        "silhouette": silhouette,
        "mean_confidence": confidence.mean(),
        "ambiguous_pct": 100 * (confidence < AMBIGUOUS_THRESHOLD).mean(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara K-Means e GMM.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--covariance", default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="Tipo de covariância do GMM (padrão: full).")
    parser.add_argument("--save", type=Path, default=None,
                        help="Se indicado, grava o CSV com clusters e probabilidades do GMM (k=5).")
    parser.add_argument("--k-only", type=int, default=None,
                        help="Avalia apenas este k (mais rápido) em vez de percorrer 2-10.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    df = clean_data(load_dataset(args.input_csv))
    scaled = StandardScaler().fit_transform(df[FEATURE_COLUMNS])

    # 1) GMM para vários k
    k_values = [args.k_only] if args.k_only else K_RANGE
    rows = [evaluate_gmm(scaled, k, args.covariance) for k in k_values]
    table = pd.DataFrame(rows).set_index("k")
    print(f"\nGMM (covariância '{args.covariance}') para vários k:")
    print(table.round({"bic": 0, "aic": 0, "silhouette": 3,
                       "mean_confidence": 3, "ambiguous_pct": 1}).to_string())

    # 2) K-Means vs GMM com k=5
    kmeans_labels = KMeans(
        n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(scaled)
    gmm = fit_gmm(scaled, N_CLUSTERS, args.covariance)
    gmm_labels = gmm.predict(scaled)

    sample = np.random.default_rng(RANDOM_STATE).choice(
        len(scaled), size=min(SAMPLE_SIZE, len(scaled)), replace=False
    )
    print(f"\nk={N_CLUSTERS}: comparação direta")
    print(f"  Silhouette K-Means : {silhouette_score(scaled[sample], kmeans_labels[sample]):.3f}")
    print(f"  Silhouette GMM     : {silhouette_score(scaled[sample], gmm_labels[sample]):.3f}")
    print(f"  Concordância (ARI) : {adjusted_rand_score(kmeans_labels, gmm_labels):.3f}"
          "  (1 = iguais, 0 = sem relação)")
    print("\nTabela cruzada (linhas: K-Means, colunas: GMM):")
    print(pd.crosstab(kmeans_labels, gmm_labels).to_string())

    # 3) Opcional: exporta as probabilidades por faixa
    if args.save:
        proba = gmm.predict_proba(scaled)
        out = df.copy()
        out["gmm_cluster"] = gmm_labels
        out["gmm_confidence"] = proba.max(axis=1)
        for i in range(N_CLUSTERS):
            out[f"p_cluster_{i}"] = proba[:, i]
        args.save.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.save, index=False)
        logger.info("Probabilidades do GMM guardadas em %s", args.save)


if __name__ == "__main__":
    main()