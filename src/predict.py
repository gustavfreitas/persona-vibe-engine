"""Classifica faixas novas com o modelo já treinado (sem retreinar)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import DEFAULT_MODEL_PATH, MoodModel, load_model, save_dataset

logger = logging.getLogger(__name__)


def classify_dataframe(model: MoodModel, raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Classifica as linhas válidas de `raw`.

    Devolve (DataFrame com 'cluster' e 'cluster_name', nº de linhas descartadas).
    Linhas com features nulas, não numéricas ou infinitas são descartadas, porque
    não há como calcular a distância aos centróides.
    """
    missing = set(model.feature_columns) - set(raw.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no CSV: {sorted(missing)}")

    features = (
        raw[model.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = features.notna().all(axis=1)
    if not valid.any():
        raise ValueError("Nenhuma linha com features válidas.")

    df = raw[valid].copy()
    df[model.feature_columns] = features[valid]
    return model.predict(df), int((~valid).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atribui um mood a faixas novas.")
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="CSV com as colunas de audio features.")
    parser.add_argument("--output", type=Path, default=Path("data/new_tracks_clustered.csv"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    model = load_model(args.model_path)
    result, dropped = classify_dataframe(model, pd.read_csv(args.input_csv))
    if dropped:
        logger.warning("%d linha(s) ignorada(s) por features inválidas.", dropped)

    save_dataset(result, args.output)
    print(result.drop(columns=model.feature_columns).head(10).to_string())


if __name__ == "__main__":
    main()