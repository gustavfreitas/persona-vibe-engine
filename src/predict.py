"""Classifica faixas novas com o modelo já treinado (sem retreinar)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.model import DEFAULT_MODEL_PATH, load_model, save_dataset

logger = logging.getLogger(__name__)


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
    df = pd.read_csv(args.input_csv)

    # Linhas com features inválidas não podem ser classificadas: são descartadas
    features = df[model.feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = features.notna().all(axis=1)
    if not valid.all():
        logger.warning("%d linha(s) ignorada(s) por features inválidas.", (~valid).sum())
    df = df[valid].copy()
    df[model.feature_columns] = features[valid]

    result = model.predict(df)
    save_dataset(result, args.output)
    print(result.drop(columns=model.feature_columns).head(10).to_string())


if __name__ == "__main__":
    main()