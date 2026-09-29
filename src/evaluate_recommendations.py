"""Avalia configurações de recomendação com métricas objetivas (offline)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app import recommend
from src.model import DEFAULT_OUTPUT, GENRE_COLUMN, RANDOM_STATE

N_RECS = 5

# Cada configuração é um conjunto de argumentos para `recommend`.
# Pesos omissos valem 1. O filtro de género fica sempre desligado (ver docstring).
CONFIGS: dict[str, dict] = {
    "igual (atual)": {},
    "popularidade": {"prefer_popular": True},
    "valence x3, tempo 0": {"weights": {"valence": 3.0, "tempo": 0.0}},
    "energy x3": {"weights": {"energy": 3.0}},
    "acousticness x3": {"weights": {"acousticness": 3.0}},
    "instrumentalness x3": {"weights": {"instrumentalness": 3.0}},
}


def random_in_cluster(df: pd.DataFrame, selected: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Baseline: n faixas aleatórias do mesmo cluster."""
    pool = df[(df["cluster"] == selected["cluster"]) & (df["label"] != selected["label"])]
    return pool.sample(min(N_RECS, len(pool)), random_state=int(rng.integers(2**31 - 1)))


def random_anywhere(df: pd.DataFrame, selected: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Baseline: n faixas aleatórias do dataset todo (piso de referência)."""
    pool = df[df["label"] != selected["label"]]
    return pool.sample(N_RECS, random_state=int(rng.integers(2**31 - 1)))


def evaluate(
    queries: pd.DataFrame, name: str, recommender: Callable[[pd.Series], pd.DataFrame]
) -> dict:
    """Corre `recommender` em cada faixa de `queries` e agrega as métricas."""
    hits, diversity, popularity = [], [], []
    for _, selected in queries.iterrows():
        recs = recommender(selected)
        if recs.empty:
            continue
        hits.append((recs[GENRE_COLUMN] == selected[GENRE_COLUMN]).mean())
        first_artist = recs["artists"].str.split(";").str[0]
        diversity.append(first_artist.nunique() / len(recs))
        if "popularity" in recs.columns:
            popularity.append(recs["popularity"].mean())

    hits_arr = np.array(hits)
    n = len(hits_arr)
    if n == 0:
        raise ValueError(f"Nenhuma recomendação gerada para '{name}'.")
    # Intervalo de confiança de 95 % da média (aprox. normal) sobre as faixas
    ci = 1.96 * hits_arr.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {
        "config": name,
        "n_faixas": n,
        "genero_igual_%": 100 * hits_arr.mean(),
        "±95%": 100 * ci,
        "artistas_distintos": float(np.mean(diversity)),
        "popularidade_media": float(np.mean(popularity)) if popularity else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia configurações de recomendação.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["label"] = df["artists"] + " – " + df["track_name"]
    if GENRE_COLUMN not in df.columns:
        raise SystemExit(f"O CSV não tem a coluna '{GENRE_COLUMN}': não há como avaliar.")

    queries = df.sample(min(args.n_queries, len(df)), random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    rows = [
        evaluate(queries, name, lambda sel, kw=kw: recommend(df, sel, same_genre=False, **kw))
        for name, kw in CONFIGS.items()
    ]
    rows.append(evaluate(queries, "ALEATÓRIO no cluster", lambda sel: random_in_cluster(df, sel, rng)))
    rows.append(evaluate(queries, "ALEATÓRIO no dataset", lambda sel: random_anywhere(df, sel, rng)))

    table = pd.DataFrame(rows).set_index("config")
    print(f"\n{len(queries)} faixas de partida, {N_RECS} recomendações cada:\n")
    print(table.round({"genero_igual_%": 1, "±95%": 1, "artistas_distintos": 2,
                       "popularidade_media": 1}).to_string())


if __name__ == "__main__":
    main()