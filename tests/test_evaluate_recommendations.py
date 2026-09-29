"""Testes da avaliação de recomendações, com dados sintéticos."""

import numpy as np
import pytest

from app import recommend
from src.evaluate_recommendations import evaluate, random_anywhere
from src.model import clean_data, cluster_tracks
from tests.test_model import make_tracks


@pytest.fixture
def clustered():
    df = cluster_tracks(clean_data(make_tracks()))
    df["label"] = df["artists"] + " – " + df["track_name"]
    return df


def test_evaluate_returns_metrics_in_valid_ranges(clustered):
    queries = clustered.head(10)
    result = evaluate(queries, "x", lambda sel: recommend(clustered, sel))
    assert 0 <= result["genero_igual_%"] <= 100
    assert 0 < result["artistas_distintos"] <= 1
    assert result["n_faixas"] == 10


def test_genre_filter_gives_full_consistency(clustered):
    queries = clustered.head(10)
    result = evaluate(queries, "x", lambda sel: recommend(clustered, sel, same_genre=True))
    assert result["genero_igual_%"] == 100


def test_random_baseline_returns_n_tracks(clustered):
    rng = np.random.default_rng(0)
    assert len(random_anywhere(clustered, clustered.iloc[0], rng)) == 5