"""Testes da avaliação do GMM com dados sintéticos bem separados."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.compare_gmm import evaluate_gmm
from src.model import FEATURE_COLUMNS, clean_data
from tests.test_model import make_tracks


@pytest.fixture
def scaled():
    df = clean_data(make_tracks())
    return StandardScaler().fit_transform(df[FEATURE_COLUMNS])


def test_evaluate_gmm_returns_expected_metrics(scaled):
    result = evaluate_gmm(scaled, k=5)
    assert {"k", "bic", "aic", "silhouette", "mean_confidence", "ambiguous_pct"} <= set(result)
    assert result["k"] == 5


def test_well_separated_groups_have_high_confidence(scaled):
    result = evaluate_gmm(scaled, k=5)
    assert result["mean_confidence"] > 0.95
    assert result["ambiguous_pct"] < 5


def test_true_k_has_lower_bic_than_too_few_clusters(scaled):
    assert evaluate_gmm(scaled, k=5)["bic"] < evaluate_gmm(scaled, k=2)["bic"]