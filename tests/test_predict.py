"""Testes da classificação de faixas novas (src/predict.py)."""

import pytest

from src.model import clean_data, fit_model
from src.predict import classify_dataframe
from tests.test_model import make_tracks


@pytest.fixture
def model():
    return fit_model(clean_data(make_tracks()))


def test_drops_invalid_rows(model):
    new = make_tracks(n_per_group=2)
    new["tempo"] = new["tempo"].astype(object)
    new.loc[0, "tempo"] = "abc"
    result, dropped = classify_dataframe(model, new)
    assert dropped == 1
    assert len(result) == len(new) - 1
    assert {"cluster", "cluster_name"} <= set(result.columns)


def test_missing_columns_raise(model):
    with pytest.raises(KeyError):
        classify_dataframe(model, make_tracks(n_per_group=2)[["valence"]])


def test_all_rows_invalid_raises(model):
    new = make_tracks(n_per_group=2)
    new["valence"] = None
    with pytest.raises(ValueError):
        classify_dataframe(model, new)