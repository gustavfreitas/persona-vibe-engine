"""Testes da persistência do modelo (joblib) e da previsão em faixas novas."""

import numpy as np
import pandas as pd
import pytest

from src.model import FEATURE_COLUMNS, clean_data, fit_model, load_model, save_model
from tests.test_model import make_tracks


@pytest.fixture
def trained():
    df = clean_data(make_tracks())
    return df, fit_model(df)


def test_roundtrip_gives_same_predictions(trained, tmp_path):
    df, model = trained
    path = tmp_path / "models" / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.array_equal(model.predict(df)["cluster"], loaded.predict(df)["cluster"])
    assert loaded.cluster_names == model.cluster_names


def test_load_model_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nao_existe.joblib")


def test_new_track_lands_in_expected_cluster(trained):
    df, model = trained
    train = model.predict(df)
    expected = train.loc[train["group"] == 3, "cluster"].iloc[0]
    # make_tracks centra o grupo g em g*10, logo o grupo 3 fica em 30
    new_track = pd.DataFrame([{c: 30.0 for c in FEATURE_COLUMNS}])
    assert model.predict(new_track)["cluster"].iloc[0] == expected


def test_predict_requires_all_feature_columns(trained):
    _, model = trained
    with pytest.raises(KeyError):
        model.predict(pd.DataFrame({"valence": [0.5]}))


def test_predict_rejects_nulls(trained):
    _, model = trained
    bad = pd.DataFrame([{c: 1.0 for c in FEATURE_COLUMNS}])
    bad.loc[0, "tempo"] = np.nan
    with pytest.raises(ValueError):
        model.predict(bad)