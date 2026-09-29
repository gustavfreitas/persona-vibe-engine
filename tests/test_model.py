"""Testes do pipeline de treino (src/model.py) com dados sintéticos."""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    FEATURE_COLUMNS,
    N_CLUSTERS,
    clean_data,
    cluster_tracks,
    load_dataset,
    save_dataset,
)


def make_tracks(n_per_group: int = 20, n_groups: int = 5, seed: int = 0) -> pd.DataFrame:
    """Gera faixas em grupos bem separados, para o K-Means os recuperar."""
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        features = rng.normal(loc=g * 10, scale=0.5, size=(n_per_group, len(FEATURE_COLUMNS)))
        frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        frame["track_id"] = [f"{g}-{i}" for i in range(n_per_group)]
        frame["track_name"] = [f"Song {g}-{i}" for i in range(n_per_group)]
        frame["artists"] = f"Artist {g}"
        frame["track_genre"] = ["pop" if i % 2 == 0 else "rock" for i in range(n_per_group)]
        frame["group"] = g  # verdade-terreno, só para os testes
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------- load_dataset ---------------------------- #
def test_load_dataset_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nao_existe.csv")


def test_load_dataset_missing_columns(tmp_path):
    path = tmp_path / "incompleto.csv"
    pd.DataFrame({"track_name": ["a"]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_dataset(path)


def test_load_dataset_drops_unnamed_index_column(tmp_path):
    path = tmp_path / "com_indice.csv"
    make_tracks().to_csv(path)  # index=True cria a coluna 'Unnamed: 0'
    loaded = load_dataset(path)
    assert not any(col.startswith("Unnamed") for col in loaded.columns)


# ---------------------------- clean_data ----------------------------- #
def test_clean_data_removes_nulls_and_duplicates():
    df = make_tracks(n_per_group=4)  # 20 linhas
    df.loc[0, "valence"] = np.nan          # feature nula
    df.loc[1, "track_name"] = None         # metadado nulo
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # track_id repetido
    assert len(clean_data(df)) == 18


def test_clean_data_coerces_non_numeric_values():
    df = make_tracks(n_per_group=4)
    df["tempo"] = df["tempo"].astype(object)
    df.loc[0, "tempo"] = "abc"
    cleaned = clean_data(df)
    assert len(cleaned) == 19
    assert pd.api.types.is_numeric_dtype(cleaned["tempo"])


def test_clean_data_fills_missing_genre():
    df = make_tracks(n_per_group=4)
    df.loc[2, "track_genre"] = None
    cleaned = clean_data(df)
    assert (cleaned["track_genre"] == "desconhecido").sum() == 1


# -------------------------- cluster_tracks --------------------------- #
def test_cluster_tracks_adds_columns_and_expected_clusters():
    result = cluster_tracks(clean_data(make_tracks()))
    assert {"cluster", "cluster_name"} <= set(result.columns)
    assert result["cluster"].nunique() == N_CLUSTERS


def test_cluster_tracks_recovers_separated_groups():
    result = cluster_tracks(clean_data(make_tracks()))
    # cada grupo verdadeiro deve cair num único cluster
    assert (result.groupby("group")["cluster"].nunique() == 1).all()


def test_cluster_tracks_is_reproducible():
    df = clean_data(make_tracks())
    first = cluster_tracks(df)["cluster"].to_numpy()
    second = cluster_tracks(df)["cluster"].to_numpy()
    assert np.array_equal(first, second)


def test_cluster_names_are_unique():
    result = cluster_tracks(clean_data(make_tracks()))
    assert result["cluster_name"].nunique() == N_CLUSTERS


def test_cluster_tracks_needs_enough_rows():
    with pytest.raises(ValueError):
        cluster_tracks(make_tracks().head(3))


def test_cluster_tracks_does_not_mutate_input():
    df = clean_data(make_tracks())
    cluster_tracks(df)
    assert "cluster" not in df.columns


# --------------------------- save_dataset ---------------------------- #
def test_save_dataset_creates_parent_folders(tmp_path):
    output = tmp_path / "sub" / "pasta" / "saida.csv"
    save_dataset(make_tracks(n_per_group=2), output)
    assert output.exists()