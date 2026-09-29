"""Testes da lógica de busca e recomendação (app.py)."""

import pytest
import pandas as pd

from app import MAX_SEARCH_RESULTS, base_title, recommend, search_tracks
from src.model import GENRE_COLUMN, clean_data, cluster_tracks
from tests.test_model import make_tracks


@pytest.fixture
def clustered():
    df = cluster_tracks(clean_data(make_tracks()))
    df["label"] = df["artists"] + " – " + df["track_name"]
    return df


# ------------------------------ search ------------------------------- #
def test_search_is_case_insensitive(clustered):
    results = search_tracks(clustered, "sOnG 0-1")
    assert len(results) > 0
    assert results["track_name"].str.lower().str.contains("song 0-1").all()


def test_search_treats_query_as_plain_text(clustered):
    # regex=False: caracteres especiais não podem provocar erro
    assert search_tracks(clustered, "(").empty


def test_search_limits_number_of_results(clustered):
    assert len(search_tracks(clustered, "Song")) == MAX_SEARCH_RESULTS


# ---------------------------- recommend ------------------------------ #
def test_recommend_returns_n_from_same_cluster(clustered):
    selected = clustered.iloc[0]
    recs = recommend(clustered, selected)
    assert len(recs) == 5
    assert (recs["cluster"] == selected["cluster"]).all()


def test_recommend_excludes_selected_track(clustered):
    selected = clustered.iloc[0]
    assert selected["label"] not in recommend(clustered, selected)["label"].values


def test_recommend_is_sorted_by_distance(clustered):
    recs = recommend(clustered, clustered.iloc[0])
    assert recs["distance"].is_monotonic_increasing


def test_recommend_same_genre_filter(clustered):
    selected = clustered.iloc[0]
    recs = recommend(clustered, selected, same_genre=True)
    assert len(recs) > 0
    assert (recs[GENRE_COLUMN] == selected[GENRE_COLUMN]).all()


def test_recommend_returns_empty_when_no_candidates(clustered):
    single = clustered.iloc[[0]]
    assert recommend(single, single.iloc[0]).empty


def test_recommend_has_no_duplicate_labels(clustered):
    import pandas as pd

    # acrescenta uma cópia de uma faixa do mesmo cluster (outra "versão")
    df = pd.concat([clustered, clustered.iloc[[1]]], ignore_index=True)
    recs = recommend(df, df.iloc[0], n=19)
    assert len(recs) == 19
    assert recs["label"].is_unique

def test_base_title_strips_version_suffixes():
    assert base_title("Can't Help Falling In Love - Piano Version") == "can't help falling in love"
    assert base_title("Song (feat. X) - Remastered 2011") == "song"
    assert base_title("(Intro)") == "(intro)"  # não devolve string vazia


def test_recommend_excludes_alternative_versions(clustered):
    # cria uma "versão piano" com features idênticas às da faixa escolhida
    alt = clustered.iloc[[0]].copy()
    alt["track_name"] = alt["track_name"] + " - Piano Version"
    alt["label"] = alt["artists"] + " – " + alt["track_name"]
    df = pd.concat([clustered, alt], ignore_index=True)

    recs = recommend(df, df.iloc[0], n=19)
    assert alt["label"].iloc[0] not in recs["label"].values
    assert len(recs) == 19