"""Testes dos auxiliares da interface (links do Spotify, duração, capas)."""

import numpy as np
import pandas as pd
import pytest

from app import cover_html, esc, format_duration, parse_track_id

TRACK_ID = "6rqhFgbbKwnb9MLmUQDhG6"


@pytest.mark.parametrize(
    "text",
    [
        f"https://open.spotify.com/track/{TRACK_ID}",
        f"https://open.spotify.com/track/{TRACK_ID}?si=abc123",
        f"https://open.spotify.com/intl-pt/track/{TRACK_ID}",
        f"spotify:track:{TRACK_ID}",
        f"  https://open.spotify.com/track/{TRACK_ID}  ",
    ],
)
def test_parse_track_id_accepts_links_and_uris(text):
    assert parse_track_id(text) == TRACK_ID


@pytest.mark.parametrize(
    "text",
    [
        "Yellow",
        f"https://open.spotify.com/album/{TRACK_ID}",
        f"https://open.spotify.com/playlist/{TRACK_ID}",
        "spotify:track:curto",
        "",
    ],
)
def test_parse_track_id_rejects_everything_else(text):
    assert parse_track_id(text) is None


def test_format_duration():
    assert format_duration(215_000) == "3:35"
    assert format_duration(59_999) == "0:59"
    assert format_duration(float("nan")) == "–"


def test_esc_escapes_html_and_handles_missing_values():
    assert esc("<b>&") == "&lt;b&gt;&amp;"
    assert esc(np.nan) == ""


def test_cover_depends_on_valence_and_energy():
    sad = cover_html(pd.Series({"valence": 0.0, "energy": 0.5}))
    happy = cover_html(pd.Series({"valence": 1.0, "energy": 0.5}))
    assert sad != happy
    assert "hsl(0," in sad


def test_cover_clips_out_of_range_values():
    wild = cover_html(pd.Series({"valence": 5.0, "energy": -3.0}))
    clipped = cover_html(pd.Series({"valence": 1.0, "energy": 0.0}))
    assert wild == clipped
