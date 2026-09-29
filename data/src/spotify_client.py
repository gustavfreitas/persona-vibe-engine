"""Cliente Spotify: extrai faixas de playlists e respetivas audio features."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)

# A API aceita no máximo 100 IDs por pedido de audio-features.
AUDIO_FEATURES_BATCH_SIZE = 100

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


class SpotifyClient:
    """Encapsula a autenticação e as chamadas à API do Spotify."""

    def __init__(self) -> None:
        # Lê SPOTIPY_CLIENT_ID e SPOTIPY_CLIENT_SECRET do ficheiro .env
        load_dotenv()
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise EnvironmentError(
                "Credenciais em falta: define SPOTIPY_CLIENT_ID e "
                "SPOTIPY_CLIENT_SECRET no ficheiro .env."
            )

        auth = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        self._sp = spotipy.Spotify(auth_manager=auth, retries=3)
