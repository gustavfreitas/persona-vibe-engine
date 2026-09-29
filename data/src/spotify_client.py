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
  # ------------------------------------------------------------------ #
    # Métodos públicos
    # ------------------------------------------------------------------ #
    def get_features_dataframe(self, playlist_ids: Iterable[str]) -> pd.DataFrame:
        """Devolve um DataFrame com metadados + audio features de todas as
        faixas das playlists indicadas (sem duplicados)."""
        tracks: list[dict] = []
        for playlist_id in playlist_ids:
            logger.info("A obter faixas da playlist %s", playlist_id)
            tracks.extend(self._get_playlist_tracks(playlist_id))

        if not tracks:
            raise ValueError("Nenhuma faixa encontrada nas playlists fornecidas.")

        # Remove faixas repetidas entre playlists (mesmo track_id)
        meta_df = pd.DataFrame(tracks).drop_duplicates(subset="track_id")

        features_df = pd.DataFrame(
            self._get_audio_features(meta_df["track_id"].tolist())
        )
        if features_df.empty:
            raise ValueError("A API não devolveu audio features para as faixas.")

        # 'id' da API corresponde ao nosso 'track_id'
        features_df = features_df.rename(columns={"id": "track_id"})
        features_df = features_df[["track_id", *AUDIO_FEATURE_KEYS]]

        return meta_df.merge(features_df, on="track_id", how="inner").reset_index(
            drop=True
        )

    # ------------------------------------------------------------------ #
    # Métodos privados
    # ------------------------------------------------------------------ #
    def _get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """Percorre todas as páginas de uma playlist e extrai metadados."""
        results = self._sp.playlist_items(
            playlist_id,
            fields="items(track(id,name,artists(name))),next",
            additional_types=("track",),
            limit=100,
        )
        tracks: list[dict] = []
        while results:
            for item in results["items"]:
                track = item.get("track")
                # Ignora faixas locais / removidas (sem ID)
                if not track or not track.get("id"):
                    continue
                tracks.append(
                    {
                        "track_id": track["id"],
                        "track_name": track["name"],
                        "artists": ", ".join(a["name"] for a in track["artists"]),
                    }
                )
            results = self._sp.next(results) if results.get("next") else None
        return tracks

    def _get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """Obtém audio features em lotes de 100 IDs."""
        features: list[dict] = []
        for start in range(0, len(track_ids), AUDIO_FEATURES_BATCH_SIZE):
            batch = track_ids[start : start + AUDIO_FEATURES_BATCH_SIZE]
            try:
                response = self._sp.audio_features(batch)
            except spotipy.SpotifyException as exc:
                if exc.http_status == 403:
                    raise RuntimeError(
                        "403 no endpoint audio-features: apps criadas após "
                        "27/11/2024 já não têm acesso. Usa --input-csv com "
                        "um dataset alternativo."
                    ) from exc
                raise
            # A API devolve None para faixas sem features
            features.extend(f for f in response if f)
        return features
