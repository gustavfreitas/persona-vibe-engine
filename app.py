"""Interface Streamlit: recomendações por mood e classificação de faixas novas."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT,
    FEATURE_COLUMNS,
    GENRE_COLUMN,
    build_artifacts,
    load_model,
)
from src.predict import classify_dataframe

CSV_PATH: Path = DEFAULT_OUTPUT
DATASET_PATH = Path("data/dataset.csv")
N_RECOMMENDATIONS = 5
MAX_SEARCH_RESULTS = 50  # evita menus gigantescos em buscas muito genéricas
POPULAR_POOL_SIZE = 30  # vizinhos mais próximos entre os quais se escolhem os mais populares


# ------------------------------ dados -------------------------------- #
def _artifacts_are_fresh() -> bool:
    """Os ficheiros gerados têm de existir e ser mais recentes que o código e os dados."""
    generated = [CSV_PATH, DEFAULT_MODEL_PATH]
    sources = [Path("src/model.py"), DATASET_PATH]
    if not all(p.exists() for p in generated + sources):
        return False
    return min(p.stat().st_mtime for p in generated) >= max(p.stat().st_mtime for p in sources)


@st.cache_resource(show_spinner="A preparar o modelo (só na primeira execução)...")
def ensure_artifacts() -> None:
    """Gera o CSV com clusters e o modelo se faltarem ou estiverem desatualizados.

    Num deploy (Streamlit Cloud) estes ficheiros não vêm no Git. O cache_resource
    garante que só uma sessão executa o treino.
    """
    if _artifacts_are_fresh():
        return
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Falta `{DATASET_PATH}`. Coloque o CSV do Kaggle nessa pasta."
        )
    build_artifacts(DATASET_PATH, CSV_PATH, DEFAULT_MODEL_PATH)


@st.cache_data
def load_data(path: Path, mtime: float) -> pd.DataFrame:
    """Lê o CSV uma única vez. O 'mtime' faz parte da chave da cache, por isso
    a cache é invalidada automaticamente quando o modelo volta a ser treinado."""
    df = pd.read_csv(path)
    df["label"] = df["artists"] + " – " + df["track_name"]
    return df


@st.cache_resource
def get_model(path: Path, mtime: float):
    """Carrega o modelo uma só vez (também invalidado pelo 'mtime')."""
    return load_model(path)


# --------------------------- recomendação ---------------------------- #
def search_tracks(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Busca por nome da música (case-insensitive, correspondência parcial)."""
    mask = df["track_name"].str.contains(query, case=False, regex=False, na=False)
    return df[mask].head(MAX_SEARCH_RESULTS)


_BRACKETS = re.compile(r"\s*[\(\[].*?[\)\]]")


def base_title(name: str) -> str:
    """Título sem sufixos de versão: 'Song (feat. X) - Piano Version' -> 'song'."""
    base = _BRACKETS.sub("", name).split(" - ")[0].strip().lower()
    return base or name.strip().lower()


def version_key(artists: pd.Series, names: pd.Series) -> pd.Series:
    """Chave que identifica 'a mesma música': primeiro artista + título base."""
    first_artist = artists.astype(str).str.split(";").str[0].str.strip().str.lower()
    return first_artist + "|" + names.astype(str).map(base_title)


def recommend(
    df: pd.DataFrame,
    selected: pd.Series,
    n: int = N_RECOMMENDATIONS,
    same_genre: bool = False,
    weights: dict[str, float] | None = None,
    prefer_popular: bool = False,
) -> pd.DataFrame:
    """Retorna as n faixas do mesmo cluster mais próximas da selecionada.

    - `weights`: importância de cada feature na distância (omissos = 1).
      A distância é sqrt(sum(w * diff²)) sobre features padronizadas.
    - `prefer_popular`: entre as POPULAR_POOL_SIZE mais próximas, devolve as
      mais populares (requer a coluna 'popularity').
    Exclui outras versões da própria música e não repete versões.
    """
    mask = (df["cluster"] == selected["cluster"]) & (df["label"] != selected["label"])
    if same_genre and GENRE_COLUMN in df.columns:
        mask &= df[GENRE_COLUMN] == selected[GENRE_COLUMN]

    candidates = df[mask].copy()
    if candidates.empty:
        return candidates

    selected_key = version_key(
        pd.Series([selected["artists"]]), pd.Series([selected["track_name"]])
    ).iloc[0]
    candidates["_version_key"] = version_key(candidates["artists"], candidates["track_name"])
    candidates = candidates[candidates["_version_key"] != selected_key]
    if candidates.empty:
        return candidates.drop(columns="_version_key")

    # Pesos: omissos valem 1; se o utilizador zerar tudo, volta ao peso igual
    w = pd.Series(weights or {}, dtype=float).reindex(FEATURE_COLUMNS).fillna(1.0)
    if w.sum() <= 0:
        w[:] = 1.0

    std = df[FEATURE_COLUMNS].std().replace(0, 1)
    diff = (candidates[FEATURE_COLUMNS] - selected[FEATURE_COLUMNS].astype(float)) / std
    candidates["distance"] = np.sqrt((diff**2 * w).sum(axis=1))

    ranked = candidates.sort_values("distance").drop_duplicates(subset="_version_key")
    if prefer_popular and "popularity" in ranked.columns:
        ranked = ranked.head(POPULAR_POOL_SIZE).sort_values(
            "popularity", ascending=False, kind="stable"
        )
    return ranked.head(n).drop(columns="_version_key")


# ------------------------------ abas --------------------------------- #
def render_weights_sidebar() -> dict[str, float]:
    """Sliders com a importância de cada característica na recomendação."""
    st.sidebar.header("Importância das características")
    st.sidebar.caption("1 = normal · 0 = ignorar · 3 = o triplo. Só afeta a ordenação dentro do mood.")
    return {f: st.sidebar.slider(f, 0.0, 3.0, 1.0, 0.5) for f in FEATURE_COLUMNS}


def render_recommend_tab(df: pd.DataFrame, weights: dict[str, float]) -> None:
    has_genre = GENRE_COLUMN in df.columns
    has_popularity = "popularity" in df.columns

    query = st.text_input("Buscar música pelo nome:", placeholder="ex.: Yellow")
    same_genre = (
        st.checkbox("Restringir ao mesmo género", value=True) if has_genre else False
    )
    prefer_popular = (
        st.checkbox("Preferir faixas mais populares", value=False) if has_popularity else False
    )

    if not query.strip():
        st.info("Digite o nome de uma música para começar.")
        return

    matches = search_tracks(df, query.strip())
    if matches.empty:
        st.warning("Nenhuma música encontrada. Tente outro termo.")
        return

    chosen_index = st.selectbox(
        f"{len(matches)} resultado(s) — selecione a faixa:",
        options=matches.index,
        format_func=lambda i: df.at[i, "label"],
    )
    selected = df.loc[chosen_index]

    genre_info = f" · Género: **{selected[GENRE_COLUMN]}**" if has_genre else ""
    st.caption(f"Mood: **{selected['cluster_name']}**{genre_info}")

    recs = recommend(
        df, selected, same_genre=same_genre, weights=weights, prefer_popular=prefer_popular
    )
    if recs.empty:
        st.warning("Não há outras faixas com estes critérios. Tente desativar o filtro de género.")
        return

    st.subheader(f"{len(recs)} músicas do mesmo mood")
    if len(recs) < N_RECOMMENDATIONS:
        st.info("Poucas faixas com estes critérios; desative o filtro de género para ver mais.")

    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        genre = f" · _{row[GENRE_COLUMN]}_" if has_genre else ""
        pop = f" · popularidade {int(row['popularity'])}" if has_popularity else ""
        st.markdown(f"**{i}. {row['track_name']}** — {row['artists']}{genre}{pop}")

    cols = [
        "track_name", "artists",
        *([GENRE_COLUMN] if has_genre else []),
        *(["popularity"] if has_popularity else []),
        *FEATURE_COLUMNS,
    ]
    with st.expander("Ver características acústicas"):
        st.dataframe(recs[cols], hide_index=True)


def render_classify_tab() -> None:
    """Aba para atribuir um mood a faixas novas, usando o modelo guardado."""
    if not DEFAULT_MODEL_PATH.exists():
        st.error(f"Modelo `{DEFAULT_MODEL_PATH}` não encontrado. Recarregue a página.")
        return

    model = get_model(DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH.stat().st_mtime)
    st.write("Carregue um CSV com as colunas: " + ", ".join(f"`{c}`" for c in model.feature_columns))

    upload = st.file_uploader("CSV de faixas novas", type="csv")
    if upload is None:
        return

    try:
        result, dropped = classify_dataframe(model, pd.read_csv(upload))
    except KeyError as exc:
        st.error(exc.args[0])
        return
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"Não foi possível classificar o ficheiro: {exc}")
        return

    if dropped:
        st.warning(f"{dropped} linha(s) ignorada(s) por features nulas ou inválidas.")
    st.success(f"{len(result)} faixa(s) classificada(s).")

    st.bar_chart(result["cluster_name"].value_counts())

    identity = [c for c in ("track_name", "artists") if c in result.columns]
    st.dataframe(result[[*identity, "cluster_name", *model.feature_columns]], hide_index=True)

    st.download_button(
        "Descarregar resultado (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="faixas_classificadas.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Mood Recommender", page_icon="🎧")
    st.title("🎧 Recomendador Musical por Mood")

    try:
        ensure_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = load_data(CSV_PATH, CSV_PATH.stat().st_mtime)
    if "cluster_name" not in df.columns:
        st.error("O CSV é de uma versão antiga. Apague `data/clustered_tracks.csv` e recarregue.")
        st.stop()

    weights = render_weights_sidebar()
    tab_recommend, tab_classify = st.tabs(["Recomendar", "Classificar faixas novas"])
    with tab_recommend:
        render_recommend_tab(df, weights)
    with tab_classify:
        render_classify_tab()


if __name__ == "__main__":
    main()