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
    load_model,
)
from src.predict import classify_dataframe

CSV_PATH: Path = DEFAULT_OUTPUT
N_RECOMMENDATIONS = 5
MAX_SEARCH_RESULTS = 50  # evita menus gigantescos em buscas muito genéricas


# ------------------------------ dados -------------------------------- #
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
) -> pd.DataFrame:
    """Retorna as n faixas do mesmo cluster mais próximas da selecionada.

    Opcionalmente restringe ao mesmo género. Ordena pela distância euclidiana
    nas features padronizadas. Exclui outras versões da própria música
    (piano, remaster, ao vivo...) e não repete versões entre as sugestões.
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

    std = df[FEATURE_COLUMNS].std().replace(0, 1)
    diff = (candidates[FEATURE_COLUMNS] - selected[FEATURE_COLUMNS].astype(float)) / std
    candidates["distance"] = np.sqrt((diff**2).sum(axis=1))

    return (
        candidates.sort_values("distance")
        .drop_duplicates(subset="_version_key")
        .head(n)
        .drop(columns="_version_key")
    )


# ------------------------------ abas --------------------------------- #
def render_recommend_tab(df: pd.DataFrame) -> None:
    has_genre = GENRE_COLUMN in df.columns

    query = st.text_input("Buscar música pelo nome:", placeholder="ex.: Yellow")
    same_genre = (
        st.checkbox("Restringir ao mesmo género", value=True) if has_genre else False
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

    recs = recommend(df, selected, same_genre=same_genre)
    if recs.empty:
        st.warning("Não há outras faixas com estes critérios. Tente desativar o filtro de género.")
        return

    st.subheader(f"{len(recs)} músicas do mesmo mood")
    if len(recs) < N_RECOMMENDATIONS:
        st.info("Poucas faixas com estes critérios; desative o filtro de género para ver mais.")

    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        genre = f" · _{row[GENRE_COLUMN]}_" if has_genre else ""
        st.markdown(f"**{i}. {row['track_name']}** — {row['artists']}{genre}")

    cols = ["track_name", "artists", *([GENRE_COLUMN] if has_genre else []), *FEATURE_COLUMNS]
    with st.expander("Ver características acústicas"):
        st.dataframe(recs[cols], hide_index=True)


def render_classify_tab() -> None:
    """Aba para atribuir um mood a faixas novas, usando o modelo guardado."""
    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Modelo `{DEFAULT_MODEL_PATH}` não encontrado. Execute antes: "
            "`python -m src.model --input-csv data/dataset.csv`"
        )
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

    if not CSV_PATH.exists():
        st.error(
            f"Arquivo `{CSV_PATH}` não encontrado. Execute antes: "
            "`python -m src.model --input-csv data/dataset.csv`"
        )
        st.stop()

    df = load_data(CSV_PATH, CSV_PATH.stat().st_mtime)
    if "cluster_name" not in df.columns:
        st.error("O CSV é de uma versão antiga. Volte a executar `python -m src.model`.")
        st.stop()

    tab_recommend, tab_classify = st.tabs(["Recomendar", "Classificar faixas novas"])
    with tab_recommend:
        render_recommend_tab(df)
    with tab_classify:
        render_classify_tab()


if __name__ == "__main__":
    main()