"""Interface Streamlit: busca uma música e recomenda outras do mesmo cluster."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.model import DEFAULT_OUTPUT, FEATURE_COLUMNS, GENRE_COLUMN

CSV_PATH: Path = DEFAULT_OUTPUT
N_RECOMMENDATIONS = 5
MAX_SEARCH_RESULTS = 50  # evita menus gigantescos em buscas muito genéricas


@st.cache_data
def load_data(path: Path, mtime: float) -> pd.DataFrame:
    """Lê o CSV uma única vez. O 'mtime' faz parte da chave da cache, por isso
    a cache é invalidada automaticamente quando o modelo volta a ser treinado."""
    df = pd.read_csv(path)
    df["label"] = df["artists"] + " – " + df["track_name"]
    return df


def search_tracks(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Busca por nome da música (case-insensitive, correspondência parcial)."""
    mask = df["track_name"].str.contains(query, case=False, regex=False, na=False)
    return df[mask].head(MAX_SEARCH_RESULTS)


def recommend(
    df: pd.DataFrame,
    selected: pd.Series,
    n: int = N_RECOMMENDATIONS,
    same_genre: bool = False,
) -> pd.DataFrame:
    """Retorna as n faixas do mesmo cluster mais próximas da selecionada.

    Opcionalmente restringe ao mesmo género. A ordenação usa a distância
    euclidiana nas features padronizadas, para que as sugestões sejam
    coerentes e não aleatórias.
    """
    mask = (df["cluster"] == selected["cluster"]) & (df["label"] != selected["label"])
    if same_genre and GENRE_COLUMN in df.columns:
        mask &= df[GENRE_COLUMN] == selected[GENRE_COLUMN]

    candidates = df[mask].copy()
    if candidates.empty:
        return candidates

    std = df[FEATURE_COLUMNS].std().replace(0, 1)
    diff = (candidates[FEATURE_COLUMNS] - selected[FEATURE_COLUMNS].astype(float)) / std
    candidates["distance"] = np.sqrt((diff**2).sum(axis=1))

    # drop_duplicates evita sugerir várias versões da mesma faixa
    return (
        candidates.sort_values("distance")
        .drop_duplicates(subset="label")
        .head(n)
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


if __name__ == "__main__":
    main()