"""Interface Streamlit estilo player: recomendações por mood e classificação de faixas novas."""

import html
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
RECS_PER_ROW = 5

# Estilo: tema escuro com verde de acento. A capa de cada faixa é gerada a partir
# das próprias features (o dataset não tem imagens), por isso a cor "diz" o mood.
_CSS = """
<style>
.cover{aspect-ratio:1/1;border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-size:2.2rem;color:rgba(255,255,255,.8);box-shadow:0 8px 24px rgba(0,0,0,.5);}
.cover.big{border-radius:12px;font-size:4.5rem;}
.chip{display:inline-block;padding:2px 12px;margin:0 6px 6px 0;border-radius:999px;
  font-size:.8rem;background:rgba(255,255,255,.12);color:#fff;}
.chip.mood{background:#1DB954;color:#000;font-weight:700;}
.chip.explicit{background:#b3b3b3;color:#000;border-radius:3px;padding:0 8px;}
.t-title{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:.6rem;}
.t-sub{color:#b3b3b3;font-size:.85rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hero h1{font-size:2.6rem;line-height:1.1;margin:0 0 .4rem 0;padding:0;}
.hero .artist{font-size:1.1rem;font-weight:600;}
</style>
"""

# Referência a uma faixa: URI (spotify:track:ID) ou link (com ou sem /intl-xx/). IDs têm 22 caracteres.
_TRACK_REF = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/(?:intl-[a-z-]+/)?track/)([A-Za-z0-9]{22})"
)


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


# ---------------------------- apresentação --------------------------- #
def parse_track_id(text: str) -> str | None:
    """Extrai o ID de uma faixa a partir de um link ou URI do Spotify (ou None)."""
    match = _TRACK_REF.search(text.strip())
    return match.group(1) if match else None


def format_duration(ms) -> str:
    """Milissegundos para m:ss."""
    if pd.isna(ms):
        return "–"
    minutes, seconds = divmod(int(ms) // 1000, 60)
    return f"{minutes}:{seconds:02d}"


def esc(value) -> str:
    """Escapa texto do dataset antes de o pôr em HTML."""
    return "" if pd.isna(value) else html.escape(str(value))


def cover_html(row: pd.Series, big: bool = False) -> str:
    """Capa gerada: o matiz acompanha a valence (frio = triste, quente = alegre)
    e a saturação acompanha a energy."""
    valence = float(np.clip(row["valence"], 0, 1))
    energy = float(np.clip(row["energy"], 0, 1))
    hue = int(valence * 280)
    sat = int(35 + energy * 55)
    style = (
        f"background:linear-gradient(135deg,hsl({hue},{sat}%,45%),"
        f"hsl({(hue + 60) % 360},{sat}%,22%));"
    )
    return f'<div class="cover{" big" if big else ""}" style="{style}">♪</div>'


def _flag(value) -> bool:
    return bool(value) if pd.notna(value) else False


# ------------------------ estado e callbacks ------------------------- #
def _go(track_id: str) -> None:
    """Passa a explorar esta faixa. O 'nav' muda as chaves dos campos de busca, o
    que os limpa (evita estados antigos nos widgets)."""
    st.query_params["track"] = track_id
    st.session_state["nav"] = st.session_state.get("nav", 0) + 1


def _enqueue(track_id: str) -> None:
    queue = st.session_state.setdefault("queue", [])
    if track_id not in queue:
        queue.append(track_id)


def _dequeue(track_id: str) -> None:
    st.session_state["queue"] = [t for t in st.session_state.get("queue", []) if t != track_id]


def _play_from_queue(track_id: str) -> None:
    _dequeue(track_id)
    _go(track_id)


def _clear_queue() -> None:
    st.session_state["queue"] = []


# ------------------------------ ecrã --------------------------------- #
def render_sidebar(df: pd.DataFrame) -> tuple[dict, dict[str, float]]:
    """Opções de recomendação e pesos das características."""
    st.sidebar.title("🎧 Mood")
    has_genre = GENRE_COLUMN in df.columns
    has_popularity = "popularity" in df.columns
    opts = {
        "same_genre": st.sidebar.checkbox(
            "Restringir ao mesmo género", value=has_genre, disabled=not has_genre
        ),
        "prefer_popular": st.sidebar.checkbox(
            "Preferir faixas populares", value=False, disabled=not has_popularity
        ),
        "n": st.sidebar.select_slider(
            "Quantas recomendações", options=[5, 10, 15], value=N_RECOMMENDATIONS
        ),
        "show_player": st.sidebar.checkbox(
            "Mostrar o player do Spotify",
            value=True,
            help="Carrega o player oficial do Spotify, que precisa de ligação à internet.",
        ),
    }
    with st.sidebar.expander("Ajustar a vibe"):
        st.caption("1 = normal, 0 = ignorar, 3 = o triplo. Só muda a ordem dentro do mood.")
        weights = {f: st.slider(f, 0.0, 3.0, 1.0, 0.5) for f in FEATURE_COLUMNS}
    return opts, weights


def render_queue(df: pd.DataFrame) -> None:
    """Fila de faixas guardadas durante a sessão."""
    queue = st.session_state.get("queue", [])
    st.sidebar.subheader(f"Fila ({len(queue)})")
    if not queue:
        st.sidebar.caption("Vazia. Use ＋ numa recomendação para guardar a faixa.")
        return
    labels = df.drop_duplicates("track_id").set_index("track_id")["label"]
    for tid in queue:
        c_label, c_play, c_del = st.sidebar.columns([6, 1.4, 1.4])
        c_label.caption(labels.get(tid, tid))
        c_play.button("▶", key=f"qplay_{tid}", on_click=_play_from_queue, args=(tid,),
                      help="Explorar esta faixa e tirá-la da fila")
        c_del.button("✕", key=f"qdel_{tid}", on_click=_dequeue, args=(tid,),
                     help="Remover da fila")
    st.sidebar.button("Limpar fila", on_click=_clear_queue)


def render_cards(rows: pd.DataFrame, section: str) -> None:
    """Grelha de faixas: capa, título, artista e botões Explorar (▶) e Fila (＋)."""
    for start in range(0, len(rows), RECS_PER_ROW):
        chunk = rows.iloc[start : start + RECS_PER_ROW]
        for col, (_, r) in zip(st.columns(RECS_PER_ROW), chunk.iterrows()):
            tid = r["track_id"]
            with col, st.container(border=True):
                st.markdown(
                    cover_html(r)
                    + f'<div class="t-title" title="{esc(r["track_name"])}">{esc(r["track_name"])}</div>'
                    + f'<div class="t-sub">{esc(r["artists"]).replace(";", ", ")}</div>',
                    unsafe_allow_html=True,
                )
                meta = [f"⏱ {format_duration(r.get('duration_ms'))}"]
                if pd.notna(r.get("popularity")):
                    meta.insert(0, f"🔥 {int(r['popularity'])}")
                st.caption("  ".join(meta))
                b_play, b_queue = st.columns(2)
                b_play.button("▶", key=f"{section}_play_{tid}", on_click=_go, args=(tid,),
                              help="Explorar a partir desta faixa")
                b_queue.button("＋", key=f"{section}_queue_{tid}", on_click=_enqueue, args=(tid,),
                               help="Adicionar à fila")


def embed_player(track_id: str) -> None:
    """Player oficial do Spotify (iframe). Usa st.iframe e, em versões antigas do
    Streamlit que ainda não o têm, o componente equivalente."""
    src = f"https://open.spotify.com/embed/track/{track_id}?theme=0"
    if hasattr(st, "iframe"):
        st.iframe(src, height=152)
    else:
        import streamlit.components.v1 as components

        components.iframe(src, height=152)


def render_hero(track: pd.Series, show_player: bool) -> None:
    """Faixa em foco: capa grande, dados e player."""
    col_cover, col_info = st.columns([1, 3], gap="large")
    with col_cover:
        st.markdown(cover_html(track, big=True), unsafe_allow_html=True)
    with col_info:
        chips = f'<span class="chip mood">{esc(track["cluster_name"])}</span>'
        if GENRE_COLUMN in track.index:
            chips += f'<span class="chip">{esc(track[GENRE_COLUMN])}</span>'
        if _flag(track.get("explicit")):
            chips += '<span class="chip explicit">Explícito</span>'
        album = f'<div class="t-sub">{esc(track.get("album_name"))}</div>' if "album_name" in track.index else ""
        st.markdown(
            f'<div class="hero"><h1>{esc(track["track_name"])}</h1>'
            f'<div class="artist">{esc(track["artists"]).replace(";", ", ")}</div>{album}'
            f'<div style="margin-top:.8rem">{chips}</div></div>',
            unsafe_allow_html=True,
        )
        facts = [f"Duração {format_duration(track.get('duration_ms'))}"]
        if pd.notna(track.get("popularity")):
            facts.append(f"Popularidade {int(track['popularity'])}")
        st.caption("  ·  ".join(facts))
    if show_player:
        embed_player(track["track_id"])


def render_trending(df: pd.DataFrame) -> None:
    """Ecrã inicial: as faixas mais populares como ponto de partida."""
    if "popularity" not in df.columns:
        st.info("Pesquise uma música para começar.")
        return
    st.subheader("Para começar")
    st.caption("As faixas mais populares do dataset. Escolha ▶ ou pesquise acima.")
    render_cards(df.nlargest(RECS_PER_ROW, "popularity"), "start")


def _apply_query(df: pd.DataFrame, query: str, nav: int) -> None:
    """Procura por nome ou resolve um link/URI do Spotify; a escolha vai para o URL."""
    linked = parse_track_id(query)
    if linked:
        if (df["track_id"] == linked).any():
            st.query_params["track"] = linked
        else:
            st.warning("Esta faixa não está no dataset da app. Experimente pesquisar pelo nome.")
        return

    matches = search_tracks(df, query)
    if matches.empty:
        st.warning("Nenhuma música encontrada. Tente outro termo.")
        return
    labels = dict(zip(matches["track_id"], matches["label"]))
    # A chave inclui a busca: ao mudar o texto, o menu recomeça vazio
    pick = st.selectbox(
        f"{len(matches)} resultado(s)",
        options=list(labels),
        format_func=labels.get,
        index=None,
        placeholder="Escolha uma faixa",
        key=f"pick_{nav}_{query.lower()}",
    )
    if pick:
        st.query_params["track"] = pick


def render_discover_tab(df: pd.DataFrame, weights: dict[str, float], opts: dict) -> None:
    nav = st.session_state.get("nav", 0)
    query = st.text_input(
        "Buscar música ou colar um link do Spotify",
        key=f"query_{nav}",
        placeholder="Yellow, ou https://open.spotify.com/track/...",
    ).strip()
    if query:
        _apply_query(df, query, nav)

    track_id = st.query_params.get("track")
    found = df[df["track_id"] == track_id] if track_id else df.iloc[0:0]
    if found.empty:
        if track_id:
            st.warning("Esta faixa não está no dataset da app. Pesquise outra.")
        render_trending(df)
        return

    selected = found.iloc[0]
    render_hero(selected, opts["show_player"])

    recs = recommend(
        df, selected, n=opts["n"], same_genre=opts["same_genre"],
        weights=weights, prefer_popular=opts["prefer_popular"],
    )
    st.subheader("Para continuar na mesma vibe")
    if recs.empty:
        st.warning("Não há outras faixas com estes critérios. Tente desativar o filtro de género.")
        return
    st.caption(f"Faixas do mood {selected['cluster_name']}, da mais parecida para a menos.")
    if len(recs) < opts["n"]:
        st.info("Poucas faixas com estes critérios; desative o filtro de género para ver mais.")
    render_cards(recs, "rec")

    has_genre = GENRE_COLUMN in df.columns
    cols = [
        "track_name", "artists",
        *([GENRE_COLUMN] if has_genre else []),
        *(["popularity"] if "popularity" in recs.columns else []),
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
    st.set_page_config(page_title="Mood Recommender", page_icon="🎧", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        ensure_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = load_data(CSV_PATH, CSV_PATH.stat().st_mtime)
    if "cluster_name" not in df.columns:
        st.error("O CSV é de uma versão antiga. Apague `data/clustered_tracks.csv` e recarregue.")
        st.stop()

    opts, weights = render_sidebar(df)
    render_queue(df)

    tab_discover, tab_classify = st.tabs(["Descobrir", "Classificar faixas novas"])
    with tab_discover:
        render_discover_tab(df, weights, opts)
    with tab_classify:
        render_classify_tab()


if __name__ == "__main__":
    main()
