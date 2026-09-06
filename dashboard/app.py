"""Dashboard Streamlit: log delle query riuscite del Cinema Creative Team,
filtrabile per agenti coinvolti e modello usato."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import load_merged

st.set_page_config(page_title="Cinema Team — Dashboard", page_icon="🎬", layout="wide")

st.title("🎬 Cinema Creative Team — Log")
st.caption("Domande non fallite fatte al team, con risposta completa e motivazione del giudice.")

if st.sidebar.button("🔄 Ricarica dati"):
    st.cache_data.clear()


@st.cache_data(ttl=10)
def get_data() -> pd.DataFrame:
    return load_merged()


df = get_data()

if df.empty:
    st.info("Nessuna query registrata ancora in `tmp/query_log.csv`. Lancia qualche domanda dal team e ricarica.")
    st.stop()

# Solo le domande non fallite.
df = df[df["status"] == "COMPLETED"]

if df.empty:
    st.info("Nessuna domanda completata con successo trovata nel log.")
    st.stop()

# ── Helper condivisi (agenti coinvolti per riga) ─────────────────────────────
_COORDINATOR_LABEL = "(solo coordinator)"


def _row_agent_set(agenti: str) -> set[str]:
    """Il set di agenti DELEGATI di una riga, escluso il placeholder 'solo coordinator'
    (il Coordinator e' sempre presente in ogni run, quindi non e' un agente 'delegato')."""
    parti = {nome.strip() for nome in str(agenti).split(",")}
    parti.discard(_COORDINATOR_LABEL)
    return parti


# ── Istogrammi token per domanda, per combinazione di agenti ────────────────
st.subheader("Distribuzione dei token per domanda")


def _nice_bin_width(raw: float) -> float:
    """Arrotonda l'ampiezza di fascia a un numero 'tondo' (1/2/5 * 10^n)."""
    if raw <= 0:
        return 1000
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    nice = 1 if residual < 1.5 else 2 if residual < 3 else 5 if residual < 7 else 10
    return nice * magnitude


def _render_token_histogram(sub_df: pd.DataFrame, color: str, n_bins: int | None = None) -> int:
    """Disegna l'istogramma e restituisce il numero di fasce usate (cosi' i grafici
    successivi possono richiedere lo stesso numero, per restare confrontabili)."""
    tok_df = sub_df.dropna(subset=["total_tokens"]).copy()
    if tok_df.empty:
        st.info("Nessuna domanda in questa categoria.")
        return n_bins or 0

    max_tok = tok_df["total_tokens"].max()
    if n_bins is None:
        # Solo il primo grafico decide il numero di fasce (punta a circa 8).
        bin_width = _nice_bin_width(max_tok / 8)
        n_bins = int(max_tok // bin_width) + 1
    else:
        # Gli altri grafici usano lo STESSO numero di fasce del primo, ma scalato
        # sul proprio massimo (i range di token sono molto diversi tra categorie).
        bin_width = max_tok / n_bins if max_tok > 0 else 1
    edges = [i * bin_width for i in range(n_bins + 1)]

    tok_df["fascia_idx"] = (tok_df["total_tokens"] / bin_width).clip(upper=n_bins - 0.001).astype(int)

    def _fmt(v: float) -> str:
        return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"

    counts, labels, hover_texts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bucket = tok_df[tok_df["fascia_idx"] == i]
        counts.append(len(bucket))
        labels.append(f"{_fmt(lo)}–{_fmt(hi)}")
        domande = [f"• {str(d)[:80]}" for d in bucket["domanda"].tolist()]
        hover_texts.append("<br>".join(domande) if domande else "nessuna domanda")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            customdata=hover_texts,
            marker_color=color,
            hovertemplate="Fascia %{x} token<br>%{y} domande<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Fascia di token",
        yaxis_title="Numero di domande",
        yaxis=dict(dtick=1),
        xaxis_tickangle=-45,
        margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    return n_bins


# Colori categoriali (slot 1/2/3 della palette validata: passano tutti i controlli
# di accessibilita' CVD/contrasto per differenziazione a coppie e a triple).
_COLOR_GRAPH = "#2a78d6"      # blu
_COLOR_SEMANTIC = "#eb6834"   # arancione
_COLOR_BOTH = "#1baf7a"       # acqua

# Il Coordinator e' sempre presente in ogni run (delega o meno ai worker), quindi
# le tre categorie qui sotto si basano solo su QUALI agenti sono stati delegati.
graph_only_df = df[df["agenti_chiamati"].apply(lambda a: _row_agent_set(a) == {"Graph Query Agent"})]
semantic_only_df = df[df["agenti_chiamati"].apply(lambda a: _row_agent_set(a) == {"Semantic Query Agent"})]
both_df = df[df["agenti_chiamati"].apply(lambda a: _row_agent_set(a) == {"Graph Query Agent", "Semantic Query Agent"})]

hist_col1, hist_col2, hist_col3 = st.columns(3, gap="large", border=True)
with hist_col1:
    st.markdown(f'<span style="color:{_COLOR_GRAPH}">●</span> **Coordinator + Graph Query Agent**', unsafe_allow_html=True)
    n_bins_reference = _render_token_histogram(graph_only_df, _COLOR_GRAPH)
with hist_col2:
    st.markdown(f'<span style="color:{_COLOR_SEMANTIC}">●</span> **Coordinator + Semantic Query Agent**', unsafe_allow_html=True)
    _render_token_histogram(semantic_only_df, _COLOR_SEMANTIC, n_bins=n_bins_reference or None)
with hist_col3:
    st.markdown(f'<span style="color:{_COLOR_BOTH}">●</span> **Coordinator + entrambi gli agenti**', unsafe_allow_html=True)
    _render_token_histogram(both_df, _COLOR_BOTH, n_bins=n_bins_reference or None)

st.divider()

# ── Istogrammi delle valutazioni del giudice, per combinazione di agenti ────
st.subheader("Distribuzione delle valutazioni del giudice")


def _render_score_histogram(sub_df: pd.DataFrame, color: str) -> None:
    scored = sub_df.dropna(subset=["score"]).copy()
    if scored.empty:
        st.info("Nessuna valutazione disponibile in questa categoria.")
        return

    scored["score"] = scored["score"].astype(int)
    mean_score = scored["score"].mean()

    counts_by_score = scored.groupby("score").size()
    xs = list(range(1, 11))
    counts = [int(counts_by_score.get(s, 0)) for s in xs]
    hover_texts = []
    for s in xs:
        domande = [f"• {str(d)[:80]}" for d in scored[scored["score"] == s]["domanda"].tolist()]
        hover_texts.append("<br>".join(domande) if domande else "nessuna domanda")

    fig = go.Figure(
        go.Bar(
            x=xs,
            y=counts,
            customdata=hover_texts,
            marker_color=color,
            hovertemplate="Punteggio %{x}/10<br>%{y} domande<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"media: {mean_score:.1f}",
        annotation_position="top",
        annotation_yshift=10,
    )
    fig.update_layout(
        xaxis_title="Punteggio (1-10)",
        yaxis_title="Numero di domande",
        yaxis=dict(dtick=1, range=[0, max(counts) * 1.3 if max(counts) else 1]),
        xaxis=dict(dtick=1, range=[0.5, 10.5]),
        margin=dict(t=40),
    )
    st.plotly_chart(fig, use_container_width=True)


score_col1, score_col2, score_col3 = st.columns(3, gap="large", border=True)
with score_col1:
    st.markdown(f'<span style="color:{_COLOR_GRAPH}">●</span> **Coordinator + Graph Query Agent**', unsafe_allow_html=True)
    _render_score_histogram(graph_only_df, _COLOR_GRAPH)
with score_col2:
    st.markdown(f'<span style="color:{_COLOR_SEMANTIC}">●</span> **Coordinator + Semantic Query Agent**', unsafe_allow_html=True)
    _render_score_histogram(semantic_only_df, _COLOR_SEMANTIC)
with score_col3:
    st.markdown(f'<span style="color:{_COLOR_BOTH}">●</span> **Coordinator + entrambi gli agenti**', unsafe_allow_html=True)
    _render_score_histogram(both_df, _COLOR_BOTH)

st.divider()

# ── Filtri ────────────────────────────────────────────────────────────────
_TUTTI = "Tutti"

_UNKNOWN_MODEL = "(sconosciuto — query precedenti al tracciamento)"
model_series = df["model"].replace("", pd.NA).fillna(_UNKNOWN_MODEL)
model_options = sorted(model_series.unique().tolist())

st.write("**Agenti coinvolti** — corrispondenza esatta: seleziona una combinazione per vedere solo le domande gestite esattamente da quegli agenti (il Coordinator è sempre incluso).")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.checkbox("Coordinator", value=True, disabled=True)
with c2:
    show_graph = st.checkbox("Graph Query Agent", value=True)
with c3:
    show_semantic = st.checkbox("Semantic Query Agent", value=True)
with c4:
    selected_model = st.selectbox("Modello usato", [_TUTTI] + model_options)

selected_agent_set: set[str] = set()
if show_graph:
    selected_agent_set.add("Graph Query Agent")
if show_semantic:
    selected_agent_set.add("Semantic Query Agent")

filtered = df.copy()
filtered["_modello_normalizzato"] = model_series

# Mostra sempre le righe "solo coordinator" (set vuoto) + quelle che corrispondono
# esattamente alla combinazione di agenti selezionata (non "almeno uno", il set esatto).
filtered = filtered[
    filtered["agenti_chiamati"].apply(
        lambda agenti: _row_agent_set(agenti) == set() or _row_agent_set(agenti) == selected_agent_set
    )
]

if selected_model != _TUTTI:
    filtered = filtered[filtered["_modello_normalizzato"] == selected_model]

st.caption(f"{len(filtered)} domande mostrate su {len(df)} completate con successo in totale.")

# ── Log in formato tabella ────────────────────────────────────────────────
table_df = filtered.sort_values("timestamp", ascending=False).reset_index(drop=True).copy()
table_df["Data"] = table_df["timestamp"].dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d %H:%M")
table_df["Domanda"] = table_df["domanda"]
table_df["Valutazione"] = table_df["score"].apply(lambda s: f"{s:.0f}/10" if pd.notna(s) else "—")

st.caption("Clicca su una riga per vedere tutti i dettagli di quella domanda.")
selection_event = st.dataframe(
    table_df[["Data", "Domanda", "Valutazione"]],
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = selection_event.selection.rows if selection_event and selection_event.selection else []
if selected_rows:
    row = table_df.iloc[selected_rows[0]]
    st.divider()
    st.subheader("Dettaglio domanda selezionata")
    st.markdown(f"**Data:** {row['Data']}")
    st.markdown(f"**Domanda:** {row['domanda']}")
    st.markdown(f"**Agenti coinvolti:** {row['agenti_chiamati']}")
    fallback_badge = " · ⚠️ **fallback attivato** (risposta da Groq, Gemini non disponibile)" if row.get("fallback_usato") == "Sì" else ""
    st.markdown(f"**Modello:** {row['model'] or '—'} ({row['model_provider'] or '—'}){fallback_badge}")
    if pd.notna(row.get("total_tokens")):
        st.markdown(
            f"**Token:** {row['total_tokens']:,.0f} totali "
            f"({row['input_tokens']:,.0f} input / {row['output_tokens']:,.0f} output)"
            + (f" · **costo stimato:** ${row['costo_stimato_usd']:.5f}" if pd.notna(row.get("costo_stimato_usd")) else "")
            + (f" · **tempo:** {row['durata_sec']:.1f}s" if pd.notna(row.get("durata_sec")) else "")
        )
    st.markdown("**Risposta:**")
    st.markdown(str(row["risposta"]) if pd.notna(row["risposta"]) else "_(vuota)_")
    if pd.notna(row.get("reason")):
        st.markdown(f"**Motivazione del giudice ({row.get('score')}/10):** {row['reason']}")
