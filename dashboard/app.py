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

# ── Istogramma token per domanda ────────────────────────────────────────────
st.subheader("Distribuzione dei token per domanda")
tok_df = df.dropna(subset=["total_tokens"]).copy()
if tok_df.empty:
    st.info("Nessun dato sui token disponibile.")
else:
    def _nice_bin_width(raw: float) -> float:
        """Arrotonda l'ampiezza di fascia a un numero 'tondo' (1/2/5 * 10^n)."""
        if raw <= 0:
            return 1000
        magnitude = 10 ** math.floor(math.log10(raw))
        residual = raw / magnitude
        nice = 1 if residual < 1.5 else 2 if residual < 3 else 5 if residual < 7 else 10
        return nice * magnitude

    max_tok = tok_df["total_tokens"].max()
    bin_width = _nice_bin_width(max_tok / 8)  # punta a circa 8 fasce
    n_bins = int(max_tok // bin_width) + 1
    edges = [i * bin_width for i in range(n_bins + 1)]

    tok_df["fascia_idx"] = (tok_df["total_tokens"] // bin_width).astype(int)

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
            hovertemplate="Fascia %{x} token<br>%{y} domande<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Fascia di token", yaxis_title="Numero di domande", yaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Filtri ────────────────────────────────────────────────────────────────
_TUTTI = "Tutti"
_COORDINATOR_LABEL = "(solo coordinator)"


def _row_agent_set(agenti: str) -> set[str]:
    """Il set di agenti DELEGATI di una riga, escluso il placeholder 'solo coordinator'."""
    parti = {nome.strip() for nome in str(agenti).split(",")}
    parti.discard(_COORDINATOR_LABEL)
    return parti


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
table_df["Data"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
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
    st.markdown(f"**Modello:** {row['model'] or '—'} ({row['model_provider'] or '—'})")
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
