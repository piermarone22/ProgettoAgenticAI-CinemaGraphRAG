"""Dashboard Streamlit per l'architettura a router deterministico (versione_2):
log delle query, con enfasi su quanto spesso il router riesce a evitare il
percorso costoso (Creative Agent / Team coordinate) rispetto ai percorsi
diretti (graph/semantic) o al workflow deterministico (pitch).

Adattata da dashboard/app.py (root): stessa struttura a istogrammi tripli e
stessa logica di binning, ma raggruppata per PERCORSO scelto dal router invece
che per combinazione di agenti delegati (qui non serve: ogni riga appartiene
a un solo percorso, deciso deterministicamente prima di ogni chiamata LLM)."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import load_merged

st.set_page_config(page_title="Router (v2) — Dashboard", page_icon="🧭", layout="wide")

st.title("🧭 Cinema GraphRAG — Router deterministico (versione_2)")
st.caption(
    "Domande instradate dal router (graph / semantic / pitch / ambiguous) — confronta con la "
    "dashboard dell'architettura originale per valutare il risparmio di token e tempo."
)

if st.sidebar.button("🔄 Ricarica dati"):
    st.cache_data.clear()


@st.cache_data(ttl=10)
def get_data() -> pd.DataFrame:
    return load_merged()


df_raw = get_data()

if df_raw.empty:
    st.info("Nessuna query registrata ancora nella tabella `query_log` di `versione_2/tmp/cinema_traces_v2.db`. Lancia qualche domanda su /query e ricarica.")
    st.stop()

# ── Percorsi possibili e colori (stesse tonalità/logica di accessibilità della
# dashboard originale, + una quarta per 'ambiguous') ─────────────────────────
_PERCORSI_ORDINE = ["graph", "semantic", "pitch", "ambiguous"]
_PERCORSO_LABEL = {
    "graph": "Graph Query Agent (diretto)",
    "semantic": "Semantic Query Agent (diretto)",
    "pitch": "Workflow parallelo (pitch)",
    "ambiguous": "Creative Agent (fallback)",
}
_PERCORSO_COLOR = {
    "graph": "#2a78d6",      # blu
    "semantic": "#eb6834",   # arancione
    "pitch": "#1baf7a",      # acqua
    "ambiguous": "#8a8a8a",  # grigio — è il percorso "costoso" di fallback
}

n_bloccate = int((df_raw["status"] == "BLOCKED").sum())
n_errori = int((~df_raw["status"].isin(["COMPLETED", "BLOCKED"])).sum())

df = df_raw[df_raw["status"] == "COMPLETED"]
if df.empty:
    st.info("Nessuna domanda completata con successo trovata nel log.")
    st.stop()

# ── KPI di sintesi: quanto spesso il router evita il percorso costoso ───────
st.subheader("Distribuzione dei percorsi scelti dal router")
kpi_cols = st.columns(len(_PERCORSI_ORDINE) + 1)
totale_completate = len(df)
for col, percorso in zip(kpi_cols, _PERCORSI_ORDINE):
    n = int((df["percorso"] == percorso).sum())
    pct = (n / totale_completate * 100) if totale_completate else 0
    col.metric(_PERCORSO_LABEL[percorso], f"{n}", f"{pct:.0f}% del totale")
extra_col = kpi_cols[-1]
extra_col.metric("Bloccate / errori", f"{n_bloccate + n_errori}", f"{n_bloccate} injection, {n_errori} altri")

st.caption(
    "Più alta è la quota di 'graph', 'semantic' e 'pitch' rispetto ad 'ambiguous', più il router sta "
    "evitando con successo il costo di orchestrazione del Creative Agent — l'obiettivo dell'esperimento."
)

st.divider()


# ── Helper condivisi per il binning (identici a dashboard/app.py in root) ───
def _nice_bin_width(raw: float) -> float:
    if raw <= 0:
        return 1000
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    nice = 1 if residual < 1.5 else 2 if residual < 3 else 5 if residual < 7 else 10
    return nice * magnitude


def _render_histogram(sub_df: pd.DataFrame, colonna: str, color: str, xaxis_title: str,
                       fmt, n_bins: int | None = None) -> int:
    val_df = sub_df.dropna(subset=[colonna]).copy()
    if val_df.empty:
        st.info("Nessuna domanda in questa categoria.")
        return n_bins or 0

    max_val = val_df[colonna].max()
    if n_bins is None:
        bin_width = _nice_bin_width(max_val / 8)
        n_bins = int(max_val // bin_width) + 1
    else:
        bin_width = max_val / n_bins if max_val > 0 else 1
    edges = [i * bin_width for i in range(n_bins + 1)]

    val_df["fascia_idx"] = (val_df[colonna] / bin_width).clip(upper=n_bins - 0.001).astype(int)

    counts, labels, hover_texts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bucket = val_df[val_df["fascia_idx"] == i]
        counts.append(len(bucket))
        labels.append(f"{fmt(lo)}–{fmt(hi)}")
        domande = [f"• {str(d)[:80]}" for d in bucket["domanda"].tolist()]
        hover_texts.append("<br>".join(domande) if domande else "nessuna domanda")

    fig = go.Figure(
        go.Bar(
            x=labels, y=counts, customdata=hover_texts, marker_color=color,
            hovertemplate=f"Fascia %{{x}}<br>%{{y}} domande<br>%{{customdata}}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title="Numero di domande",
        yaxis=dict(dtick=1), xaxis_tickangle=-45, margin=dict(t=10),
    )
    st.plotly_chart(fig, width="stretch")
    return n_bins


def _fmt_tokens(v: float) -> str:
    return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def _fmt_seconds(v: float) -> str:
    return f"{v/60:.0f}m" if v >= 60 else f"{v:.0f}s"


_percorsi_presenti = [p for p in _PERCORSI_ORDINE if (df["percorso"] == p).any()]
_sub_df_per_percorso = {p: df[df["percorso"] == p] for p in _percorsi_presenti}

# ── Istogrammi token per domanda, per percorso ───────────────────────────────
st.subheader("Distribuzione dei token per domanda")
cols = st.columns(len(_percorsi_presenti), gap="large", border=True)
n_bins_reference = None
for col, percorso in zip(cols, _percorsi_presenti):
    with col:
        color = _PERCORSO_COLOR[percorso]
        st.markdown(f'<span style="color:{color}">●</span> **{_PERCORSO_LABEL[percorso]}**', unsafe_allow_html=True)
        n_bins = _render_histogram(
            _sub_df_per_percorso[percorso], "total_tokens", color, "Fascia di token", _fmt_tokens,
            n_bins=n_bins_reference,
        )
        if n_bins_reference is None:
            n_bins_reference = n_bins or None

st.divider()

# ── Istogrammi tempo di risposta, per percorso ──────────────────────────────
st.subheader("Distribuzione dei tempi di risposta")
st.caption(
    "Escluse le domande che hanno impiegato più di 3 minuti (outlier dovuti a rate limit/blackout "
    "del servizio Gemini, non rappresentativi del comportamento normale)."
)
_DURATA_OUTLIER_THRESHOLD_SEC = 180
_no_outlier_df = {p: sub[sub["durata_sec"].fillna(0) <= _DURATA_OUTLIER_THRESHOLD_SEC] for p, sub in _sub_df_per_percorso.items()}

cols = st.columns(len(_percorsi_presenti), gap="large", border=True)
dur_n_bins_reference = None
for col, percorso in zip(cols, _percorsi_presenti):
    with col:
        color = _PERCORSO_COLOR[percorso]
        st.markdown(f'<span style="color:{color}">●</span> **{_PERCORSO_LABEL[percorso]}**', unsafe_allow_html=True)
        n_bins = _render_histogram(
            _no_outlier_df[percorso], "durata_sec", color, "Fascia di tempo di risposta", _fmt_seconds,
            n_bins=dur_n_bins_reference,
        )
        if dur_n_bins_reference is None:
            dur_n_bins_reference = n_bins or None

st.divider()

# ── Istogrammi delle valutazioni del giudice, per percorso ──────────────────
st.subheader("Distribuzione delle valutazioni del giudice")
st.caption(
    "Il giudice (AgentAsJudgeEval) valuta ogni percorso: sul percorso 'ambiguous' scatta automaticamente "
    "come post_hook nativo di cinema_team; su 'graph'/'semantic'/'pitch', che bypassano il Team apposta "
    "per risparmiare il costo di orchestrazione, viene invocato esplicitamente (stesso giudice, stesso "
    "db) — un costo extra accettato per non perdere la copertura di qualità sui percorsi diretti."
)


def _render_score_histogram(sub_df: pd.DataFrame, color: str) -> None:
    scored = sub_df.dropna(subset=["score"]).copy()
    if scored.empty:
        st.info("Nessuna valutazione disponibile per questo percorso.")
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
        go.Bar(x=xs, y=counts, customdata=hover_texts, marker_color=color,
               hovertemplate="Punteggio %{x}/10<br>%{y} domande<br>%{customdata}<extra></extra>")
    )
    fig.add_vline(x=mean_score, line_dash="dash", line_color="gray",
                  annotation_text=f"media: {mean_score:.1f}", annotation_position="top", annotation_yshift=10)
    fig.update_layout(
        xaxis_title="Punteggio (1-10)", yaxis_title="Numero di domande",
        yaxis=dict(dtick=1, range=[0, max(counts) * 1.3 if max(counts) else 1]),
        xaxis=dict(dtick=1, range=[0.5, 10.5]), margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch")


cols = st.columns(len(_percorsi_presenti), gap="large", border=True)
for col, percorso in zip(cols, _percorsi_presenti):
    with col:
        color = _PERCORSO_COLOR[percorso]
        st.markdown(f'<span style="color:{color}">●</span> **{_PERCORSO_LABEL[percorso]}**', unsafe_allow_html=True)
        _render_score_histogram(_sub_df_per_percorso[percorso], color)

st.divider()

# ── Filtri ────────────────────────────────────────────────────────────────
_TUTTI = "Tutti"

_UNKNOWN_MODEL = "(sconosciuto)"
model_series = df["model"].replace("", pd.NA).fillna(_UNKNOWN_MODEL) if "model" in df.columns else pd.Series([_UNKNOWN_MODEL] * len(df), index=df.index)
model_options = sorted(model_series.unique().tolist())

st.write("**Percorso** — deseleziona tutti per mostrare ogni domanda, indipendentemente dal percorso.")
percorso_cols = st.columns(len(_PERCORSI_ORDINE) + 1)
percorsi_selezionati = set()
for col, percorso in zip(percorso_cols, _PERCORSI_ORDINE):
    if col.checkbox(_PERCORSO_LABEL[percorso], value=True, key=f"chk_{percorso}"):
        percorsi_selezionati.add(percorso)
selected_model = percorso_cols[-1].selectbox("Modello usato", [_TUTTI] + model_options)

filtered = df.copy()
filtered["_modello_normalizzato"] = model_series

if percorsi_selezionati:
    filtered = filtered[filtered["percorso"].isin(percorsi_selezionati)]
# nessun percorso selezionato -> nessun filtro, mostra tutto (stessa scelta UX della dashboard originale)

if selected_model != _TUTTI:
    filtered = filtered[filtered["_modello_normalizzato"] == selected_model]

st.caption(f"{len(filtered)} domande mostrate su {len(df)} completate con successo in totale.")

# ── Log in formato tabella ────────────────────────────────────────────────
table_df = filtered.sort_values("timestamp", ascending=False).reset_index(drop=True).copy()
table_df["Data"] = table_df["timestamp"].dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d %H:%M")
table_df["Domanda"] = table_df["domanda"]
table_df["Percorso"] = table_df["percorso"].map(_PERCORSO_LABEL).fillna(table_df["percorso"])
table_df["Valutazione"] = table_df["score"].apply(lambda s: f"{s:.0f}/10" if pd.notna(s) else "—")

st.caption("Clicca su una riga per vedere tutti i dettagli di quella domanda.")
selection_event = st.dataframe(
    table_df[["Data", "Percorso", "Domanda", "Valutazione"]],
    width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
)

selected_rows = selection_event.selection.rows if selection_event and selection_event.selection else []
if selected_rows:
    row = table_df.iloc[selected_rows[0]]
    st.divider()
    st.subheader("Dettaglio domanda selezionata")
    st.markdown(f"**Data:** {row['Data']}")
    st.markdown(f"**Domanda:** {row['domanda']}")
    st.markdown(f"**Percorso:** {row['Percorso']}")
    st.markdown(f"**Agenti coinvolti:** {row['agenti_coinvolti']}")
    fallback_badge = " · ⚠️ **fallback attivato** (Groq, Gemini non disponibile su almeno una chiamata)" if row.get("fallback_usato") == "Sì" else ""
    st.markdown(f"**Modello:** {row.get('model') or '—'} ({row.get('model_provider') or '—'}){fallback_badge}")
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
    else:
        st.caption(
            "Nessuna valutazione ancora disponibile per questa riga: il giudice viene invocato in "
            "background (non blocca la risposta) e potrebbe non aver ancora scritto il risultato — "
            "ricarica i dati tra qualche secondo."
        )
