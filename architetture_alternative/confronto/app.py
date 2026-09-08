"""Dashboard di confronto diretto tra versione_1 (root, sempre attraverso il
Creative Agent) e versione_2 (router deterministico): token, tempo, costo,
qualità del giudice — con un confronto rigoroso "a parità di domanda" oltre
alle statistiche aggregate complessive."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import load_confronto, load_domande_comuni, load_v1, load_v2

st.set_page_config(page_title="Confronto v1 vs v2 — Dashboard", page_icon="⚖️", layout="wide")

st.title("⚖️ Confronto architetture: versione_1 vs versione_2")
st.caption(
    "versione_1 = ogni domanda passa sempre dal Creative Agent (Team coordinate). "
    "versione_2 = router deterministico che instrada direttamente a Graph/Semantic Agent o al "
    "workflow parallelo, usando il Creative Agent solo per i casi ambigui."
)

if st.sidebar.button("🔄 Ricarica dati"):
    st.cache_data.clear()

_COLOR_V1 = "#2a78d6"  # blu
_COLOR_V2 = "#1baf7a"  # acqua


@st.cache_data(ttl=10)
def get_data():
    return load_confronto(), load_domande_comuni()


df_all, df_comuni = get_data()

if df_all.empty:
    st.info("Nessun log trovato né in `tmp/query_log.csv` né nella tabella `query_log` di `versione_2/tmp/cinema_traces_v2.db`.")
    st.stop()

df = df_all[df_all["status"] == "COMPLETED"].copy()

# ═════════════════════════════════════════════════════════════════════════
# 1. KPI aggregati per architettura
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Panoramica aggregata")

_ARCH = ["versione_1", "versione_2"]
_ARCH_LABEL = {"versione_1": "Versione 1 (root)", "versione_2": "Versione 2 (router)"}
_ARCH_COLOR = {"versione_1": _COLOR_V1, "versione_2": _COLOR_V2}

cols = st.columns(2)
for col, arch in zip(cols, _ARCH):
    sub = df[df["architettura"] == arch]
    with col:
        st.markdown(f'<span style="color:{_ARCH_COLOR[arch]}">●</span> **{_ARCH_LABEL[arch]}**', unsafe_allow_html=True)
        if sub.empty:
            st.info("Nessuna domanda completata registrata.")
            continue
        no_cache = sub[sub["risposta_da_cache"] != "Sì"]
        no_outlier = no_cache[no_cache["durata_sec"].fillna(0) <= 180]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Domande completate", f"{len(sub)}")
        k2.metric("Token medi", f"{sub['total_tokens'].mean():,.0f}" if sub["total_tokens"].notna().any() else "—")
        k3.metric("Tempo medio", f"{no_outlier['durata_sec'].mean():.1f}s" if no_outlier["durata_sec"].notna().any() else "—")
        media_chiamate = sub["n_chiamate_llm"].dropna().mean() if sub["n_chiamate_llm"].notna().any() else None
        k4.metric("Chiamate LLM medie", f"{media_chiamate:.1f}" if media_chiamate is not None else "—")
        k5, k6, k7 = st.columns(3)
        costo_tot = sub["costo_stimato_usd"].sum()
        k5.metric("Costo stimato tot.", f"${costo_tot:.4f}" if pd.notna(costo_tot) else "—")
        pct_fallback = (sub["fallback_usato"] == "Sì").mean() * 100 if len(sub) else 0
        k6.metric("Fallback attivato", f"{pct_fallback:.0f}%")
        media_score = sub["score"].dropna().astype(float).mean() if sub["score"].notna().any() else None
        k7.metric("Punteggio giudice medio", f"{media_score:.1f}/10" if media_score is not None else "—")
        st.caption(
            f"Tempo/token esclusi outlier >180s{' e le risposte da cache' if arch == 'versione_1' else ''}. "
            "Le chiamate LLM contano i run di agenti/coordinator nella pipeline di risposta, non la "
            "valutazione del giudice in background (simmetrica tra le due architetture)."
        )

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 2. Confronto rigoroso sulle domande poste su ENTRAMBE le architetture
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Confronto diretto: stessa domanda, architetture diverse")
st.caption(
    "L'unico confronto davvero equo: stesse domande, stessi tool e stessi dati sottostanti — l'unica "
    "variabile è la strategia di instradamento. Preso solo dalle domande poste (testo identico) su entrambe le architetture."
)

if df_comuni.empty:
    st.info(
        "Nessuna domanda testualmente identica trovata su entrambe le architetture ancora. "
        "Esegui le stesse domande su versione_1 e versione_2 per popolare questo confronto."
    )
else:
    media_delta_token = df_comuni["delta_tokens_pct"].dropna().mean()
    media_delta_durata = df_comuni["delta_durata_pct"].dropna().mean()
    media_delta_chiamate = df_comuni["delta_chiamate"].dropna().mean()
    hk1, hk2, hk3, hk4 = st.columns(4)
    hk1.metric("Domande confrontabili", f"{len(df_comuni)}")
    hk2.metric(
        "Variazione media token (v2 vs v1)",
        f"{media_delta_token:+.0f}%" if pd.notna(media_delta_token) else "—",
        delta_color="inverse",
    )
    hk3.metric(
        "Variazione media tempo (v2 vs v1)",
        f"{media_delta_durata:+.0f}%" if pd.notna(media_delta_durata) else "—",
        delta_color="inverse",
    )
    hk4.metric(
        "Variazione media chiamate LLM (v2 vs v1)",
        f"{media_delta_chiamate:+.1f}" if pd.notna(media_delta_chiamate) else "—",
        delta_color="inverse",
    )
    st.caption(
        "Valori negativi = versione_2 usa meno token / risponde più in fretta / fa meno chiamate LLM "
        "della versione_1 sulla stessa identica domanda."
    )

    tabella = df_comuni.copy().sort_values("delta_durata_pct")
    tabella["Domanda"] = tabella["domanda"]
    tabella["Percorso (v2)"] = tabella["percorso_v2"]
    tabella["Chiamate LLM v1 → v2"] = tabella.apply(lambda r: f"{r['v1_chiamate']:.0f} → {r['v2_chiamate']:.0f}", axis=1)
    tabella["Token v1 → v2"] = tabella.apply(lambda r: f"{r['v1_tokens']:,.0f} → {r['v2_tokens']:,.0f}", axis=1)
    tabella["Δ token"] = tabella["delta_tokens_pct"].apply(lambda v: f"{v:+.0f}%" if pd.notna(v) else "—")
    tabella["Tempo v1 → v2"] = tabella.apply(lambda r: f"{r['v1_durata_sec']:.1f}s → {r['v2_durata_sec']:.1f}s", axis=1)
    tabella["Δ tempo"] = tabella["delta_durata_pct"].apply(lambda v: f"{v:+.0f}%" if pd.notna(v) else "—")
    st.dataframe(
        tabella[["Domanda", "Percorso (v2)", "Chiamate LLM v1 → v2", "Token v1 → v2", "Δ token", "Tempo v1 → v2", "Δ tempo"]],
        width="stretch", hide_index=True,
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 3. Composizione interna di ciascuna architettura
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Come si distribuiscono le domande all'interno di ciascuna architettura")
st.caption(
    "versione_1: quali agenti delega il coordinator per ogni domanda (sempre attraverso il Team). "
    "versione_2: quale percorso sceglie il router (spesso bypassando il Team del tutto)."
)


def _bar_composizione(sub: pd.DataFrame, color: str) -> go.Figure:
    counts = sub["gruppo"].value_counts()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color=color))
    fig.update_layout(xaxis_title="", yaxis_title="Numero di domande", yaxis=dict(dtick=1), margin=dict(t=10))
    return fig


comp_cols = st.columns(2)
for col, arch in zip(comp_cols, _ARCH):
    sub = df[df["architettura"] == arch]
    with col:
        st.markdown(f'<span style="color:{_ARCH_COLOR[arch]}">●</span> **{_ARCH_LABEL[arch]}**', unsafe_allow_html=True)
        if sub.empty:
            st.info("Nessun dato.")
            continue
        st.plotly_chart(_bar_composizione(sub, _ARCH_COLOR[arch]), width="stretch")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 4. Distribuzioni sovrapposte: token e tempo di risposta
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Distribuzioni a confronto (tutte le domande, non solo quelle comuni)")


def _nice_bin_width(raw: float) -> float:
    if raw <= 0:
        return 1000
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    nice = 1 if residual < 1.5 else 2 if residual < 3 else 5 if residual < 7 else 10
    return nice * magnitude


def _overlay_histogram(col_name: str, fmt, xaxis_title: str, exclude_cache: bool = False, exclude_outliers: bool = False):
    sub_v1 = df[df["architettura"] == "versione_1"].dropna(subset=[col_name])
    sub_v2 = df[df["architettura"] == "versione_2"].dropna(subset=[col_name])
    if exclude_cache:
        sub_v1 = sub_v1[sub_v1["risposta_da_cache"] != "Sì"]
    if exclude_outliers:
        sub_v1 = sub_v1[sub_v1[col_name] <= 180]
        sub_v2 = sub_v2[sub_v2[col_name] <= 180]

    if sub_v1.empty and sub_v2.empty:
        st.info("Nessun dato disponibile.")
        return

    max_val = max(sub_v1[col_name].max() if not sub_v1.empty else 0, sub_v2[col_name].max() if not sub_v2.empty else 0)
    bin_width = _nice_bin_width(max_val / 10)
    n_bins = int(max_val // bin_width) + 1
    edges = [i * bin_width for i in range(n_bins + 1)]
    labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(n_bins)]

    def _counts(sub: pd.DataFrame) -> list[int]:
        if sub.empty:
            return [0] * n_bins
        idx = (sub[col_name] / bin_width).clip(upper=n_bins - 0.001).astype(int)
        return [int((idx == i).sum()) for i in range(n_bins)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=_counts(sub_v1), name=_ARCH_LABEL["versione_1"], marker_color=_COLOR_V1))
    fig.add_trace(go.Bar(x=labels, y=_counts(sub_v2), name=_ARCH_LABEL["versione_2"], marker_color=_COLOR_V2))
    fig.update_layout(
        barmode="group", xaxis_title=xaxis_title, yaxis_title="Numero di domande",
        yaxis=dict(dtick=1), xaxis_tickangle=-45, margin=dict(t=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def _fmt_tokens(v: float) -> str:
    return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def _fmt_seconds(v: float) -> str:
    return f"{v/60:.0f}m" if v >= 60 else f"{v:.0f}s"


st.markdown("**Token per domanda**")
_overlay_histogram("total_tokens", _fmt_tokens, "Fascia di token")

st.markdown("**Tempo di risposta**")
st.caption("Escluse le risposte da cache (v1) e gli outlier oltre 180s (rate limit/blackout Gemini).")
_overlay_histogram("durata_sec", _fmt_seconds, "Fascia di tempo di risposta", exclude_cache=True, exclude_outliers=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 5. Qualità: punteggi del giudice a confronto
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Qualità delle risposte: punteggi del giudice a confronto")
st.caption(
    "Stesso giudice (AgentAsJudgeEval, stessi criteri) per entrambe le architetture: su versione_1 scatta "
    "sempre; su versione_2 scatta come hook nativo sul percorso 'ambiguous' e viene invocato esplicitamente "
    "sugli altri percorsi (vedi versione_2/main.py)."
)

score_cols = st.columns(2)
for col, arch in zip(score_cols, _ARCH):
    sub = df[(df["architettura"] == arch) & df["score"].notna()].copy()
    with col:
        st.markdown(f'<span style="color:{_ARCH_COLOR[arch]}">●</span> **{_ARCH_LABEL[arch]}**', unsafe_allow_html=True)
        if sub.empty:
            st.info("Nessuna valutazione disponibile.")
            continue
        sub["score"] = sub["score"].astype(int)
        mean_score = sub["score"].mean()
        counts_by_score = sub.groupby("score").size()
        xs = list(range(1, 11))
        counts = [int(counts_by_score.get(s, 0)) for s in xs]
        fig = go.Figure(go.Bar(x=xs, y=counts, marker_color=_ARCH_COLOR[arch]))
        fig.add_vline(x=mean_score, line_dash="dash", line_color="gray",
                      annotation_text=f"media: {mean_score:.1f}", annotation_position="top", annotation_yshift=10)
        fig.update_layout(
            xaxis_title="Punteggio (1-10)", yaxis_title="Numero di domande",
            yaxis=dict(dtick=1, range=[0, max(counts) * 1.3 if max(counts) else 1]),
            xaxis=dict(dtick=1, range=[0.5, 10.5]), margin=dict(t=40),
        )
        st.plotly_chart(fig, width="stretch")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 6. Tabella completa (entrambe le architetture)
# ═════════════════════════════════════════════════════════════════════════
st.subheader("Log completo (entrambe le architetture)")

_UNKNOWN_ARCH = "Tutte"
arch_filter = st.selectbox("Architettura", [_UNKNOWN_ARCH] + [_ARCH_LABEL[a] for a in _ARCH])
filtered = df.copy()
if arch_filter != _UNKNOWN_ARCH:
    arch_key = {v: k for k, v in _ARCH_LABEL.items()}[arch_filter]
    filtered = filtered[filtered["architettura"] == arch_key]

table_df = filtered.sort_values("timestamp", ascending=False).reset_index(drop=True).copy()
table_df["Data"] = table_df["timestamp"].dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d %H:%M")
table_df["Architettura"] = table_df["architettura"].map(_ARCH_LABEL)
table_df["Gruppo/Percorso"] = table_df["gruppo"]
table_df["Token"] = table_df["total_tokens"]
table_df["Tempo (s)"] = table_df["durata_sec"]
table_df["Valutazione"] = table_df["score"].apply(lambda s: f"{s:.0f}/10" if pd.notna(s) else "—")

st.dataframe(
    table_df[["Data", "Architettura", "Gruppo/Percorso", "domanda", "Token", "Tempo (s)", "Valutazione"]]
    .rename(columns={"domanda": "Domanda"}),
    width="stretch", hide_index=True,
)
