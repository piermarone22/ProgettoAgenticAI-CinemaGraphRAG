"""Caricamento unificato dei log di versione_1 e versione_2 per la dashboard di
confronto. Duplica la logica di caricamento/join con gli eval invece di
importare dashboard/data.py e versione_2/dashboard/data.py (stesso nome di
modulo in path diversi, import simultaneo non praticabile).
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

# .parent.parent.parent: questo file vive in architetture_alternative/confronto/,
# due livelli sotto la vera radice del progetto.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
V1_LOG_PATH = ROOT_DIR / "tmp" / "query_log.csv"
V2_LOG_DB_PATH = ROOT_DIR / "architetture_alternative" / "versione_2" / "tmp" / "cinema_traces_v2.db"
# Db degli score del giudice: resta quello di versione_1, perché quality_eval
# ha il proprio db cablato alla costruzione (app/agent.py) e lo usa per
# entrambe le architetture — vedi versione_2/dashboard/data.py per la nota
# sul rischio di join tra domande testualmente identiche poste a ridosso nel
# tempo su architetture diverse.
TRACES_DB_PATH = ROOT_DIR / "tmp" / "cinema_traces.db"

_COLONNE_COMUNI = [
    "timestamp", "architettura", "domanda", "gruppo", "risposta", "status",
    "model", "model_provider", "fallback_usato", "risposta_da_cache",
    "input_tokens", "output_tokens", "total_tokens", "costo_stimato_usd",
    "durata_sec", "score", "passed", "n_chiamate_llm",
]


def _conta_agenti(campo) -> int:
    """Numero di agenti nominati nel campo 'agenti_chiamati'/'agenti_coinvolti'
    (esclusi i placeholder per 'nessun worker delegato')."""
    if pd.isna(campo):
        return 0
    parti = [p.strip() for p in str(campo).split(",")]
    parti = [p for p in parti if p and p not in ("(solo coordinator)", "(nessuno)")]
    return len(parti)


def _n_chiamate_v1(agenti_chiamati) -> int:
    """v1: il coordinator è sempre coinvolto (+1), più un run per worker delegato."""
    return 1 + _conta_agenti(agenti_chiamati)


def _n_chiamate_v2(percorso, agenti_coinvolti) -> int:
    """v2: nessun +1 di coordinator sui percorsi diretti (il router non è un
    LLM); sul percorso 'ambiguous' la struttura è identica a v1."""
    n_agenti = _conta_agenti(agenti_coinvolti)
    if percorso == "ambiguous":
        return 1 + n_agenti
    return n_agenti


def _load_eval_scores() -> pd.DataFrame:
    if not TRACES_DB_PATH.exists():
        return pd.DataFrame(columns=["input", "score", "passed", "eval_created_at"])
    conn = sqlite3.connect(TRACES_DB_PATH)
    try:
        cur = conn.execute("SELECT eval_data, created_at FROM agno_eval_runs WHERE eval_type = 'agent_as_judge'")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["input", "score", "passed", "eval_created_at"])
    finally:
        conn.close()

    records = []
    for eval_data_raw, created_at in rows:
        try:
            eval_data = json.loads(eval_data_raw)
        except (TypeError, json.JSONDecodeError):
            continue
        results = eval_data.get("results") or []
        if not results:
            continue
        result = results[0]
        records.append({
            "input": result.get("input"),
            "score": result.get("score"),
            "passed": result.get("passed"),
            "eval_created_at": pd.to_datetime(int(created_at), unit="s", utc=True) if created_at else pd.NaT,
        })
    return pd.DataFrame.from_records(records)


_eval_scores_cache: pd.DataFrame | None = None


def _get_eval_scores() -> pd.DataFrame:
    global _eval_scores_cache
    if _eval_scores_cache is None:
        _eval_scores_cache = _load_eval_scores()
    return _eval_scores_cache


def _merge_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = pd.NA
    df["passed"] = pd.NA
    eval_scores = _get_eval_scores()
    if eval_scores.empty or df.empty:
        return df

    completed = df[df["status"] == "COMPLETED"]
    used_indices: set = set()
    for _, ev in eval_scores.sort_values("eval_created_at").iterrows():
        candidates = completed[
            (completed["domanda"] == ev["input"])
            & (~completed.index.isin(used_indices))
            & (completed["timestamp"] <= ev["eval_created_at"])
        ]
        if candidates.empty:
            continue
        best_idx = (ev["eval_created_at"] - candidates["timestamp"]).idxmin()
        used_indices.add(best_idx)
        df.loc[best_idx, ["score", "passed"]] = [ev["score"], ev["passed"]]
    return df


def _v1_gruppo(agenti: str) -> str:
    parti = {p.strip() for p in str(agenti).split(",")}
    parti.discard("(solo coordinator)")
    if parti == set():
        return "solo coordinator"
    if parti == {"Graph Query Agent"}:
        return "graph"
    if parti == {"Semantic Query Agent"}:
        return "semantic"
    if parti == {"Graph Query Agent", "Semantic Query Agent"}:
        return "entrambi (pitch)"
    return "altro"


def load_v1() -> pd.DataFrame:
    if not V1_LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(V1_LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["architettura"] = "versione_1"
    df["gruppo"] = df["agenti_chiamati"].fillna("(solo coordinator)").apply(_v1_gruppo)
    df["n_chiamate_llm"] = df["agenti_chiamati"].apply(_n_chiamate_v1)
    if "risposta_da_cache" not in df.columns:
        df["risposta_da_cache"] = "No"
    return _merge_scores(df)


def load_v2() -> pd.DataFrame:
    if not V2_LOG_DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(V2_LOG_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM query_log ORDER BY id", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame()
    finally:
        conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["architettura"] = "versione_2"  # normalizzato (il CSV sorgente usa "versione_2_router")
    df["gruppo"] = df["percorso"]
    df["n_chiamate_llm"] = df.apply(lambda r: _n_chiamate_v2(r["percorso"], r["agenti_coinvolti"]), axis=1)
    df["risposta_da_cache"] = "No"  # versione_2 non implementa cache: nessuna riga e' mai da cache
    return _merge_scores(df)


def load_confronto() -> pd.DataFrame:
    """Concatena i due log in un DataFrame con schema comune per confronti
    aggregati. Il campo 'gruppo' non è comparabile 1:1 tra le architetture."""
    frames = []
    for df in (load_v1(), load_v2()):
        if df.empty:
            continue
        for c in _COLONNE_COMUNI:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[_COLONNE_COMUNI])
    if not frames:
        return pd.DataFrame(columns=_COLONNE_COMUNI)
    return pd.concat(frames, ignore_index=True)


def load_domande_comuni() -> pd.DataFrame:
    """Confronto sulle domande poste (testo identico) su entrambe le
    architetture: prende l'esecuzione genuina più recente per ciascuna
    (escluse le risposte da cache di v1, che avrebbero durata falsamente
    vicina a zero)."""
    v1 = load_v1()
    v2 = load_v2()
    if v1.empty or v2.empty:
        return pd.DataFrame()

    v1_genuine = v1[v1["risposta_da_cache"] != "Sì"]
    v1c = v1_genuine[v1_genuine["status"] == "COMPLETED"].sort_values("timestamp").groupby("domanda").last().reset_index()
    v2c = v2[v2["status"] == "COMPLETED"].sort_values("timestamp").groupby("domanda").last().reset_index()

    comuni = set(v1c["domanda"]) & set(v2c["domanda"])
    righe = []
    for domanda in comuni:
        r1 = v1c[v1c["domanda"] == domanda].iloc[0]
        r2 = v2c[v2c["domanda"] == domanda].iloc[0]
        tok1, tok2 = r1["total_tokens"], r2["total_tokens"]
        dur1, dur2 = r1["durata_sec"], r2["durata_sec"]
        n1, n2 = r1["n_chiamate_llm"], r2["n_chiamate_llm"]
        righe.append({
            "domanda": domanda,
            "percorso_v2": r2["gruppo"],
            "v1_tokens": tok1, "v2_tokens": tok2,
            "delta_tokens_pct": ((tok2 - tok1) / tok1 * 100) if pd.notna(tok1) and tok1 else pd.NA,
            "v1_durata_sec": dur1, "v2_durata_sec": dur2,
            "delta_durata_pct": ((dur2 - dur1) / dur1 * 100) if pd.notna(dur1) and dur1 else pd.NA,
            "v1_costo": r1["costo_stimato_usd"], "v2_costo": r2["costo_stimato_usd"],
            "v1_chiamate": n1, "v2_chiamate": n2,
            "delta_chiamate": (n2 - n1) if pd.notna(n1) and pd.notna(n2) else pd.NA,
        })
    return pd.DataFrame(righe)
