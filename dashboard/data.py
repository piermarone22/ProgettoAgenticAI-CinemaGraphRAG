"""Caricamento e normalizzazione dei dati per la dashboard.

Due fonti:
- tmp/query_log.csv: una riga per ogni run del team (domanda, risposta, agenti coinvolti, status)
- tmp/cinema_traces.db (tabella agno_eval_runs): i punteggi del giudice (AgentAsJudgeEval),
  collegati alle run tramite run_id.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
QUERY_LOG_PATH = BASE_DIR / "tmp" / "query_log.csv"
TRACES_DB_PATH = BASE_DIR / "tmp" / "cinema_traces.db"


def load_query_log() -> pd.DataFrame:
    if not QUERY_LOG_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp", "team_id", "session_id", "run_id", "domanda", "agenti_chiamati", "risposta", "status"]
        )

    df = pd.read_csv(QUERY_LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["agenti_chiamati"] = df["agenti_chiamati"].fillna("(solo coordinator)")
    return df


def load_eval_scores() -> pd.DataFrame:
    """Estrae input, score, passed, reason da agno_eval_runs (una riga per valutazione).

    Nota: l'eval genera un proprio run_id indipendente da quello della run del team
    (AgentAsJudgeEval non lo propaga), quindi non e' utilizzabile come chiave di join.
    Si usa invece il testo della domanda ("input") abbinato al timestamp piu' vicino.
    """
    if not TRACES_DB_PATH.exists():
        return pd.DataFrame(columns=["input", "score", "passed", "reason", "eval_created_at"])

    conn = sqlite3.connect(TRACES_DB_PATH)
    try:
        cur = conn.execute(
            "SELECT eval_data, created_at FROM agno_eval_runs WHERE eval_type = 'agent_as_judge'"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["input", "score", "passed", "reason", "eval_created_at"])
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
        result = results[0]  # una sola valutazione per run in questo setup
        records.append(
            {
                "input": result.get("input"),
                "score": result.get("score"),
                "passed": result.get("passed"),
                "reason": result.get("reason"),
                "eval_created_at": pd.to_datetime(int(created_at), unit="s", utc=True) if created_at else pd.NaT,
            }
        )

    return pd.DataFrame.from_records(records)


def load_merged() -> pd.DataFrame:
    """Query log + punteggi del giudice, abbinati per testo della domanda + timestamp piu' vicino."""
    query_log = load_query_log()
    query_log["score"] = pd.NA
    query_log["passed"] = pd.NA
    query_log["reason"] = pd.NA

    eval_scores = load_eval_scores()
    if eval_scores.empty or query_log.empty:
        return query_log

    completed = query_log[query_log["status"] == "COMPLETED"]
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
        query_log.loc[best_idx, ["score", "passed", "reason"]] = [ev["score"], ev["passed"], ev["reason"]]

    return query_log
