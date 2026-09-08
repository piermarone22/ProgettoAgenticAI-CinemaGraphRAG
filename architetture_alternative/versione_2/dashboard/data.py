"""Caricamento e normalizzazione dei dati per la dashboard di versione_2.

Due fonti, due db distinti:
- versione_2/tmp/cinema_traces_v2.db (tabella query_log): una riga per ogni
  chiamata a /query (domanda, risposta, PERCORSO scelto dal router, status).
- tmp/cinema_traces.db (tabella agno_eval_runs) in ROOT: i punteggi del giudice
  (AgentAsJudgeEval). Questo database resta quello dell'architettura originale
  perché quality_eval (usato sia dal post_hook di cinema_team sia dalle
  chiamate esplicite di versione_2 sui percorsi diretti) ha il proprio db
  cablato alla costruzione in app/agent.py — non e' quindi spostabile insieme
  al resto del tracing di versione_2.

Nota / limite noto: essendo il db degli eval condiviso tra le due architetture,
se la STESSA identica domanda venisse posta sia sull'architettura originale sia
su versione_2 nella stessa finestra temporale, il join per (testo domanda +
timestamp piu' vicino) potrebbe in teoria abbinare una riga di versione_2 a una
valutazione generata dall'altra architettura. Nella pratica e' un caso raro
(richiede domanda testualmente identica ravvicinata nel tempo tra le due), ma va
tenuto presente leggendo i punteggi.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

_V2_DIR = Path(__file__).resolve().parent.parent
# _V2_DIR.parent.parent, non _V2_DIR.parent: versione_2/ vive sotto
# architetture_alternative/, un livello sotto la vera radice del progetto.
_ROOT_DIR = _V2_DIR.parent.parent

QUERY_LOG_DB_PATH = _V2_DIR / "tmp" / "cinema_traces_v2.db"
TRACES_DB_PATH = _ROOT_DIR / "tmp" / "cinema_traces.db"


def load_query_log() -> pd.DataFrame:
    colonne = ["timestamp", "architettura", "domanda", "percorso", "agenti_coinvolti", "risposta", "status"]
    if not QUERY_LOG_DB_PATH.exists():
        return pd.DataFrame(columns=colonne)

    conn = sqlite3.connect(QUERY_LOG_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM query_log ORDER BY id", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame(columns=colonne)
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["agenti_coinvolti"] = df["agenti_coinvolti"].fillna("(nessuno)")
    return df


def load_eval_scores() -> pd.DataFrame:
    """Stessa logica di join di dashboard/data.py in root (vedi nota lì): l'eval
    genera un run_id indipendente da quello della run originale, quindi si usa
    il testo della domanda ("input") abbinato al timestamp piu' vicino."""
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
        result = results[0]
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
    """Query log di versione_2 + punteggi del giudice, abbinati per testo della
    domanda + timestamp piu' vicino (solo le righe percorso='ambiguous' possono
    avere un match, vedi docstring del modulo)."""
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
