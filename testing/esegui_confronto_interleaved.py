"""Esegue le STESSE domande su entrambe le architetture (versione_1/root e
versione_2/router), alternando (interleaved) invece di testarle in blocco una
dopo l'altra, aspettando sempre che la valutazione del giudice sia
effettivamente comparsa nel db prima di passare alla domanda successiva, e
scrivendo un CSV dedicato SOLO ai risultati di questa esecuzione.

Vedi testing/FRAMEWORK_CONFRONTO.md per il perché di ogni scelta qui sotto:
domande held-out (mai usate come esempi del classificatore di versione_2),
esecuzione alternata (non v1-tutto-poi-v2-tutto, per non far pesare le
condizioni esterne — quota Gemini, 429/503 — in modo diverso sulle due
architetture), attesa esplicita della valutazione (mai dare per buono un
batch se lo score nel db e' rimasto None).

Perché un CSV dedicato invece di riusare i log condivisi (tmp/query_log.csv,
versione_2/tmp/query_log_v2.csv) o la dashboard di confronto: quei log
accumulano TUTTE le domande mai eseguite, in sessioni diverse, con condizioni
diverse (quota, orario, eventuale fallback). La dashboard di confronto
(architetture_alternative/confronto/) per le "domande comuni" prende la run
più RECENTE di ciascuna domanda in ciascun log — se la stessa domanda fosse
mai stata eseguita anche in una sessione precedente, un confronto letto da lì
rischierebbe di mescolare un'esecuzione di oggi con una di ieri. Questo script
invece legge, subito dopo ogni chiamata, ESATTAMENTE la riga appena scritta
nel log dell'architettura interessata (per testo della domanda + timestamp
successivo all'invio) e la registra nel proprio CSV di output — un confronto
autosufficiente, limitato a QUESTA esecuzione, senza alcuna ambiguità.

Prerequisiti:
- main.py (root, porta 8000) e architetture_alternative/versione_2/main.py
  (porta 8001) devono essere gia' avviati.
- La quota giornaliera di Gemini deve essere disponibile (controllare prima:
  un 429/503 con 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' nei log
  significa che le valutazioni falliranno comunque, indipendentemente da
  questo script — vedi FRAMEWORK_CONFRONTO.md, checklist pre-test).

Uso:
    uv run python testing/esegui_confronto_interleaved.py
    uv run python testing/esegui_confronto_interleaved.py --n 5
    uv run python testing/esegui_confronto_interleaved.py --timeout-giudice 120 --pausa 8
    uv run python testing/esegui_confronto_interleaved.py --file testing/domande_confronto_holdout.csv

Output: testing/risultati_confronto/confronto_<timestamp>.csv — una riga per
domanda, con le colonne v1_* e v2_* affiancate (token, costo, durata, modello,
fallback, punteggio del giudice) pronte per un confronto diretto in un foglio
di calcolo, senza passare dalla dashboard.
"""

import argparse
import csv
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_HOLDOUT_PATH = BASE_DIR / "domande_confronto_holdout.csv"
TRACES_DB_PATH = ROOT_DIR / "tmp" / "cinema_traces.db"
V1_LOG_PATH = ROOT_DIR / "tmp" / "query_log.csv"
V2_LOG_PATH = ROOT_DIR / "architetture_alternative" / "versione_2" / "tmp" / "query_log_v2.csv"
OUTPUT_DIR = BASE_DIR / "risultati_confronto"

V1_URL = "http://127.0.0.1:8000/teams/cinema-creative-team/runs"
V2_URL = "http://127.0.0.1:8001/query"

OUTPUT_FIELDS = [
    "categoria", "domanda", "ground_truth",
    "v1_status", "v1_model", "v1_provider", "v1_fallback",
    "v1_input_tokens", "v1_output_tokens", "v1_total_tokens", "v1_costo_usd", "v1_durata_sec",
    "v1_score", "v1_score_reason",
    "v2_status", "v2_percorso", "v2_model", "v2_provider", "v2_fallback",
    "v2_input_tokens", "v2_output_tokens", "v2_total_tokens", "v2_costo_usd", "v2_durata_sec",
    "v2_score", "v2_score_reason",
    "delta_token_pct", "delta_durata_pct", "delta_score",
]


def carica_domande(percorso_csv: Path, n: int | None) -> list[dict]:
    with percorso_csv.open(newline="", encoding="utf-8") as f:
        righe = list(csv.DictReader(f))
    return righe[:n] if n else righe


def esegui_v1(domanda: str) -> dict:
    try:
        resp = requests.post(
            V1_URL,
            files={"message": (None, domanda), "stream": (None, "false")},
            timeout=400,
        )
    except requests.exceptions.RequestException as exc:
        return {"status": "ERRORE_CONNESSIONE", "content": str(exc)}
    if resp.status_code != 200:
        return {"status": f"HTTP {resp.status_code}", "content": resp.text[:200]}
    data = resp.json()
    return {"status": data.get("status", "?"), "content": data.get("content", "")}


def esegui_v2(domanda: str) -> dict:
    try:
        resp = requests.post(V2_URL, json={"message": domanda}, timeout=400)
    except requests.exceptions.RequestException as exc:
        return {"status": "ERRORE_CONNESSIONE", "content": str(exc), "percorso": "?"}
    if resp.status_code != 200:
        return {"status": f"HTTP {resp.status_code}", "content": resp.text[:200], "percorso": "?"}
    data = resp.json()
    return {"status": data.get("status", "?"), "content": data.get("content", ""), "percorso": data.get("percorso", "?")}


def attendi_valutazione(domanda: str, timestamp_invio: float, timeout_sec: float, poll_ogni: float = 3.0) -> dict | None:
    """Interroga tmp/cinema_traces.db finche' non compare una valutazione del
    giudice per questa esatta domanda, generata DOPO l'invio della richiesta,
    con uno score valido (non None — una riga con score None e' un fallimento
    del giudice, non una valutazione riuscita, vedi limite noto in
    FRAMEWORK_CONFRONTO.md). Ritorna None se scade il timeout."""
    scadenza = time.time() + timeout_sec
    soglia_timestamp = int(timestamp_invio) - 2  # piccolo margine di sicurezza
    while time.time() < scadenza:
        if TRACES_DB_PATH.exists():
            con = sqlite3.connect(TRACES_DB_PATH)
            try:
                cur = con.execute(
                    "SELECT eval_data, created_at FROM agno_eval_runs "
                    "WHERE eval_type = 'agent_as_judge' AND created_at >= ? "
                    "ORDER BY created_at DESC",
                    (soglia_timestamp,),
                )
                righe = cur.fetchall()
            except sqlite3.OperationalError:
                righe = []
            finally:
                con.close()

            for eval_data_raw, _ in righe:
                try:
                    eval_data = json.loads(eval_data_raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                risultati = eval_data.get("results") or []
                if not risultati:
                    continue
                r = risultati[0]
                if r.get("input") == domanda and r.get("score") is not None:
                    return {"score": r.get("score"), "passed": r.get("passed"), "reason": r.get("reason")}
        time.sleep(poll_ogni)
    return None


def leggi_riga_appena_scritta(csv_path: Path, domanda: str, timestamp_invio: float, tentativi: int = 5, attesa: float = 1.0) -> dict | None:
    """Legge dal log persistente dell'architettura la riga appena aggiunta per
    QUESTA domanda — quella con timestamp piu' vicino, ma non precedente,
    all'invio della richiesta in questo script. Evita di prendere per sbaglio
    un'esecuzione precedente (di un'altra sessione) della stessa domanda."""
    soglia = timestamp_invio - 2
    for _ in range(tentativi):
        if csv_path.exists():
            candidate = []
            with csv_path.open(newline="", encoding="utf-8") as f:
                for riga in csv.DictReader(f):
                    if riga.get("domanda") != domanda:
                        continue
                    try:
                        ts = datetime.fromisoformat(riga["timestamp"]).timestamp()
                    except (KeyError, ValueError):
                        continue
                    if ts >= soglia:
                        candidate.append((ts, riga))
            if candidate:
                candidate.sort(key=lambda x: x[0])
                return candidate[0][1]
        time.sleep(attesa)
    return None


def _num(valore: str | None) -> float | None:
    try:
        return float(valore) if valore not in (None, "") else None
    except ValueError:
        return None


def _delta_pct(v1: float | None, v2: float | None) -> str:
    if v1 is None or v2 is None or v1 == 0:
        return ""
    return f"{(v2 - v1) / v1 * 100:+.0f}"


def stampa_esito(architettura: str, risultato: dict, valutazione: dict | None) -> None:
    status = risultato.get("status")
    anteprima = str(risultato.get("content", ""))[:150].replace("\n", " ")
    percorso = f" [percorso: {risultato['percorso']}]" if "percorso" in risultato else ""
    icona = "✅" if status == "COMPLETED" else "⚠️"
    print(f"  {icona} {architettura}{percorso} — {status} — {anteprima}")
    if valutazione is None:
        print("     ⚠️  GIUDICE: nessuna valutazione arrivata entro il timeout (vedi FRAMEWORK_CONFRONTO.md)")
    else:
        print(f"     ⚖️  GIUDICE: {valutazione['score']}/10 — {str(valutazione['reason'])[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esegue le stesse domande su versione_1 e versione_2 alternando, aspettando sempre la valutazione del giudice."
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_HOLDOUT_PATH, help="CSV di domande held-out da usare (default: domande_confronto_holdout.csv).")
    parser.add_argument("--n", type=int, default=None, help="Limita il numero di domande (default: tutte quelle nel file).")
    parser.add_argument("--pausa", type=float, default=5.0, help="Secondi di pausa tra una domanda e l'altra (default 5).")
    parser.add_argument("--timeout-giudice", type=float, default=90.0, help="Secondi massimi di attesa per la valutazione del giudice, per ciascuna chiamata (default 90).")
    args = parser.parse_args()

    domande = carica_domande(args.file, args.n)
    if not domande:
        print(f"Nessuna domanda trovata in {args.file}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"confronto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Eseguo {len(domande)} domande su ENTRAMBE le architetture (alternando v1/v2), da {args.file.name}.")
    print(f"Timeout valutazione giudice: {args.timeout_giudice}s per chiamata. Pausa tra domande: {args.pausa}s.")
    print(f"Risultati di questa esecuzione: {output_path}\n")

    coppie_complete = 0
    valutazioni_mancanti = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, riga in enumerate(domande, 1):
            domanda = riga["domanda"]
            print(f"[{i}/{len(domande)}] ({riga['categoria']}) {domanda[:90]}")
            if riga.get("ground_truth"):
                print(f"    ℹ️  ground truth attesa: {riga['ground_truth']}")

            t_v1 = time.time()
            esito_v1 = esegui_v1(domanda)
            log_v1 = leggi_riga_appena_scritta(V1_LOG_PATH, domanda, t_v1) if esito_v1.get("status") == "COMPLETED" else None
            valutazione_v1 = attendi_valutazione(domanda, t_v1, args.timeout_giudice) if esito_v1.get("status") == "COMPLETED" else None
            stampa_esito("versione_1", esito_v1, valutazione_v1)

            t_v2 = time.time()
            esito_v2 = esegui_v2(domanda)
            log_v2 = leggi_riga_appena_scritta(V2_LOG_PATH, domanda, t_v2) if esito_v2.get("status") == "COMPLETED" else None
            valutazione_v2 = attendi_valutazione(domanda, t_v2, args.timeout_giudice) if esito_v2.get("status") == "COMPLETED" else None
            stampa_esito("versione_2", esito_v2, valutazione_v2)

            if valutazione_v1 is not None and valutazione_v2 is not None:
                coppie_complete += 1
            else:
                valutazioni_mancanti += 1

            v1_tok = _num((log_v1 or {}).get("total_tokens"))
            v2_tok = _num((log_v2 or {}).get("total_tokens"))
            v1_dur = _num((log_v1 or {}).get("durata_sec"))
            v2_dur = _num((log_v2 or {}).get("durata_sec"))
            v1_score = valutazione_v1["score"] if valutazione_v1 else None
            v2_score = valutazione_v2["score"] if valutazione_v2 else None

            writer.writerow({
                "categoria": riga["categoria"],
                "domanda": domanda,
                "ground_truth": riga.get("ground_truth", ""),
                "v1_status": esito_v1.get("status"),
                "v1_model": (log_v1 or {}).get("model", ""),
                "v1_provider": (log_v1 or {}).get("model_provider", ""),
                "v1_fallback": (log_v1 or {}).get("fallback_usato", ""),
                "v1_input_tokens": (log_v1 or {}).get("input_tokens", ""),
                "v1_output_tokens": (log_v1 or {}).get("output_tokens", ""),
                "v1_total_tokens": (log_v1 or {}).get("total_tokens", ""),
                "v1_costo_usd": (log_v1 or {}).get("costo_stimato_usd", ""),
                "v1_durata_sec": (log_v1 or {}).get("durata_sec", ""),
                "v1_score": v1_score if v1_score is not None else "",
                "v1_score_reason": valutazione_v1["reason"] if valutazione_v1 else "",
                "v2_status": esito_v2.get("status"),
                "v2_percorso": esito_v2.get("percorso", ""),
                "v2_model": (log_v2 or {}).get("model", ""),
                "v2_provider": (log_v2 or {}).get("model_provider", ""),
                "v2_fallback": (log_v2 or {}).get("fallback_usato", ""),
                "v2_input_tokens": (log_v2 or {}).get("input_tokens", ""),
                "v2_output_tokens": (log_v2 or {}).get("output_tokens", ""),
                "v2_total_tokens": (log_v2 or {}).get("total_tokens", ""),
                "v2_costo_usd": (log_v2 or {}).get("costo_stimato_usd", ""),
                "v2_durata_sec": (log_v2 or {}).get("durata_sec", ""),
                "v2_score": v2_score if v2_score is not None else "",
                "v2_score_reason": valutazione_v2["reason"] if valutazione_v2 else "",
                "delta_token_pct": _delta_pct(v1_tok, v2_tok),
                "delta_durata_pct": _delta_pct(v1_dur, v2_dur),
                "delta_score": (v2_score - v1_score) if (v1_score is not None and v2_score is not None) else "",
            })
            out_f.flush()

            if i < len(domande):
                time.sleep(args.pausa)
            print()

    print(f"Fatto: {coppie_complete}/{len(domande)} domande con valutazione completa su ENTRAMBE le architetture.")
    if valutazioni_mancanti:
        print(f"⚠️  {valutazioni_mancanti} domande con almeno una valutazione mancante — vedi la colonna score vuota nel CSV di output.")
    print(f"\nRisultati completi (autosufficienti, limitati a questa esecuzione): {output_path}")
    print("Per un'analisi visiva su tutto lo storico invece: uv run streamlit run architetture_alternative/confronto/app.py")


if __name__ == "__main__":
    main()
