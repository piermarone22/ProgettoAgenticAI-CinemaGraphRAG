"""Esegue N domande dal set di test contro il team Cinema Creative in esecuzione
su FastAPI (main.py deve essere gia' avviato su localhost:8000).

Ogni chiamata passa comunque dal QueryLoggerMiddleware, quindi il risultato
completo (risposta, token, costo, tempo, eventuale fallback) finisce anche in
tmp/query_log.csv — questo script serve solo per lanciarle in serie con feedback
live, non duplica il logging.

Uso:
    uv run python testing/esegui_test.py 10
    uv run python testing/esegui_test.py 5 --categoria pitch_completo
    uv run python testing/esegui_test.py 5 --peso leggera
    uv run python testing/esegui_test.py 43 --pausa 8   # tutte, con piu' respiro tra una e l'altra
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "domande_di_test.csv"
API_URL = "http://127.0.0.1:8000/teams/cinema-creative-team/runs"


def load_domande(categoria: str | None, peso: str | None) -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if categoria:
        rows = [r for r in rows if r["categoria"] == categoria]
    if peso:
        rows = [r for r in rows if r["peso"] == peso]
    return rows


def esegui_domanda(domanda: str) -> dict:
    try:
        resp = requests.post(
            API_URL,
            files={"message": (None, domanda), "stream": (None, "false")},
            timeout=400,
        )
    except requests.exceptions.RequestException as exc:
        return {"status": "ERRORE_CONNESSIONE", "content": str(exc)}

    if resp.status_code != 200:
        return {"status": f"HTTP {resp.status_code}", "content": resp.text[:200]}

    data = resp.json()
    return {"status": data.get("status", "?"), "content": data.get("content", "")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue N domande di test contro il Cinema Creative Team.")
    parser.add_argument("n", type=int, help="Numero di domande da eseguire (nell'ordine del CSV).")
    parser.add_argument("--categoria", default=None, help="Filtra per categoria (es. pitch_completo).")
    parser.add_argument("--peso", default=None, choices=["leggera", "media", "pesante"], help="Filtra per peso.")
    parser.add_argument("--pausa", type=float, default=5.0, help="Secondi di pausa tra una domanda e l'altra (default 5).")
    args = parser.parse_args()

    domande = load_domande(args.categoria, args.peso)
    if not domande:
        print("Nessuna domanda trovata con questi filtri.")
        sys.exit(1)

    domande = domande[: args.n]
    print(f"Eseguo {len(domande)} domande (pausa {args.pausa}s tra una e l'altra)...\n")

    ok, errori = 0, 0
    for i, riga in enumerate(domande, 1):
        domanda = riga["domanda"]
        print(f"[{i}/{len(domande)}] ({riga['categoria']}, {riga['peso']}) {domanda[:90]}")
        risultato = esegui_domanda(domanda)
        status = risultato["status"]
        anteprima = str(risultato["content"])[:150].replace("\n", " ")
        if status == "COMPLETED":
            ok += 1
            print(f"    ✅ {status} — {anteprima}")
        else:
            errori += 1
            print(f"    ⚠️  {status} — {anteprima}")

        if riga.get("ground_truth"):
            print(f"    ℹ️  ground truth attesa: {riga['ground_truth']}")

        if i < len(domande):
            time.sleep(args.pausa)
        print()

    print(f"Fatto: {ok} completate, {errori} con problemi su {len(domande)} totali.")
    print("Dettagli completi (risposta integrale, token, costo, tempo, fallback) in tmp/query_log.csv.")


if __name__ == "__main__":
    main()
