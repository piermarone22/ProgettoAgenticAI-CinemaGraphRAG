"""Esegue K domande per OGNI categoria del set di test, contro l'architettura a
router deterministico (versione_2/main.py deve essere già avviato su localhost:8001).

Adattato da testing/esegui_per_categoria.py (root) per usare esegui_domanda di
versione_2/testing/esegui_test.py (endpoint /query, JSON).

Uso:
    uv run python versione_2/testing/esegui_per_categoria.py
    uv run python versione_2/testing/esegui_per_categoria.py --k 1
    uv run python versione_2/testing/esegui_per_categoria.py --k 3 --pausa 8
"""

import argparse
import time
from collections import defaultdict

from esegui_test import CSV_PATH, esegui_domanda, load_domande

K_DOMANDE_PER_CATEGORIA_DEFAULT = 2


def raggruppa_per_categoria(domande: list[dict]) -> dict[str, list[dict]]:
    gruppi: dict[str, list[dict]] = defaultdict(list)
    for riga in domande:
        gruppi[riga["categoria"]].append(riga)
    return gruppi


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue K domande per ogni categoria contro l'architettura router (versione_2).")
    parser.add_argument(
        "--k", type=int, default=K_DOMANDE_PER_CATEGORIA_DEFAULT,
        help=f"Quante domande eseguire per ciascuna categoria (default {K_DOMANDE_PER_CATEGORIA_DEFAULT}).",
    )
    parser.add_argument("--pausa", type=float, default=5.0, help="Secondi di pausa tra una domanda e l'altra (default 5).")
    args = parser.parse_args()

    tutte = load_domande(categoria=None, peso=None)
    gruppi = raggruppa_per_categoria(tutte)

    piano = [(cat, righe[: args.k]) for cat, righe in gruppi.items()]
    totale = sum(len(righe) for _, righe in piano)
    print(f"{len(gruppi)} categorie trovate in {CSV_PATH.name}, {args.k} domande ciascuna -> {totale} domande totali.\n")

    ok, errori = 0, 0
    contatore = 0
    for categoria, righe in piano:
        print(f"── {categoria} ({len(righe)} domande) ──────────────────────────")
        for riga in righe:
            contatore += 1
            domanda = riga["domanda"]
            print(f"[{contatore}/{totale}] ({riga['peso']}) {domanda[:90]}")
            risultato = esegui_domanda(domanda)
            status = risultato["status"]
            anteprima = str(risultato["content"])[:150].replace("\n", " ")
            if status == "COMPLETED":
                ok += 1
                print(f"    ✅ {status} [percorso: {risultato['percorso']}] — {anteprima}")
            else:
                errori += 1
                print(f"    ⚠️  {status} [percorso: {risultato['percorso']}] — {anteprima}")

            if riga.get("ground_truth"):
                print(f"    ℹ️  ground truth attesa: {riga['ground_truth']}")

            if contatore < totale:
                time.sleep(args.pausa)
            print()

    print(f"Fatto: {ok} completate, {errori} con problemi su {totale} totali ({len(gruppi)} categorie).")
    print("Dettagli completi (risposta integrale, percorso, token, costo, tempo) in versione_2/tmp/query_log_v2.csv.")


if __name__ == "__main__":
    main()
