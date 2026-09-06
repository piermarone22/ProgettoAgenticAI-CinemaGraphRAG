"""Esegue K domande per OGNI categoria del set di test (invece di un blocco unico
di N domande in ordine di file, vedi esegui_test.py per quello).

K e' il parametro --k, facilmente cambiabile da riga di comando (default 2).

Uso:
    uv run python testing/esegui_per_categoria.py           # 2 domande per categoria (default)
    uv run python testing/esegui_per_categoria.py --k 1     # 1 sola domanda per categoria
    uv run python testing/esegui_per_categoria.py --k 3 --pausa 8

Come per esegui_test.py, ogni chiamata passa dal QueryLoggerMiddleware: i
risultati completi finiscono comunque in tmp/query_log.csv.
"""

import argparse
import time
from collections import defaultdict

from esegui_test import CSV_PATH, esegui_domanda, load_domande

# Valore di default se non specificato da riga di comando — cambialo qui se preferisci
# non passare sempre --k a mano.
K_DOMANDE_PER_CATEGORIA_DEFAULT = 2


def raggruppa_per_categoria(domande: list[dict]) -> dict[str, list[dict]]:
    gruppi: dict[str, list[dict]] = defaultdict(list)
    for riga in domande:
        gruppi[riga["categoria"]].append(riga)
    return gruppi


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue K domande per ogni categoria del set di test.")
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
                print(f"    ✅ {status} — {anteprima}")
            else:
                errori += 1
                print(f"    ⚠️  {status} — {anteprima}")

            if riga.get("ground_truth"):
                print(f"    ℹ️  ground truth attesa: {riga['ground_truth']}")

            if contatore < totale:
                time.sleep(args.pausa)
            print()

    print(f"Fatto: {ok} completate, {errori} con problemi su {totale} totali ({len(gruppi)} categorie).")
    print("Dettagli completi (risposta integrale, token, costo, tempo, fallback) in tmp/query_log.csv.")


if __name__ == "__main__":
    main()
