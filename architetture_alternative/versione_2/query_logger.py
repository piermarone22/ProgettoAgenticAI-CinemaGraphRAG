"""Logging delle query per l'architettura a router deterministico (versione_2).

A differenza di query_logger.py in root (middleware su AgentOS che intercetta
testo grezzo SSE/JSON), qui si lavora direttamente sugli oggetti RunOutput
restituiti da .arun() — piu' semplice perche' non c'e' uno strato HTTP/streaming
da intercettare. Vale pero' la stessa cautela sul conteggio dei token scoperta
nell'architettura originale: il percorso 'ambiguous' passa dal Team
(cinema_team), il cui '.metrics' di primo livello copre SOLO il coordinator,
non i worker delegati (in '.member_responses') — vanno sommati esplicitamente.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

CSV_PATH = Path(__file__).parent / "tmp" / "query_log_v2.csv"

# Valore costante scritto su ogni riga: il file di log e' gia' fisicamente
# separato da quello dell'architettura originale (tmp/query_log.csv in root),
# ma un campo esplicito permette comunque di riconoscere la provenienza di ogni
# riga anche se in futuro i due log venissero uniti/confrontati in un'unica
# dashboard, senza doversi affidare solo al nome del file sorgente.
_ARCHITETTURA = "versione_2_router"

CSV_FIELDS = [
    "timestamp", "architettura", "domanda", "percorso", "agenti_coinvolti", "risposta", "status",
    "model", "model_provider", "fallback_usato",
    "input_tokens", "output_tokens", "total_tokens", "costo_stimato_usd", "durata_sec",
]

# Provider primario atteso (Gemini/Google). Se tra i RunOutput coinvolti in una
# query compare un provider diverso, il fallback (FallbackConfig, vedi root
# agent.py) e' scattato su almeno una delle chiamate.
_PRIMARY_PROVIDER = "Google"

# Stesse tariffe indicative usate in root/query_logger.py (tier a pagamento,
# solo per stimare "quanto costerebbe se non si fosse sul free tier").
_PRICING_PER_MILLION_TOKENS_USD = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "qwen/qwen3.8-27b": {"input": 0.80, "output": 4.00},
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = _PRICING_PER_MILLION_TOKENS_USD.get(model)
    if pricing is None:
        return None
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def _flatten(result) -> list:
    """Un RunOutput di Agent e' gia' un blocco singolo. Un TeamRunOutput (solo
    il percorso 'ambiguous', cinema_team) ha anche '.member_responses': i
    worker delegati, i cui metrics NON sono inclusi in quelli del coordinator."""
    if result is None:
        return []
    outputs = [result]
    outputs.extend(getattr(result, "member_responses", None) or [])
    return outputs


def _aggregate(results: list) -> dict:
    flat = [r for result in results for r in _flatten(result)]

    input_tokens = output_tokens = 0
    cost_usd = 0.0
    cost_unknown = False
    modelli, provider_visti = set(), set()

    for r in flat:
        if getattr(r, "model", None):
            modelli.add(r.model)
        if getattr(r, "model_provider", None):
            provider_visti.add(r.model_provider)
        metrics = getattr(r, "metrics", None)
        if metrics is None:
            continue
        in_tok = metrics.input_tokens or 0
        out_tok = metrics.output_tokens or 0
        input_tokens += in_tok
        output_tokens += out_tok
        if r.model:
            cost = _estimate_cost_usd(r.model, in_tok, out_tok)
            if cost is None:
                cost_unknown = True
            else:
                cost_usd += cost

    fallback_usato = ""
    if provider_visti:
        fallback_usato = "Sì" if any(p != _PRIMARY_PROVIDER for p in provider_visti) else "No"

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "costo_stimato_usd": None if (cost_unknown and cost_usd == 0) else round(cost_usd, 6),
        "model": "+".join(sorted(modelli)),
        "model_provider": "+".join(sorted(provider_visti)),
        "fallback_usato": fallback_usato,
    }


def _ensure_csv_header() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def log_query(
    domanda: str,
    percorso: str,
    agenti_coinvolti: list[str],
    content: str,
    status: str,
    results: list,
    durata_sec: float,
) -> None:
    """Logga una riga in tmp/query_log_v2.csv. 'results' e' la lista dei
    RunOutput/TeamRunOutput grezzi prodotti per rispondere alla domanda (uno
    per graph/semantic, tre per pitch, uno per ambiguous, vuota per blocked)."""
    _ensure_csv_header()
    tok = _aggregate(results)

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "architettura": _ARCHITETTURA,
            "domanda": domanda,
            "percorso": percorso,
            "agenti_coinvolti": ", ".join(agenti_coinvolti) if agenti_coinvolti else "(nessuno)",
            "risposta": content,
            "status": status,
            "durata_sec": round(durata_sec, 2),
            **tok,
        })
