"""Middleware che traccia in tmp/query_log.csv ogni richiesta fatta al team
(domanda, risposta finale, agenti coinvolti), sia per le run in streaming (SSE)
sia per quelle non-streaming."""

import csv
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

import response_cache

# .parent.parent: questo file vive in app/, ma tmp/ e' alla radice del progetto
# (condivisa con dashboard/, testing/, ecc.) — non .parent da solo, altrimenti
# punterebbe a app/tmp/ invece che alla vera radice.
CSV_PATH = Path(__file__).resolve().parent.parent / "tmp" / "query_log.csv"
CSV_FIELDS = [
    "timestamp", "team_id", "session_id", "run_id", "domanda", "agenti_chiamati",
    "risposta", "status", "model", "model_provider", "fallback_usato", "risposta_da_cache",
    "input_tokens", "output_tokens", "total_tokens", "costo_stimato_usd", "durata_sec",
]

# Provider primario del coordinator (gemini-3.5-flash-lite / Google). Se una run
# risponde con un provider diverso, significa che il fallback (FallbackConfig su
# cinema_team, vedi agent.py) e' effettivamente scattato — es. Groq dopo un 429/503.
_PRIMARY_PROVIDER = "Google"

_RUNS_PATH_RE = re.compile(r"^/teams/[^/]+/runs/?$")
_MULTIPART_FIELD_RE = re.compile(r'name="([^"]+)"')
_AGENT_NAME_RE = re.compile(r'"agent_name":"([^"]*)"')
_MODEL_RE = re.compile(r'"model":"([^"]*)"')
_MODEL_PROVIDER_RE = re.compile(r'"model_provider":"([^"]*)"')
_FINAL_EVENT_RE = re.compile(r"event:\s*(TeamRunCompleted|TeamRunError)\r?\ndata:\s*(\{.*\})\r?\n")

# Prezzo per milione di token (tier a pagamento, non free-tier che e' $0).
# Fonti: https://www.cloudzero.com/blog/gemini-pricing/ (Gemini 3.5 Flash-Lite)
#        https://www.aipricing.guru/blog/groq-api-pricing-guide-2026/ (Groq gpt-oss-120b, modello di fallback)
# Usato solo per stimare "quanto costerebbe se non fossi sul free tier".
_PRICING_PER_MILLION_TOKENS_USD = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "qwen/qwen3.8-27b": {"input": 0.80, "output": 4.00},  # modello di fallback attuale
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},  # fallback precedente, tenuto per le run storiche
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = _PRICING_PER_MILLION_TOKENS_USD.get(model)
    if pricing is None:
        return None
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def _compute_tokens_and_cost(data: dict) -> dict:
    """Somma i token del coordinator (metrics di primo livello) + di ogni agente membro
    (member_responses[].metrics) — il campo 'metrics' di primo livello da solo copre
    SOLO il coordinator, non i worker delegati. Il costo del giudice (post-hook, gira in
    background dopo la risposta) non e' incluso: non arriva nel payload della run."""
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    cost_unknown = False

    blocks = [data.get("metrics") or {}]
    for mr in (data.get("member_responses") or []):
        blocks.append(mr.get("metrics") or {})

    for block in blocks:
        for model_info in (block.get("details", {}) or {}).get("model", []) or []:
            in_tok = model_info.get("input_tokens", 0) or 0
            out_tok = model_info.get("output_tokens", 0) or 0
            input_tokens += in_tok
            output_tokens += out_tok
            cost = _estimate_cost_usd(model_info.get("id", ""), in_tok, out_tok)
            if cost is None:
                cost_unknown = True
            else:
                cost_usd += cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "costo_stimato_usd": None if cost_unknown and cost_usd == 0 else round(cost_usd, 6),
    }


def _ensure_csv_header() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        return

    # Migrazione: se il file esiste gia' con un header piu' vecchio (senza le colonne
    # nuove), lo riscrive con l'header aggiornato, preservando le righe esistenti
    # (le colonne nuove restano vuote per le righe pre-esistenti).
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        if existing_fields == CSV_FIELDS:
            return
        rows = list(reader)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _append_csv_row(row: dict) -> None:
    _ensure_csv_header()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def _parse_multipart_text_fields(body: bytes, content_type: str) -> dict[str, str]:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m:
        return {}
    boundary = ("--" + m.group(1)).encode()
    fields: dict[str, str] = {}
    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        headers_raw, sep, value = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers_text = headers_raw.decode(errors="ignore")
        if "filename=" in headers_text:
            continue  # salta i file, ci interessano solo i campi testuali
        name_match = _MULTIPART_FIELD_RE.search(headers_text)
        if name_match:
            fields[name_match.group(1)] = value.rstrip(b"\r\n").decode(errors="ignore")
    return fields


def _clean_content(content) -> str:
    if not isinstance(content, str):
        return content or ""
    if content.startswith("<bound method"):
        # errore grezzo della libreria (es. 429 non formattato): tieni solo la prima riga
        return content.split("\n", 1)[0][:300]
    return content


def _extract_from_stream(raw_text: str) -> dict:
    agent_names: list[str] = []
    for name in _AGENT_NAME_RE.findall(raw_text):
        if name and name not in agent_names:
            agent_names.append(name)

    # Il campo "model"/"model_provider" NON e' presente nell'evento TeamRunCompleted
    # (solo nella risposta non-streaming). E' pero' presente nei precedenti eventi
    # TeamModelRequestCompleted/Started: l'ultima occorrenza nel testo corrisponde
    # cronologicamente alla sintesi finale del coordinator (utile per verificare
    # se e' scattato il fallback a Groq).
    model_matches = _MODEL_RE.findall(raw_text)
    provider_matches = _MODEL_PROVIDER_RE.findall(raw_text)
    last_model = model_matches[-1] if model_matches else ""
    last_provider = provider_matches[-1] if provider_matches else ""

    final_data: dict = {}
    final_event = ""
    for match in _FINAL_EVENT_RE.finditer(raw_text):
        final_event = match.group(1)
        try:
            final_data = json.loads(match.group(2))
        except json.JSONDecodeError:
            final_data = {}

    return {
        "session_id": final_data.get("session_id", ""),
        "run_id": final_data.get("run_id", ""),
        "content": final_data.get("content", ""),
        "status": final_data.get("status") or ("ERROR" if final_event == "TeamRunError" else "COMPLETED" if final_event else ""),
        "agent_names": agent_names,
        "model": final_data.get("model") or last_model,
        "model_provider": final_data.get("model_provider") or last_provider,
        **_compute_tokens_and_cost(final_data),
    }


def _extract_from_json(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = {}

    agent_names: list[str] = []
    for mr in (data.get("member_responses") or []):
        name = mr.get("agent_name") or mr.get("name") or mr.get("agent_id")
        if name and name not in agent_names:
            agent_names.append(name)

    return {
        "session_id": data.get("session_id", ""),
        "run_id": data.get("run_id", ""),
        "content": data.get("content", ""),
        "status": data.get("status", ""),
        "agent_names": agent_names,
        "model": data.get("model", ""),
        "model_provider": data.get("model_provider", ""),
        **_compute_tokens_and_cost(data),
    }


def _log_extracted(team_id: str, message: str, extracted: dict, duration_sec: float, da_cache: bool) -> None:
    provider = extracted["model_provider"]
    fallback_usato = "" if not provider else ("Sì" if provider != _PRIMARY_PROVIDER else "No")

    _append_csv_row({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "team_id": team_id,
        "session_id": extracted["session_id"],
        "run_id": extracted["run_id"],
        "domanda": message,
        "agenti_chiamati": ", ".join(extracted["agent_names"]) if extracted["agent_names"] else "(solo coordinator)",
        "risposta": _clean_content(extracted["content"]),
        "status": extracted["status"],
        "model": extracted["model"],
        "model_provider": provider,
        "fallback_usato": fallback_usato,
        "risposta_da_cache": "Sì" if da_cache else "No",
        "input_tokens": extracted["input_tokens"],
        "output_tokens": extracted["output_tokens"],
        "total_tokens": extracted["total_tokens"],
        "costo_stimato_usd": extracted["costo_stimato_usd"],
        "durata_sec": round(duration_sec, 2),
    })


def _log_run(team_id: str, message: str, raw: bytes, is_stream: bool, duration_sec: float) -> dict:
    """Estrae i dati dalla risposta grezza, li logga su CSV e li restituisce
    (cosi' il chiamante puo' eventualmente metterli in cache)."""
    text = raw.decode("utf-8", errors="replace")
    extracted = _extract_from_stream(text) if is_stream else _extract_from_json(text)
    _log_extracted(team_id, message, extracted, duration_sec, da_cache=False)
    return extracted


def _build_cached_response(extracted: dict, wants_stream: bool):
    """Costruisce una risposta sintetica (JSON o SSE) a partire da una voce di
    cache, nello stesso formato che AgentOS restituirebbe per una run vera."""
    payload = {
        "run_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "content": extracted.get("content", ""),
        "content_type": "str",
        "status": extracted.get("status", "COMPLETED"),
        "model": extracted.get("model", ""),
        "model_provider": extracted.get("model_provider", ""),
        "member_responses": [],
        "cached": True,
    }

    if not wants_stream:
        return JSONResponse(payload)

    sse_body = f"event: TeamRunCompleted\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _gen():
        yield sse_body.encode()

    return StreamingResponse(_gen(), media_type="text/event-stream")


class QueryLoggerMiddleware(BaseHTTPMiddleware):
    """Traccia in CSV le chiamate POST /teams/{team_id}/runs e le serve dalla
    cache locale (response_cache) quando la stessa identica domanda e' gia'
    stata risposta con successo di recente, evitando di rieseguire l'intera
    pipeline multi-agente."""

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or not _RUNS_PATH_RE.match(request.url.path):
            return await call_next(request)

        team_id = request.url.path.split("/")[2]
        body = await request.body()
        fields = _parse_multipart_text_fields(body, request.headers.get("content-type", ""))
        message = fields.get("message", "")
        wants_stream = fields.get("stream", "true").strip().lower() != "false"
        start_time = time.monotonic()

        cached = response_cache.get(team_id, message)
        if cached is not None:
            duration_sec = time.monotonic() - start_time
            try:
                _log_extracted(team_id, message, cached, duration_sec, da_cache=True)
            except Exception as exc:
                print(f"[QueryLogger] Errore durante il logging (cache hit): {exc}")
            return _build_cached_response(cached, wants_stream)

        response = await call_next(request)
        is_stream = "text/event-stream" in response.headers.get("content-type", "")
        original_iterator = response.body_iterator

        async def tee_and_log():
            buf = bytearray()
            async for chunk in original_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                buf.extend(chunk)
                yield chunk
            # Tempo reale end-to-end (dalla richiesta all'ultimo byte inviato al client):
            # include anche le attese dei retry/fallback, a differenza dei "duration"
            # interni dei singoli model call.
            duration_sec = time.monotonic() - start_time
            try:
                extracted = _log_run(team_id, message, bytes(buf), is_stream, duration_sec)
                if extracted.get("status") == "COMPLETED":
                    response_cache.put(team_id, message, extracted)
            except Exception as exc:  # non deve mai rompere la risposta all'utente
                print(f"[QueryLogger] Errore durante il logging: {exc}")

        response.body_iterator = tee_and_log()
        return response
