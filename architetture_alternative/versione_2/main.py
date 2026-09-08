"""FastAPI dell'architettura router (versione_2): instrada ogni domanda al
percorso deciso da router.classifica(), riusando gli agenti di app/agent.py.
Porta 8001, per girare in parallelo all'architettura originale (porta 8000).
"""

import asyncio
import sys
import time
from pathlib import Path

# .parent.parent.parent: questo file vive in architetture_alternative/versione_2/,
# due livelli sotto la radice del progetto (non uno solo).
_ROOT = Path(__file__).resolve().parent.parent.parent
# append, non insert(0, ...): la directory di versione_2 deve restare la PRIMA
# voce di sys.path, altrimenti quando uvicorn re-importa "main:app" per stringa
# risolverebbe "main" sul main.py di app/ (stesso nome di modulo) invece che
# su questo file.
sys.path.append(str(_ROOT / "app"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(dotenv_path=str(_ROOT / ".env"))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent import graph_agent, semantic_agent, cinema_team, quality_eval  # noqa: E402

from router import classifica, controlla_prompt_injection, Percorso  # noqa: E402
from pitch_workflow import esegui_pitch  # noqa: E402
from query_logger import log_query  # noqa: E402

app = FastAPI(title="Cinema GraphRAG — Router sperimentale (versione_2)")


class DomandaRequest(BaseModel):
    message: str


async def _valuta_in_background(domanda: str, content: str) -> None:
    """Richiama il giudice (quality_eval) a mano, fire-and-forget: i percorsi
    diretti bypassano cinema_team e quindi il suo post_hook automatico."""
    try:
        await quality_eval.arun(input=domanda, output=str(content))
    except Exception as exc:
        print(f"[versione_2] Errore durante la valutazione del giudice: {exc}")


@app.post("/query")
async def query(req: DomandaRequest) -> dict:
    start = time.monotonic()
    domanda = req.message

    pattern_sospetto = controlla_prompt_injection(domanda)
    if pattern_sospetto:
        durata = time.monotonic() - start
        content = f"Richiesta bloccata: rilevato pattern sospetto di prompt injection ('{pattern_sospetto}')."
        log_query(domanda, "blocked", [], content, "BLOCKED", [], durata)
        return {"status": "BLOCKED", "percorso": "blocked", "content": content, "durata_sec": round(durata, 2)}

    percorso = classifica(domanda)

    try:
        if percorso == Percorso.GRAPH:
            result = await graph_agent.arun(domanda)
            content, agenti, raw_results = result.content, ["Graph Query Agent"], [result]
            status = result.status.value
            asyncio.create_task(_valuta_in_background(domanda, content))
        elif percorso == Percorso.SEMANTIC:
            result = await semantic_agent.arun(domanda)
            content, agenti, raw_results = result.content, ["Semantic Query Agent"], [result]
            status = result.status.value
            asyncio.create_task(_valuta_in_background(domanda, content))
        elif percorso == Percorso.PITCH:
            pitch = await esegui_pitch(domanda)
            content, agenti, raw_results = pitch["content"], pitch["agent_names"], pitch["raw_results"]
            status = "COMPLETED"  # esegui_pitch non propaga uno status aggregato: se arriva qui, tutti e 3 i run sono andati a buon fine
            asyncio.create_task(_valuta_in_background(domanda, content))
        else:  # AMBIGUOUS: fallback sull'architettura originale (Team coordinate)
            result = await cinema_team.arun(domanda)
            content, agenti, raw_results = result.content, ["Creative Agent (coordinate, fallback)"], [result]
            status = result.status.value
            # NON serve chiamare _valuta_in_background qui: cinema_team ha gia'
            # quality_eval come post_hook nativo (run_in_background=True), che
            # scatta automaticamente dentro cinema_team.arun() — chiamarlo di
            # nuovo qui duplicherebbe la valutazione (due righe in eval_runs
            # per la stessa domanda).
    except Exception as exc:
        durata = time.monotonic() - start
        content = f"Errore durante l'esecuzione: {exc}"
        log_query(domanda, percorso.value, [], content, "ERROR", [], durata)
        return {"status": "ERROR", "percorso": percorso.value, "content": content, "durata_sec": round(durata, 2)}

    durata = time.monotonic() - start
    log_query(domanda, percorso.value, agenti, content, status, raw_results, durata)

    return {
        "status": status,
        "percorso": percorso.value,
        "agenti_coinvolti": agenti,
        "content": content,
        "durata_sec": round(durata, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
