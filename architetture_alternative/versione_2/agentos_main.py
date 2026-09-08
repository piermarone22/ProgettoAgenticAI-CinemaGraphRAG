"""AgentOS per l'architettura a router (versione_2): stessa logica di
instradamento di router.classifica()/main.py, ma esposta come Workflow con
uno step Router, così da essere esplorabile dal pannello AgentOS Web come
versione_1 (app/main.py). Porta 8002, separata da main.py (porta 8001).

Nota: Workflow non supporta un post_hook automatico come Team, quindi ogni
step richiama quality_eval a mano (fire-and-forget, stessa logica di
_valuta_in_background in main.py) sui percorsi diretti — sul percorso
AMBIGUOUS no, perché cinema_team ha gia' quality_eval come post_hook nativo e
richiamarlo di nuovo duplicherebbe la valutazione. Scrive inoltre nella stessa
tabella query_log (log_query(), identica logica di main.py per ciascun
percorso), cosi' le domande fatte dal pannello AgentOS Web compaiono anche
nella dashboard di versione_2.

Il tracing di AgentOS (sessioni/run nelle tabelle agno_*) va su un db dedicato,
cinema_traces_v2.db, separato da quello di versione_1 — con un'eccezione: il
percorso AMBIGUOUS riusa l'oggetto cinema_team importato da app/agent.py, che
ha il proprio db (quello di versione_1) già cablato alla costruzione, quindi
la SESSIONE AgentOS del solo fallback al Team finisce comunque nel db
originale (stesso comportamento di main.py, che riusa lo stesso oggetto). La
riga di query_log, invece, va sempre nel db di versione_2 per tutti i
percorsi: log_query() e' una scrittura SQL esplicita, indipendente dal db
interno di cinema_team.
"""

import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_ROOT / "app"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(dotenv_path=str(_ROOT / ".env"))

from agno.db.sqlite import SqliteDb  # noqa: E402
from agno.os import AgentOS  # noqa: E402
from agno.workflow.router import Router  # noqa: E402
from agno.workflow.step import Step  # noqa: E402
from agno.workflow.types import StepInput, StepOutput  # noqa: E402
from agno.workflow.workflow import Workflow  # noqa: E402

from agent import graph_agent, semantic_agent, cinema_team, quality_eval  # noqa: E402

# Db dedicato a versione_2: tiene separate le sessioni/run di questo Workflow
# da quelle di versione_1 (che vivono in tmp/cinema_traces.db).
db_v2 = SqliteDb(db_file=str(Path(__file__).resolve().parent / "tmp" / "cinema_traces_v2.db"))

from router import classifica, controlla_prompt_injection, Percorso  # noqa: E402
from pitch_workflow import esegui_pitch  # noqa: E402
from query_logger import log_query  # noqa: E402


async def _valuta_in_background(domanda: str, content: str) -> None:
    """Richiama il giudice (quality_eval) a mano, fire-and-forget: i percorsi
    diretti bypassano cinema_team e quindi il suo post_hook automatico
    (identica funzione di main.py)."""
    try:
        await quality_eval.arun(input=domanda, output=str(content))
    except Exception as exc:
        print(f"[agentos_main] Errore durante la valutazione del giudice: {exc}")


async def _esegui_graph_step(step_input: StepInput) -> StepOutput:
    domanda = step_input.get_input_as_string() or ""
    start = time.monotonic()
    result = await graph_agent.arun(domanda)
    log_query(domanda, Percorso.GRAPH.value, ["Graph Query Agent"], result.content, result.status.value, [result], time.monotonic() - start)
    asyncio.create_task(_valuta_in_background(domanda, result.content))
    return StepOutput(content=result.content)


async def _esegui_semantic_step(step_input: StepInput) -> StepOutput:
    domanda = step_input.get_input_as_string() or ""
    start = time.monotonic()
    result = await semantic_agent.arun(domanda)
    log_query(domanda, Percorso.SEMANTIC.value, ["Semantic Query Agent"], result.content, result.status.value, [result], time.monotonic() - start)
    asyncio.create_task(_valuta_in_background(domanda, result.content))
    return StepOutput(content=result.content)


async def _esegui_pitch_step(step_input: StepInput) -> StepOutput:
    domanda = step_input.get_input_as_string() or ""
    start = time.monotonic()
    pitch = await esegui_pitch(domanda)
    log_query(domanda, Percorso.PITCH.value, pitch["agent_names"], pitch["content"], "COMPLETED", pitch["raw_results"], time.monotonic() - start)
    asyncio.create_task(_valuta_in_background(domanda, pitch["content"]))
    return StepOutput(content=pitch["content"])


async def _esegui_ambiguous_step(step_input: StepInput) -> StepOutput:
    domanda = step_input.get_input_as_string() or ""
    start = time.monotonic()
    result = await cinema_team.arun(domanda)
    log_query(domanda, Percorso.AMBIGUOUS.value, ["Creative Agent (coordinate, fallback)"], result.content, result.status.value, [result], time.monotonic() - start)
    # NON richiamare _valuta_in_background qui: cinema_team ha gia' quality_eval
    # come post_hook nativo (run_in_background=True), gia' scattato dentro
    # cinema_team.arun() — richiamarlo di nuovo duplicherebbe la valutazione.
    return StepOutput(content=result.content)


def _richiesta_bloccata_step(step_input: StepInput) -> StepOutput:
    domanda = step_input.get_input_as_string() or ""
    pattern = controlla_prompt_injection(domanda)
    content = f"Richiesta bloccata: rilevato pattern sospetto di prompt injection ('{pattern}')."
    log_query(domanda, "blocked", [], content, "BLOCKED", [], 0.0)
    return StepOutput(content=content)


step_graph = Step(name="Graph Query Agent", executor=_esegui_graph_step)
step_semantic = Step(name="Semantic Query Agent", executor=_esegui_semantic_step)
step_pitch = Step(name="Pitch (Graph+Semantic in parallelo + sintesi)", executor=_esegui_pitch_step)
step_ambiguous = Step(name="Team coordinate (fallback per casi ambigui)", executor=_esegui_ambiguous_step)
step_blocked = Step(name="Richiesta bloccata (prompt injection)", executor=_richiesta_bloccata_step)

_PERCORSO_TO_STEP = {
    Percorso.GRAPH: step_graph,
    Percorso.SEMANTIC: step_semantic,
    Percorso.PITCH: step_pitch,
}


def _seleziona_percorso(step_input: StepInput) -> Step:
    domanda = step_input.get_input_as_string() or ""
    if controlla_prompt_injection(domanda):
        return step_blocked
    return _PERCORSO_TO_STEP.get(classifica(domanda), step_ambiguous)


router_step = Router(
    name="Router (cosine similarity + top-k + majority voting)",
    description="Instrada la domanda al percorso diretto deciso da router.classifica(), o al Team coordinate se ambigua.",
    choices=[step_graph, step_semantic, step_pitch, step_ambiguous, step_blocked],
    selector=_seleziona_percorso,
)

versione2_workflow = Workflow(
    id="router-versione-2",
    name="Cinema Router (versione 2)",
    description="Router deterministico su embedding: instrada a Graph/Semantic Agent, al workflow di pitch, o al Team coordinate per i casi ambigui.",
    steps=[router_step],
    db=db_v2,
)

agent_os = AgentOS(
    name="Cinema GraphRAG — Router (versione 2)",
    workflows=[versione2_workflow],
    db=db_v2,
)

app = agent_os.get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agentos_main:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
        app_dir=str(Path(__file__).resolve().parent),
    )
