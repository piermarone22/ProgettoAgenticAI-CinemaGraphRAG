"""Workflow deterministico per i pitch: chiama Graph Query Agent e Semantic
Query Agent in parallelo (asyncio.gather), poi un agente di sintesi senza
tool fonde i due risultati nel pitch finale.
"""

import sys
from pathlib import Path

# .parent.parent.parent: questo file vive in architetture_alternative/versione_2/;
# agent.py vive in app/, alla radice del progetto.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "app"))

import asyncio  

from agno.agent import Agent  
from agno.models.fallback import FallbackConfig  

from agent import graph_agent, semantic_agent, _gemini_worker, _groq_fallback  

_synthesis_agent = Agent(
    name="Pitch Synthesizer",
    model=_gemini_worker(),
    fallback_config=FallbackConfig(on_rate_limit=[_groq_fallback()], on_error=[_groq_fallback()]),
    instructions=[
        "Ricevi due blocchi di dati già raccolti da due specialisti: elementi narrativi "
        "(trame/atmosfere, da ricerca semantica) e dati di network (collaborazioni reali, dal grafo). "
        "Non hai alcun tool: il tuo unico compito è fondere questi dati in un pitch creativo coerente.",
        "Genera un nuovo concept di film fondendo gli elementi narrativi ricevuti con creatività.",
        "Il cast proposto deve essere scelto ESCLUSIVAMENTE tra le collaborazioni realmente riportate "
        "nei dati di network ricevuti, motivandolo con i numeri reali (es. 'N film insieme').",
        "CRITICO: non inventare mai nomi, titoli o numeri di collaborazione non presenti nei dati ricevuti. "
        "Se i dati di network sono vuoti o insufficienti, dillo esplicitamente invece di inventare un cast.",
        "Non spiegare da dove vengono i dati né il processo seguito.",
        "Rispondi in italiano, in modo diretto.",
    ],
    markdown=True,
)


async def esegui_pitch(domanda: str) -> dict:
    """Esegue Graph Agent e Semantic Agent in parallelo, poi sintetizza il pitch finale."""
    semantic_result, graph_result = await asyncio.gather(
        semantic_agent.arun(domanda),
        graph_agent.arun(domanda),
    )

    prompt_sintesi = (
        f"Richiesta originale dell'utente: {domanda}\n\n"
        f"--- Dati semantici (trame/atmosfere) ---\n{semantic_result.content}\n\n"
        f"--- Dati relazionali (network/collaborazioni reali) ---\n{graph_result.content}\n\n"
        "Genera il pitch finale fondendo questi dati, secondo le tue istruzioni."
    )
    final = await _synthesis_agent.arun(prompt_sintesi)

    return {
        "content": final.content,
        "agent_names": ["Semantic Query Agent", "Graph Query Agent", "Pitch Synthesizer"],
        "raw_results": [semantic_result, graph_result, final],
    }
