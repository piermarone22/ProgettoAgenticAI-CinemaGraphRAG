"""Router deterministico: classifica la domanda con regole/keyword (NESSUNA
chiamata LLM) per instradarla verso il percorso più economico possibile.

Motivazione (vedi anche architettura in root, agent.py): lì OGNI domanda passa
dal Creative Agent (Team in mode=coordinate), anche quando è ovvio a priori
quale singolo worker serve — il coordinator spende comunque un giro di
ragionamento + una tool call ('delegate_task_to_member') anche solo per
instradare una domanda banale come "In che film ha recitato Tom Hanks?" a un
unico agente. Questo router elimina quel costo per i casi in cui il pattern è
riconoscibile a priori, e usa l'architettura originale solo come eccezione.

Instradamento:
- GRAPH / SEMANTIC: la domanda riguarda chiaramente solo dati relazionali
  (grafo) o solo ricerca semantica (trame/biografie) -> l'Agent specifico
  viene chiamato DIRETTAMENTE, saltando del tutto il Creative Agent.
- PITCH: pattern riconoscibile a priori (richiesta di pitch/casting creativo),
  che per costruzione richiede SEMPRE entrambi i worker -> instradata al
  workflow deterministico parallelo (vedi pitch_workflow.py).
- AMBIGUOUS: nessuna regola scatta con sicurezza (nessun keyword riconosciuto,
  o riconosciuti keyword di ENTRAMBI i domini) -> fallback sull'architettura
  originale (Team coordinate, il Creative Agent decide da solo). Un router a
  regole fisse su una richiesta aperta/interpretabile rischierebbe di sbagliare
  instradamento; è più sicuro lasciare che sia un LLM a ragionarci.
"""

import re
from enum import Enum


class Percorso(str, Enum):
    GRAPH = "graph"
    SEMANTIC = "semantic"
    PITCH = "pitch"
    AMBIGUOUS = "ambiguous"


# Priorità massima: se la domanda chiede esplicitamente un pitch/casting
# creativo, servono sempre entrambi i worker a prescindere da altri keyword
# presenti (es. "fondendo le atmosfere" contiene anche una parola semantica,
# ma qui non deve essere instradata come "solo semantic").
_PITCH_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpitch\b",
        r"\bfond(o|i|ere|endo)\b.{0,30}\b(tram|atmosfer)",
        r"\bcasting\b",
        r"\bcast\b.{0,30}\b(collaborat|abitual|real)",
        r"\bsequel spirituale\b",
        r"\bnuovo (film|concept)\b.{0,40}\b(atmosfer|tram)",
    ]
]

_GRAPH_KEYWORDS = [
    "recitato", "recitano", "recitare", "recitazione",
    "diretto", "dirige", "diretti", "diretta",
    "regista", "registi",
    "collaborato", "collaboratori", "collaborazion",
    "film in comune", "hanno in comune",
    "quanti film", "conteggio",
    "genere", "generi",
    "rete di", "network",
    "attori collegati",
]

_SEMANTIC_KEYWORDS = [
    "trama", "trame",
    "atmosfera", "atmosfere",
    "simile a", "somiglia", "somiglianze", "ispirazion",  # stem: ispirazione/ispirato
    "narrativ",  # stem: narrativa/narrativo/narrative (struttura, stile, tono narrativo)
    "ambientato", "ambientati", "ambientata",
    "biografi",  # stem: biografia/biografie/biografica/biografico
    "vita di", "carriera",
    "racconta", "raccontami",
]


def _contains_any(testo_lower: str, parole: list[str]) -> bool:
    return any(p in testo_lower for p in parole)


def classifica(domanda: str) -> Percorso:
    if any(pattern.search(domanda) for pattern in _PITCH_PATTERNS):
        return Percorso.PITCH

    testo_lower = domanda.lower()
    graph_hit = _contains_any(testo_lower, _GRAPH_KEYWORDS)
    semantic_hit = _contains_any(testo_lower, _SEMANTIC_KEYWORDS)

    if graph_hit and not semantic_hit:
        return Percorso.GRAPH
    if semantic_hit and not graph_hit:
        return Percorso.SEMANTIC
    # Nessun keyword riconosciuto, oppure entrambi i domini citati insieme:
    # in entrambi i casi non c'e' abbastanza certezza per un instradamento
    # diretto sicuro.
    return Percorso.AMBIGUOUS


def controlla_prompt_injection(testo: str) -> str | None:
    """Ritorna il pattern sospetto trovato (per il messaggio di blocco), o None
    se il testo e' pulito. Riusa la stessa lista di pattern del pre-hook
    dell'architettura originale (app/agent.py) — qui la protezione va
    applicata ESPLICITAMENTE nel router perché i percorsi GRAPH/SEMANTIC/PITCH
    bypassano il Team e quindi il suo pre_hook automatico (che scatta solo sul
    percorso AMBIGUOUS, dove si richiama cinema_team.arun)."""
    from agent import _PROMPT_INJECTION_PATTERNS  # import locale: app/ aggiunto al sys.path da main.py

    testo_lower = testo.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in testo_lower:
            return pattern
    return None
