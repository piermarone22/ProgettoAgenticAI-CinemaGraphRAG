"""Router: classifica la domanda (graph/semantic/pitch/ambiguous) per
instradarla direttamente all'agente giusto, senza passare dal Creative Agent.
Usa cosine similarity contro esempi etichettati (nessun training, nessuna
keyword fissa) — vedi il docstring di classifica() per i dettagli.
"""

import csv
import json
import math
import os
from collections import Counter
from enum import Enum
from pathlib import Path


class Percorso(str, Enum):
    GRAPH = "graph"
    SEMANTIC = "semantic"
    PITCH = "pitch"
    AMBIGUOUS = "ambiguous"


# Esempi etichettati per la classificazione: 112 righe (categoria, domanda) in
# esempi_classificazione.csv. Non contiene nessuna domanda presente in
# testing/domande_confronto_holdout.csv (usato per confrontare le architetture),
# per non far "riconoscere" al router le proprie stesse domande di test.
_EXEMPLARS_PATH = Path(__file__).resolve().parent / "esempi_classificazione.csv"
_CATEGORIA_TO_PERCORSO = {
    "attore_film": Percorso.GRAPH,
    "regista_film": Percorso.GRAPH,
    "collaborazione_attori": Percorso.GRAPH,
    "collaboratori_regista": Percorso.GRAPH,
    "conteggio_genere": Percorso.GRAPH,
    "network_genere": Percorso.GRAPH,
    "semantica_trame": Percorso.SEMANTIC,
    "semantica_biografia": Percorso.SEMANTIC,
    "pitch_completo": Percorso.PITCH,
}

# Embedding degli esempi precalcolati offline da
# ingestion/embedding_esempi_classificazione.ipynb: il router li legge soltanto,
# non chiama mai l'API di embedding per loro (solo per la domanda in arrivo).
_EMBEDDINGS_PATH = Path(__file__).resolve().parent / "esempi_classificazione_embeddings.json"

_TOP_K = 5  # scelto via cross-validation leave-one-out su 112 esempi (96 corretti/1 sbagliato/15 ambigui)
_MIN_VOTI_MAGGIORANZA = _TOP_K // 2 + 1  # maggioranza netta richiesta tra i top-k
_SOGLIA_CONFIDENZA = 0.60  # sotto questa similarità, il vicino migliore è troppo lontano per fidarsene

_exemplars: list[tuple[str, str, list[float]]] | None = None  # (domanda, percorso.value, embedding)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from agno.knowledge.embedder.google import GeminiEmbedder
        _embedder = GeminiEmbedder(
            id="gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            task_type="SEMANTIC_SIMILARITY",
            dimensions=768,  # bastano per confrontare domande tra loro, non serve il retrieval documentale
        )
    return _embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_exemplars() -> list[tuple[str, str, list[float]]]:
    """Carica gli esempi etichettati e i loro embedding precalcolati (una sola
    volta per processo)."""
    global _exemplars
    if _exemplars is not None:
        return _exemplars

    if not _EMBEDDINGS_PATH.exists():
        raise RuntimeError(
            f"File di embedding non trovato: {_EMBEDDINGS_PATH}. "
            "Esegui prima ingestion/embedding_esempi_classificazione.ipynb per generarlo."
        )
    embeddings = json.loads(_EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    esempi = []
    with _EXEMPLARS_PATH.open(newline="", encoding="utf-8") as f:
        for riga in csv.DictReader(f):
            percorso = _CATEGORIA_TO_PERCORSO.get(riga["categoria"])
            if percorso is None:
                continue
            domanda = riga["domanda"]
            embedding = embeddings.get(domanda)
            if embedding is None:
                print(f"[router] ATTENZIONE: nessun embedding per '{domanda[:60]}...' — esclusa.")
                continue
            esempi.append((domanda, percorso.value, embedding))

    _exemplars = esempi
    return esempi


def classifica(domanda: str) -> Percorso:
    """Cosine similarity contro gli esempi etichettati, con voto di
    maggioranza tra i top-k più simili. Instrada solo se: il vicino migliore
    supera la soglia di confidenza, c'è maggioranza netta tra i top-k, e il
    vicino migliore è della classe vincente (altrimenti resta AMBIGUOUS)."""
    esempi = _get_exemplars()
    if not esempi:
        return Percorso.AMBIGUOUS

    try:
        embedding_domanda = _get_embedder().get_embedding(domanda)
    except Exception:
        return Percorso.AMBIGUOUS
    if not embedding_domanda:
        return Percorso.AMBIGUOUS

    scored = sorted(
        ((_cosine_similarity(embedding_domanda, emb), percorso_value) for _, percorso_value, emb in esempi),
        key=lambda x: x[0],
        reverse=True,
    )
    top_k = scored[:_TOP_K]

    similarita_migliore, percorso_piu_vicino = top_k[0]
    if similarita_migliore < _SOGLIA_CONFIDENZA:
        return Percorso.AMBIGUOUS

    voti = Counter(percorso_value for _, percorso_value in top_k)
    percorso_vincente, n_voti = voti.most_common(1)[0]
    if n_voti < _MIN_VOTI_MAGGIORANZA:
        return Percorso.AMBIGUOUS
    if percorso_piu_vicino != percorso_vincente:
        return Percorso.AMBIGUOUS

    return Percorso(percorso_vincente)


def controlla_prompt_injection(testo: str) -> str | None:
    """Ritorna il pattern sospetto trovato, o None se il testo è pulito.
    Riusa i pattern del pre-hook originale (app/agent.py): qui va applicato
    esplicitamente perché i percorsi diretti bypassano il pre_hook del Team."""
    from agent import _PROMPT_INJECTION_PATTERNS  # import locale: app/ aggiunto al sys.path da main.py

    testo_lower = testo.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in testo_lower:
            return pattern
    return None
