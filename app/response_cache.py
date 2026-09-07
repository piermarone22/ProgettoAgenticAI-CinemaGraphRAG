"""Cache locale su disco per le risposte del team.

Se una domanda semanticamente equivalente a una gia' risposta con successo di
recente viene riposta, la si riserve senza rieseguire l'intera pipeline
multi-agente — utile soprattutto per le domande piu' pesanti (pitch completi:
1-2 minuti anche con la parallelizzazione dei worker) rilanciate piu' volte
durante test/demo, anche se riformulate in modo diverso.

Solo le risposte con status COMPLETED vengono messe in cache: un 429/503
transitorio non deve restare "congelato" per le richieste successive.

Corrispondenza in due passi:
1. Match ESATTO sul testo (stesso team, stessa stringa normalizzata): nessuna
   chiamata di embedding, risposta immediata.
2. Match SEMANTICO: se non c'e' un match esatto, la domanda viene confrontata
   (cosine similarity sugli embedding Gemini) con le domande gia' in cache.
   Per evitare falsi positivi pericolosi — es. "Quali film ha diretto Nolan?"
   vs "Quali film ha diretto Tarantino?" sono strutturalmente quasi identiche
   ma richiedono risposte diverse — il match semantico scatta SOLO se anche
   l'insieme dei nomi propri (entita') estratti dalle due domande coincide
   esattamente. La similarita' di embedding da sola non e' un segnale
   sufficiente quando l'unica differenza tra due domande e' un nome proprio.
"""

import json
import math
import os
import re
import time
from pathlib import Path

# .parent.parent: questo file vive in app/, ma tmp/ e' alla radice del progetto.
CACHE_PATH = Path(__file__).resolve().parent.parent / "tmp" / "response_cache.json"

# 1 ora: sufficiente per una sessione di test/demo, abbastanza corta da non
# rischiare risposte stantie se nel frattempo i dati sottostanti (Neo4j/ChromaDB,
# es. l'ingestion delle biografie ancora in corso) cambiano.
CACHE_TTL_SECONDS = 3600

# Sotto questa soglia di cosine similarity, due domande (a parita' di entita'
# citate) sono considerate diverse. Scelta alta perche' il gate sulle entita'
# gia' elimina il rischio principale di falsi positivi: qui serve solo a
# scartare domande che citano le stesse persone/film ma chiedono cose diverse
# (es. "film diretti da" vs "biografia di" sullo stesso regista).
SIMILARITY_THRESHOLD = 0.90

_cache: dict[str, dict] | None = None
_embedder = None

# Parole che iniziano tipicamente una domanda in italiano: escluse dall'euristica
# di estrazione delle entita' anche se maiuscole (altrimenti ogni domanda che
# inizia per "Quali", "Chi", "Trova", ecc. genererebbe una falsa "entita'").
_STOPWORDS_INIZIALI = {
    "in", "quali", "quale", "quanti", "quante", "chi", "trova", "genera",
    "con", "raccontami", "dammi", "ci", "esegui", "è", "e", "il", "la",
    "un", "una", "come", "dove", "cosa", "che", "per", "sono",
}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from agno.knowledge.embedder.google import GeminiEmbedder
        # dimensions ridotte (rispetto alle 1536 del KB principale): qui serve
        # solo a confrontare domande tra loro, non a fare retrieval documentale.
        _embedder = GeminiEmbedder(
            id="gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            task_type="SEMANTIC_SIMILARITY",
            dimensions=768,
        )
    return _embedder


def _extract_entities(text: str) -> frozenset[str]:
    """Euristica: sequenze di parole capitalizzate (nomi propri di persone/film),
    escludendo la prima parola della domanda e le stopword interrogative note."""
    words = text.strip().split()
    entita: set[str] = set()
    corrente: list[str] = []
    for i, w in enumerate(words):
        nucleo = re.sub(r"[^\wàèéìòùÀÈÉÌÒÙ'-]", "", w)
        capitalizzata = nucleo[:1].isupper() and nucleo.lower() not in _STOPWORDS_INIZIALI
        if capitalizzata and i > 0:
            corrente.append(nucleo)
        else:
            if len(corrente) >= 1:
                entita.add(" ".join(corrente).lower())
            corrente = []
    if len(corrente) >= 1:
        entita.add(" ".join(corrente).lower())
    return frozenset(entita)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save() -> None:
    assert _cache is not None
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _key(team_id: str, message: str) -> str:
    return f"{team_id}::{message.strip()}"


def _not_expired(entry: dict) -> bool:
    return time.time() - entry.get("cached_at", 0) <= CACHE_TTL_SECONDS


def get(team_id: str, message: str) -> dict | None:
    """Restituisce la voce di cache (dict con content/status/model/token/...) se
    presente e non scaduta — per match esatto o semantico —, altrimenti None."""
    cache = _load()

    exact = cache.get(_key(team_id, message))
    if exact is not None and _not_expired(exact):
        return exact

    entita_nuova = _extract_entities(message)
    if not entita_nuova:
        # Nessuna entita' riconosciuta nella domanda: troppo rischioso tentare
        # un match semantico senza un ancoraggio su nomi propri.
        return None

    candidati = [
        (k, v) for k, v in cache.items()
        if k.startswith(f"{team_id}::") and _not_expired(v)
        and frozenset(v.get("entita", [])) == entita_nuova
    ]
    if not candidati:
        return None

    try:
        embedding_nuova = _get_embedder().get_embedding(message)
    except Exception as exc:
        print(f"[ResponseCache] Errore embedding per match semantico: {exc}")
        return None
    if not embedding_nuova:
        return None

    migliore, punteggio_migliore = None, 0.0
    for _, entry in candidati:
        embedding_entry = entry.get("embedding")
        if not embedding_entry:
            continue
        punteggio = _cosine_similarity(embedding_nuova, embedding_entry)
        if punteggio > punteggio_migliore:
            migliore, punteggio_migliore = entry, punteggio

    if migliore is not None and punteggio_migliore >= SIMILARITY_THRESHOLD:
        return migliore
    return None


def put(team_id: str, message: str, extracted: dict) -> None:
    """Salva una risposta estratta (vedi query_logger._extract_from_*) in cache,
    insieme a embedding ed entita' della domanda per abilitare il match semantico."""
    cache = _load()
    try:
        embedding = _get_embedder().get_embedding(message)
    except Exception as exc:
        print(f"[ResponseCache] Errore embedding in salvataggio (si salva senza): {exc}")
        embedding = []
    entita = sorted(_extract_entities(message))
    cache[_key(team_id, message)] = {
        **extracted,
        "cached_at": time.time(),
        "embedding": embedding,
        "entita": entita,
    }
    _save()
