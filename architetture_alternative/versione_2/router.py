"""Router: classifica la domanda per instradarla verso il percorso più
economico possibile, usando similarità coseno sugli embedding — confronto
diretto con esempi etichettati, non un classificatore a keyword/regex fisse.

Motivazione (vedi anche architettura in app/agent.py): lì OGNI domanda passa
dal Creative Agent (Team in mode=coordinate), anche quando è ovvio a priori
quale singolo worker serve — il coordinator spende comunque un giro di
ragionamento + una tool call ('delegate_task_to_member') anche solo per
instradare una domanda banale come "In che film ha recitato Tom Hanks?" a un
unico agente. Questo router elimina quel costo quando il pattern è
riconoscibile a priori, e usa l'architettura originale solo come eccezione.

Come funziona, nessun training coinvolto: le domande del test set già
etichettate per categoria (testing/domande_di_test.csv) vengono embeddate una
sola volta (embedding pre-calcolati, tenuti in cache su disco — non un modello
addestrato/fittato, solo vettori). Per ogni nuova domanda si calcola il suo
embedding e lo si confronta per cosine similarity con TUTTI gli esempi noti;
si prendono i k più simili (top-k) e si fa votare la maggioranza tra le loro
etichette. Se il migliore è sotto una soglia minima di similarità, o se i k
vicini non sono d'accordo con margine sufficiente, la domanda resta AMBIGUOUS
invece di rischiare un instradamento sbagliato.

Perché questo e non un classificatore a keyword/regex fisse: un classificatore
a parole fisse riconosce solo le formulazioni previste in anticipo — qualunque
parafrasi che non usa quelle parole cade a vuoto (o peggio, matcha la keyword
sbagliata). Il confronto per similarità generalizza: una domanda formulata in
modo nuovo ma semanticamente vicina a un esempio noto viene comunque
riconosciuta, senza dover prevedere ogni possibile formulazione.

Perché non un classificatore supervisionato tipo XGBoost: con ~40 esempi
etichettati spalmati su 3 classi, un modello supervisionato overfitterebbe
sulle frasi viste invece di generalizzare — esattamente il problema che
vogliamo risolvere. Il confronto per similarità non richiede training: ogni
domanda aggiunta al test set migliora la copertura futura senza nessun
retraining, e il costo per domanda è una singola chiamata di embedding (stesso
ordine di grandezza della cache semantica delle risposte, response_cache.py in
app/, che usa la stessa tecnica per uno scopo diverso).

Instradamento:
- GRAPH / SEMANTIC / PITCH: i k vicini più simili concordano con sufficiente
  margine e il migliore supera la soglia di similarità -> instradamento
  diretto (Agent specifico, o workflow parallelo per i pitch).
- AMBIGUOUS: nessun vicino abbastanza simile, o i vicini non sono d'accordo
  tra loro -> fallback sull'architettura originale (Team coordinate). Un
  router che non sa decidere con sicurezza deve arrendersi verso il percorso
  più affidabile, non tirare a indovinare.
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


# Fonte degli esempi etichettati: UN SOLO file dedicato, esempi_classificazione.csv,
# 112 righe (categoria, domanda). E' un consolidato di due provenienze distinte:
# le 38 domande "mappabili" del test set di sistema (testing/domande_di_test.csv,
# escluse le 5 'edge_case', che non hanno un'unica etichetta coerente — mix di
# query fattuali su persone assenti dal dataset e tentativi di attacco) PIU' 74
# esempi aggiuntivi pensati apposta per la classificazione: attori/registi MAI
# citati nel test set principale, e formulazioni deliberatamente diverse dai
# template usati lì (es. non solo "In quali film ha recitato X?", ma anche
# "Qual è la filmografia di X?", "Elenco dei film con X nel cast", ecc.). Senza
# questa varietà, testare il classificatore su parafrasi delle stesse domande
# rischierebbe di misurare quanto bene riconosce il proprio stesso set
# d'esempio, non quanto generalizza davvero a formulazioni ed entità nuove.
#
# IMPORTANTE: testing/domande_confronto_holdout.csv (usato per confrontare le
# due architetture, vedi testing/FRAMEWORK_CONFRONTO.md) non contiene NESSUNA
# domanda presente qui — verificato esplicitamente, per non falsare il
# confronto facendo "riconoscere" al router le proprie stesse domande.
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

# Embedding degli esempi PRECALCOLATI offline da
# ingestion/embedding_esempi_classificazione.ipynb — lo stesso principio del
# resto della pipeline dati del progetto (ingestion.ipynb per film/biografie):
# generare embedding e' un lavoro di preparazione dati una tantum, non una
# responsabilita' del codice di servizio. Router.py qui si limita a LEGGERE
# questo file: non chiama mai l'API di embedding per gli esempi, solo per la
# domanda dell'utente in arrivo (che non puo' essere precalcolata).
_EMBEDDINGS_PATH = Path(__file__).resolve().parent / "esempi_classificazione_embeddings.json"

# k=5 scelto empiricamente sul set ampliato (112 esempi in
# esempi_classificazione.csv: 38 provenienti dal test set di sistema + 74
# aggiuntivi con entita' e formulazioni mai viste nel test set principale).
# Cross-validation leave-one-out (ogni
# esempio classificato usando gli altri 111): 96 corretti, 1 sbagliato,
# 15 ambigui — 99% di accuratezza tra le classificazioni con cui il sistema si
# e' detto confidente. Con meno esempi (solo i 38 originali) k=3 era la scelta
# migliore; ampliando il pool k=5 generalizza meglio, perche' le categorie
# piu' rare hanno ora abbastanza esempi da non essere diluite da vicini di
# classi piu' popolose.
_TOP_K = 5
# Serve maggioranza netta tra i k vicini, non una semplice pluralita': con k=5
# servono almeno 3 vicini d'accordo prima di fidarsi dell'instradamento.
_MIN_VOTI_MAGGIORANZA = _TOP_K // 2 + 1
# Sotto questa soglia di similarita', anche il vicino piu' vicino e' troppo
# lontano dal linguaggio delle domande note per fidarsene. Le similarita' top-1
# osservate in cross-validation sul set ampliato restano quasi sempre sopra
# 0.75: 0.60 resta un margine di sicurezza sotto quel range (non taglia mai
# nulla di in-dominio) mentre scarta domande genuinamente estranee al cinema.
_SOGLIA_CONFIDENZA = 0.60

_exemplars: list[tuple[str, str, list[float]]] | None = None  # (domanda, percorso.value, embedding)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from agno.knowledge.embedder.google import GeminiEmbedder
        # dimensions ridotte (rispetto alle 1536 del knowledge base principale):
        # qui serve solo confrontare domande tra loro, non fare retrieval
        # documentale — stessa scelta della cache semantica delle risposte.
        _embedder = GeminiEmbedder(
            id="gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            task_type="SEMANTIC_SIMILARITY",
            dimensions=768,
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
    """Carica (una sola volta per processo) le domande con categoria mappabile
    dal file esempi_classificazione.csv, abbinate ai relativi embedding
    PRECALCOLATI da ingestion/embedding_esempi_classificazione.ipynb. Nessuna
    chiamata all'API di embedding avviene qui: se un file di embedding manca
    o e' incompleto, la domanda mancante viene esclusa dagli esempi (con un
    avviso) invece di calcolarla al volo — per farlo bisogna rilanciare il
    notebook di ingestion, non modificare questo modulo."""
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
                print(
                    f"[router] ATTENZIONE: nessun embedding precalcolato per '{domanda[:60]}...' — "
                    "esclusa dagli esempi. Rilancia ingestion/embedding_esempi_classificazione.ipynb."
                )
                continue
            esempi.append((domanda, percorso.value, embedding))

    _exemplars = esempi
    return esempi


def classifica(domanda: str) -> Percorso:
    """Cosine similarity contro le domande etichettate del test set, con voto
    di maggioranza tra i top-k più simili. Tre condizioni devono valere tutte,
    altrimenti resta AMBIGUOUS:
    1. il vicino più simile in assoluto supera la soglia minima di confidenza;
    2. c'e' una maggioranza netta (non solo relativa) tra i top-k;
    3. il vicino più simile in assoluto E' della classe che vince la maggioranza.

    La condizione 3 e' emersa empiricamente (cross-validation leave-one-out sul
    test set): senza di essa, una domanda come "Trovami qualcosa di simile a
    Il Padrino ma [...] con un regista che sappia gestire bene questo tipo di
    atmosfera" — genuinamente ambigua tra semantica e pitch — veniva instradata
    su PITCH perche' 2 dei 3 vicini più simili erano esempi di pitch, anche se
    il vicino in assoluto più simile era un esempio semantico. Il voto di
    maggioranza da solo puo' quindi "scavalcare" un segnale forte del primo
    vicino con un numero maggiore di vicini debolmente correlati. Richiedendo
    che il primo vicino sia D'ACCORDO con la maggioranza, la cross-validation
    sul set ampliato (112 esempi in esempi_classificazione.csv)
    da' 96 corretti/1 sbagliato/15 ambigui — 99% di accuratezza tra le
    classificazioni con cui il sistema si e' detto confidente, e le domande
    genuinamente incerte vengono dirottate correttamente sul percorso sicuro
    invece che instradate a caso."""
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
    """Ritorna il pattern sospetto trovato (per il messaggio di blocco), o None
    se il testo e' pulito. Riusa la stessa lista di pattern del pre-hook
    dell'architettura originale (app/agent.py) — qui la protezione va
    applicata ESPLICITAMENTE nel router perché i percorsi GRAPH/SEMANTIC/PITCH
    bypassano il Team e quindi il suo pre_hook automatico (che scatta solo sul
    percorso AMBIGUOUS, dove si richiama cinema_team.arun).

    Rimane deliberatamente basato su pattern fissi, non su similarità: un
    controllo di sicurezza deve restare prevedibile e verificabile, non
    soggetto a una soglia di confidenza che un testo malevolo potrebbe imparare
    ad aggirare restando semanticamente lontano dagli esempi noti."""
    from agent import _PROMPT_INJECTION_PATTERNS  # import locale: app/ aggiunto al sys.path da main.py

    testo_lower = testo.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in testo_lower:
            return pattern
    return None
