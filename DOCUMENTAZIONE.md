# Cinema GraphRAG — Documentazione tecnica completa

Documentazione approfondita di tutte le funzionalità del progetto e delle scelte
progettuali fatte in fase di sviluppo, con relative motivazioni. Pensata come
riferimento unico per la relazione/presentazione d'esame e per chi riprende il
progetto in futuro.

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Dati e ingestion](#2-dati-e-ingestion)
3. [Team multi-agente](#3-team-multi-agente)
4. [Modelli LLM: scelta, retry, timeout](#4-modelli-llm-scelta-retry-timeout)
5. [Fallback multi-provider (Groq)](#5-fallback-multi-provider-groq)
6. [Guardrail di sicurezza](#6-guardrail-di-sicurezza)
7. [Memoria conversazionale](#7-memoria-conversazionale)
8. [Cache delle risposte](#8-cache-delle-risposte)
9. [Parallelizzazione dei worker](#9-parallelizzazione-dei-worker)
10. [Osservabilità: logging e dashboard](#10-osservabilità-logging-e-dashboard)
11. [Valutazione (LLM-as-a-Judge)](#11-valutazione-llm-as-a-judge)
12. [Test set](#12-test-set)
13. [Architettura alternativa: router deterministico (versione_2)](#13-architettura-alternativa-router-deterministico-versione_2)
14. [Confronto tra le due architetture](#14-confronto-tra-le-due-architetture)
15. [Organizzazione del repository](#15-organizzazione-del-repository)
16. [Limiti noti del sistema](#16-limiti-noti-del-sistema)
17. [Sviluppi futuri](#17-sviluppi-futuri)

---

## 1. Panoramica

**Cinema GraphRAG** è un sistema di intelligenza artificiale cinematografica basato
su un'architettura **GraphRAG ibrida**: combina un knowledge graph (Neo4j, relazioni
strutturali tra film/attori/registi/generi) con un database vettoriale (ChromaDB,
ricerca semantica su trame e biografie), orchestrati da un **team multi-agente**
(framework Agno) che risponde sia a domande fattuali precise sia a richieste
creative (pitch di nuovi film, fusioni di trame, casting motivato da dati reali).

Il sistema è esposto via **FastAPI/AgentOS**, osservabile tramite un middleware di
logging e una **dashboard Streamlit**, ed è stato validato empiricamente — mai
per plausibilità della risposta, sempre per confronto diretto con i dati sorgente
in Neo4j/ChromaDB — su un set di 43 domande di test categorizzate.

Il progetto include anche una **seconda architettura sperimentale** (§13),
costruita per confrontare l'approccio "tutto attraverso il coordinatore"
dell'architettura principale con un **router deterministico** che instrada
ogni domanda al percorso giusto (grafo, semantica, pitch parallelo) per
similarità su embedding, **prima** di qualunque chiamata LLM — evitando il
costo di orchestrazione del coordinatore quando non serve. Le due architetture
sono state confrontate empiricamente su token, tempo e qualità (§14).

### Stack tecnologico

| Livello | Tecnologia | Perché |
|---|---|---|
| Dati strutturali | Neo4j | Le relazioni (collaborazioni, filmografie, network) sono il caso d'uso naturale di un grafo: query come "attori che hanno lavorato più spesso con un regista" sono innaturali in SQL relazionale, immediate in Cypher. |
| Dati semantici | ChromaDB + `gemini-embedding-001` | Trame e biografie sono testo libero: la somiglianza "atmosfera", "tono narrativo" non è esprimibile con filtri strutturati, richiede ricerca vettoriale. |
| Orchestrazione agenti | Agno (`Team`, `Agent`, hook, eval) | Framework agentic con supporto nativo a delega multi-agente, hook pre/post-run, valutazione LLM-as-judge integrata, fallback tra provider — copre tutti i requisiti senza codice custom di orchestrazione. |
| LLM primario | Google Gemini | Free tier generoso per lo sviluppo iterativo di un progetto d'esame. |
| LLM di fallback | Groq (`qwen/qwen3.8-27b`) | Provider alternativo gratuito, scelto dopo confronto empirico diretto (vedi §5). |
| API | FastAPI via AgentOS | Generata automaticamente da Agno a partire dalla definizione del team, con supporto nativo a streaming SSE. |
| Dashboard | Streamlit + Plotly | Prototipazione rapida di UI dati, sufficiente per un tool di osservabilità interno. |
| Dataset | TMDB 5000 (film + credits) + biografie da TMDB API | Dataset pubblico strutturato, sufficientemente ricco (relazioni multiple attore-film-regista-genere) per un caso d'uso GraphRAG non banale. |

---

## 2. Dati e ingestion

### 2.1 — Schema Neo4j

```
(Film {vector_id, titolo, anno, trama})
(Actor {nome, tmdb_id, biografia, data_nascita, luogo_nascita})
(Director {stessi campi di Actor})
(Genre {nome})

(:Actor)-[:ACTED_IN]->(:Film)
(:Director)-[:DIRECTED]->(:Film)
(:Film)-[:HAS_GENRE]->(:Genre)
```

Vincoli di unicità su `Film.vector_id`, `Director.nome`, `Actor.nome`, `Genre.nome`
(cella 4 del notebook), applicati **prima** del popolamento per evitare nodi
duplicati durante un'ingestion interrotta e ripresa più volte (vedi checkpointing
sotto) — senza il vincolo, una riesecuzione parziale avrebbe creato doppioni ogni
volta che un nodo persona/genere già esistente veniva reincontrato in un film
diverso.

### 2.2 — ChromaDB e scelte sull'embedding

- **Modello**: `gemini-embedding-001`, **1536 dimensioni**. Scelta motivata da un
  compromesso qualità/costo: dimensionalità sufficiente a distinguere trame con
  temi/atmosfere sottili (es. "isolamento nello spazio" vs "isolamento in una
  casa"), senza il costo di storage e calcolo di embedding a dimensionalità
  massima, non necessario per un dataset di qualche migliaio di documenti.
- **Task type asimmetrico**: `RETRIEVAL_DOCUMENT` in fase di ingestion (i testi
  salvati) e `RETRIEVAL_QUERY` in fase di ricerca (le domande dell'utente, vedi
  `app/agent.py`, `_vector_db`). Questo è lo schema **corretto** per un caso d'uso di
  retrieval asimmetrico: l'API di embedding di Google produce vettori ottimizzati
  diversamente a seconda che il testo sia un documento da indicizzare o una query
  breve da confrontare con quei documenti — usare lo stesso task type per
  entrambi (es. `SEMANTIC_SIMILARITY`) sarebbe corretto solo per confronti
  simmetrici testo-testo (motivo per cui quel task type viene invece usato nella
  cache semantica delle risposte, §8, dove si confrontano due domande tra loro).
- **Nessun chunking**: ogni trama/biografia è indicizzata come **documento
  intero**, non spezzata in passaggi più piccoli. Motivazione: le trame TMDB sono
  sintesi già brevi (poche frasi), e le biografie recuperate da TMDB API sono di
  lunghezza contenuta — il chunking avrebbe introdotto complessità (gestione di
  passaggi parziali, aggregazione dei risultati) senza un beneficio concreto sulla
  qualità del retrieval a questa scala di testo. Per documenti molto più lunghi
  (es. sceneggiature complete) sarebbe stato necessario.
- **Un documento per film + un documento per persona**: il documento-film è
  arricchito con titolo/regista/cast/generi oltre alla trama, cosicché una singola
  ricerca semantica restituisca già il contesto necessario senza una query
  aggiuntiva a Neo4j.

### 2.3 — Pipeline di ingestion (`ingestion/ingestion.ipynb`)

Notebook strutturato in 12 sezioni sequenziali (imports → connessioni → pulizia
CSV → schema Neo4j → funzioni di ingestion → popolamento Neo4j → preparazione
documenti ChromaDB → embedding ChromaDB → verifica → biografie TMDB → export →
riepilogo). Le sezioni più significative dal punto di vista progettuale:

- **Checkpointing su ChromaDB** (cella "Embedding + Inserimento"): prima di
  ogni upsert si verifica `vector_db.content_hash_exists(content_hash)` — se il
  documento è già stato embeddato in una sessione precedente, viene saltato.
  Questo rende l'intero processo **idempotente e resumibile**: fondamentale
  perché il free tier di Gemini ha reso l'ingestion un processo di **giorni**
  anziché minuti (quote giornaliere di 15-20 richieste/minuto a seconda del
  modello), quindi il notebook doveva poter essere interrotto e ripreso decine
  di volte senza mai re-embeddare (e ri-pagare in quota) ciò che era già stato
  inserito.
- **Retry con backoff esponenziale sui 429** durante l'embedding:
  `MAX_RETRIES=3`, `BASE_RETRY_SLEEP=35s` raddoppiato a ogni tentativo, con
  uscita pulita (`stop=True`) quando la quota giornaliera risultava esaurita —
  la sessione successiva riparte automaticamente dal punto di interruzione grazie
  al checkpointing sopra.
- **Biografie recuperate dinamicamente da TMDB API** (non presenti nel CSV
  originale) e imbeddate in un secondo momento (sezione 10), con una cella
  dedicata (10a-bis) per **ricostruire i documenti da Neo4j senza richiamare
  l'API TMDB** — utile per riprendere l'embedding dopo un riavvio del kernel
  senza rifare le chiamate di rete già completate.
- **Percorsi assoluti** per `.env` e la directory ChromaDB (`PROJECT_ROOT / "tmp"
  / "chromadb"`), risolti tramite un helper `_find_project_root()`: necessario
  perché il notebook viene eseguito da directory di lavoro diverse a seconda
  dell'IDE/kernel usato, e un percorso relativo avrebbe silenziosamente creato
  un secondo database vettoriale vuoto in una posizione sbagliata.

### 2.4 — Export

`neo4j_export/export_neo4j.py` esporta l'intero grafo (nodi + relazioni) in
CSV/JSON — pensato per condividere i dati di progetto (es. con il team d'esame)
senza dover distribuire un dump binario di Neo4j o richiedere a chi riceve il
progetto di rifare l'intera ingestion (giorni di lavoro per via delle quote).

---

## 3. Team multi-agente

Il cuore del sistema è definito in `app/agent.py`: un `Team` Agno in modalità
**`coordinate`**, con un coordinatore e due agenti worker specializzati.

```
                        ┌─────────────────────────┐
                        │   Cinema Creative Team   │  (coordinator)
                        └──────────┬───────────────┘
                    delega task     │     delega task
              ┌───────────────────┐│┌────────────────────────┐
              │ Graph Query Agent │ │ Semantic Query Agent     │
              │ (Neo4j / Cypher)  │ │ (ChromaDB / embedding)   │
              └───────────────────┘ └──────────────────────────┘
```

### 3.1 — Perché `mode=coordinate` e non `route`/`broadcast`

Agno espone diverse modalità di team. `coordinate` è stata scelta perché il
caso d'uso più complesso e distintivo del progetto — il **pitch creativo** —
richiede che il coordinatore riceva le risposte di **entrambi** i worker e le
**sintetizzi** in un output nuovo (fusione di trame + casting motivato), non che
si limiti a instradare la richiesta a un solo agente (`route`) o a inoltrarla
identica a tutti (`broadcast`). Il coordinatore in modalità `coordinate` non ha
accesso diretto ai database: delega sempre ai due specialisti e sintetizza.

### 3.2 — Graph Query Agent

Sette tool custom su Neo4j (via driver `neo4j`, query Cypher parametriche) più
`ReasoningTools`:

| Tool | Scopo | Perché esiste come tool dedicato |
|---|---|---|
| `cerca_film_per_attore` | Filmografia di un attore | Query base, la più frequente |
| `cerca_film_con_attori` | Film in comune tra 2+ attori | Un'intersezione tra insiemi di film è complessa da esprimere in linguaggio naturale per un LLM che genera Cypher libero — un tool dedicato con query Cypher testata garantisce correttezza |
| `cerca_film_per_regista` | Filmografia di un regista | — |
| `collaboratori_frequenti_regista` | Attori più ricorrenti con un regista, con conteggio | Base per il "casting motivato da dati reali" richiesto nei pitch |
| `conta_film_per_genere` | Conteggio film di una persona in un genere | — |
| `trova_attori_per_network` | Attori con esperienza in un genere, connessi indirettamente (tramite film in comune) al network di un regista | Query di **secondo grado** sul grafo (ponte attore→film→attore), pensata specificamente per proporre casting realistici in un pitch (es. "trova attori esperti di horror collegati al giro di James Cameron") |
| `esegui_query_cypher` | Query Cypher libera per casi non coperti | Via di fuga controllata: **blocklist** su `CREATE/DELETE/MERGE/SET/DROP/REMOVE/DETACH` applicata prima dell'esecuzione — il sistema resta di sola lettura anche quando l'agente genera Cypher arbitrario |

**Istruzione critica**: "Rispondi SOLO con dati estratti dai tool. Non inventare
mai film, attori o relazioni." — nonostante questo, sono stati riscontrati casi
di violazione (documentati in §16): l'istruzione riduce ma non elimina il
rischio di allucinazione.

### 3.3 — Semantic Query Agent

Due tool custom su ChromaDB (`cerca_trame_simili`, `cerca_biografia`) più accesso
diretto alla `Knowledge` (per `search_knowledge=True`) e `ReasoningTools`.
Istruito a **estrarre gli elementi narrativi chiave** (tono, conflitto centrale,
ambientazione, arco del personaggio) dalle trame recuperate, non solo a
restituirle grezze — necessario perché il coordinatore le userà come materiale
grezzo per generare pitch creativi.

### 3.4 — `ReasoningTools` su entrambi i worker

Aggiunto a entrambi gli agenti (`add_instructions=True`) per scomporre domande
complesse (es. un pitch che richiede più passaggi logici: identificare gli
elementi di due trame, poi fondere) in step espliciti prima di eseguire le
chiamate ai tool, migliorando l'affidabilità su richieste multi-step rispetto a
una singola chiamata diretta.

---

## 4. Modelli LLM: scelta, retry, timeout

```python
_RETRY_KWARGS = dict(retries=3, delay_between_retries=15, exponential_backoff=True, timeout=30)
```

- **Coordinator e worker usano modelli Gemini distinti** (`_gemini_worker()` =
  `gemini-3.5-flash-lite`): quote del free tier separate per modello, così il
  coordinator non esaurisce la stessa quota condivisa con i due worker che
  fanno tool-calling molto più frequente. `gemini-2.0-flash` e
  `gemini-2.5-flash-lite` sono stati **ritirati** da Google per i nuovi utenti
  durante lo sviluppo — Google reindirizza automaticamente a
  `gemini-3.5-flash-lite`, motivo per cui il progetto è stato migrato a quel
  modello.
- **`retries=3, delay_between_retries=15, exponential_backoff=True`**: il
  client Gemini di Agno non ritenta di default (`retries=0`). I `retryDelay`
  osservati empiricamente sul free tier sono quasi sempre sotto i 60 secondi
  (finestra RPM a scorrimento), quindi 3 tentativi con backoff esponenziale a
  partire da 15s coprono la stragrande maggioranza dei 429 transitori senza
  richiedere l'intervento del fallback.
- **`timeout=30`**: introdotto dopo un disservizio reale di Gemini ("high
  demand", HTTP 503) durato **circa 3 ore e 23 minuti**. Senza un timeout
  esplicito, una singola chiamata poteva restare appesa per minuti, e poiché
  retry/fallback scattano solo **dopo** che un tentativo fallisce, un tentativo
  che non fallisce mai (resta solo appeso) non fa scattare nessuno dei due
  meccanismi di resilienza — nel caso reale osservato, questo ha causato il
  **freeze dell'intero server**, incluse route non correlate come `/config`.
  Fix verificato con un test di 40 iterazioni su `/config` durante una query
  pesante in corso, confermando risposte HTTP 200 costanti dopo l'introduzione
  del timeout.

---

## 5. Fallback multi-provider (Groq)

```python
fallback_config=FallbackConfig(on_rate_limit=[_groq_fallback()], on_error=[_groq_fallback()])
```

Applicato sia al coordinatore sia a entrambi i worker: se Gemini va in rate
limit (429) o è indisponibile (503) anche dopo i retry, la stessa run passa a
Groq senza intervento manuale.

### 5.1 — Processo di selezione del modello di fallback

La scelta non si è basata sulla documentazione ufficiale (che non riporta
affidabilità nel tool-calling), ma su un **confronto empirico diretto**: stesso
prompt, stessi tool reali, modelli diversi.

| Modello testato | Esito | Motivo di scarto/scelta |
|---|---|---|
| `gpt-oss-20b` | Scartato | Inaffidabile nel tool-calling, mai arrivato a produrre una run completa valida |
| `gpt-oss-120b` | Scartato | **Corrompe i nomi propri multi-parola** negli argomenti dei tool: "Francis Ford Coppola" → "Francis ?", "Woody Allen" → "Woody ?" — riprodotto 2/2 volte con prompt diversi, causa crash o risposte vuote |
| `groq/compound` | Scartato | Non supporta affatto il tool calling (errore diretto dall'API) |
| `qwen/qwen3.8-27b` | **Scelto** | Tool-calling corretto, dati verificati esatti contro Neo4j. Unico problema: è un modello "thinking" che lascia trapelare tag `<think>`/`<parameter>` grezzi nella risposta finale |

Il problema del `qwen` è stato risolto con `request_params={"reasoning_format":
"hidden"}`, un parametro passthrough dell'API Groq che nasconde il reasoning
interno dalla risposta finale — dopo questa correzione il modello produce
output pulito e affidabile, scelto come fallback definitivo.

Questo processo empirico è di per sé una scelta metodologica degna di nota: per
un sistema **agentic** (dove il modello non genera solo testo ma argomenti
strutturati per funzioni), l'affidabilità del tool-calling non è deducibile
dai benchmark generici pubblicati dai provider, va verificata con gli stessi
tool reali del proprio sistema.

---

## 6. Guardrail di sicurezza

### 6.1 — Pre-hook: rilevamento prompt injection

```python
def check_prompt_injection(run_input: TeamRunInput) -> None:
    ...
    raise InputCheckError(..., check_trigger=CheckTrigger.PROMPT_INJECTION)
```

Lista di pattern IT/EN (tentativi di sovrascrivere le istruzioni: "ignora le
istruzioni precedenti", "you are now", "reveal your system prompt", ecc.) più
pattern specifici del dominio (tentativi di far scrivere/cancellare dati via
Cypher). Il pre-hook viene eseguito **prima di qualunque chiamata al modello**:
verificato empiricamente che una richiesta bloccata impiega **~2ms** e non
consuma quota LLM.

**Scelta implementativa rilevante**: durante lo sviluppo è stato verificato nel
sorgente di Agno che un semplice `raise Exception(...)` nel pre-hook **non**
blocca effettivamente la run (viene gestito diversamente dal framework) —
solo sollevare `InputCheckError` con `check_trigger=CheckTrigger.PROMPT_INJECTION`
produce il blocco reale. Un dettaglio facile da implementare in modo silenziosamente
errato senza una verifica diretta.

### 6.2 — Blocklist Cypher

Nel tool `esegui_query_cypher` (l'unica via di fuga a Cypher libero concessa al
Graph Agent): blocca `CREATE, DELETE, MERGE, SET, DROP, REMOVE, DETACH` prima
dell'esecuzione, restituendo un errore esplicito invece di eseguire la query.
Verificato con un test dedicato (`MATCH (f:Film) DETACH DELETE f`) nel test set.

**Limite noto** (vedi anche §16): la protezione è applicata a livello di tool
(pattern-matching sulla query generata), non a livello di permessi del database
— non esiste un utente Neo4j dedicato in sola lettura.

---

## 7. Memoria conversazionale

**Aggiunta più recente al sistema.** Fino a questo intervento, ogni domanda
veniva processata in isolamento: anche riusando lo stesso `session_id` tra due
chiamate API consecutive (come farebbe un client di chat reale), il
coordinatore non aveva alcuna consapevolezza delle domande/risposte precedenti
della stessa sessione, perché `add_history_to_context` — il parametro di Agno
che decide se la history della sessione viene iniettata nel contesto del
modello — ha **default `False`**, e `cinema_team` non lo sovrascriveva.

Fix in `app/agent.py`:

```python
cinema_team = Team(
    ...
    add_history_to_context=True,
    num_history_runs=5,   # finestra di contesto: ultimi 5 scambi
)
```

`num_history_runs=5` limita la finestra di contesto per non far esplodere il
conteggio di token su conversazioni lunghe (vedi anche §14 sul limite di questa
finestra fissa).

**Verifica empirica** (due sessioni di test con `session_id` fisso):

1. *"Chi ha diretto Interstellar?"* → *"Christopher Nolan"*, poi *"Con quale
   attore ha collaborato più spesso?"* → *"Michael Caine, 5 film"* — il pronome
   "ha" è stato risolto correttamente su Nolan senza mai nominarlo nella seconda
   domanda. Confermato contro Neo4j: Michael Caine, esattamente 5 film con Nolan.
2. *"Quali film ha diretto Quentin Tarantino?"* → lista di 8 film, poi *"In
   quali di questi film ha recitato Samuel L. Jackson?"* → 4 film corretti,
   filtrando l'intera lista precedente (non solo una singola entità, un
   riferimento pronominale più complesso). Confermato contro Neo4j: 4/4 titoli
   esatti.

**Nota su logging e valutazione**: la memoria conversazionale non cambia il
comportamento di logging o del giudice automatico — ogni domanda produce
comunque **una riga distinta** in `tmp/query_log.csv` e **una valutazione
LLM-as-judge indipendente** (verificato: 4 domande su 2 sessioni → 4 righe di
log e 4 righe in `agno_eval_runs`). Il giudice valuta ogni risposta isolatamente,
senza sapere che fa parte di una conversazione più lunga — coerente con il
limite già noto per cui il giudice vede solo input/output testuale del singolo
run (§11).

---

## 8. Cache delle risposte

Motivazione: evitare di rieseguire l'intera pipeline multi-agente (fino a
1-2 minuti per i pitch creativi, che coinvolgono entrambi i worker in parallelo
più il giudice in background) quando una domanda già risposta con successo
viene ripetuta, comune durante sessioni di test/demo.

Implementata in `app/response_cache.py` come cache JSON persistita su disco
(`tmp/response_cache.json`), con TTL di **1 ora** (abbastanza per una sessione
di test/demo, abbastanza corto da non rischiare risposte stantie se nel
frattempo i dati sottostanti cambiano) e — punto critico — **solo le risposte
con `status=COMPLETED` vengono cachate**, in modo che un 429/503 transitorio
non resti "congelato" per le richieste successive.

### 8.1 — Due livelli di match

1. **Match esatto**: stesso `team_id` + stesso testo della domanda
   (normalizzato con `.strip()`). Nessuna chiamata di embedding, risposta
   immediata.
2. **Match semantico** (se il match esatto fallisce): la domanda viene
   confrontata con le domande già in cache tramite **similarità coseno** sugli
   embedding Gemini (`task_type="SEMANTIC_SIMILARITY"`, 768 dimensioni —
   ridotte rispetto alle 1536 del knowledge base principale perché qui serve
   solo confrontare domande tra loro, non fare retrieval documentale), con
   soglia **0.90**.

### 8.2 — Il gate sulle entità (perché la sola similarità non basta)

Un rischio specifico di questo dominio: due domande con struttura quasi
identica ma su persone diverse — *"Quali film ha diretto Wes Anderson?"* vs
*"Quali film ha diretto Tim Burton?"* — risultano **troppo simili** per un
embedding di frase puro, perché la maggior parte dei token (template della
domanda) è identica e solo il nome proprio cambia. Un cache hit sbagliato in
questo scenario significherebbe restituire dati su un regista quando ne è stato
chiesto un altro — un errore silenzioso particolarmente grave in un sistema il
cui intero design è orientato a non restituire mai dati non verificati.

Per eliminare questo rischio, il match semantico scatta **solo se l'insieme dei
nomi propri estratti dalle due domande coincide esattamente** (`_extract_entities`,
euristica su sequenze di parole capitalizzate, con una stopword list per le
parole interrogative italiane che iniziano la frase — "Quali", "Chi", "Trova",
ecc. — per non trattarle come falsi nomi propri). Solo tra domande che citano
già le stesse entità, la similarità di embedding decide se sono la **stessa
domanda riformulata** o due domande diverse sulle stesse persone (es. "film
diretti da" vs "biografia di" sullo stesso regista, correttamente NON messe in
cache reciproca nonostante condividano l'entità).

### 8.3 — Verifica empirica

| Domanda | Esito atteso | Esito osservato |
|---|---|---|
| "Quali film ha diretto Wes Anderson?" | Pipeline completa (prima volta) | 6.8s, risposta corretta |
| "Elenca i film diretti da Wes Anderson." (parafrasi, stessa entità) | Cache hit | **0.375s**, `cached: true`, stessa risposta |
| "Quali film ha diretto Tim Burton?" (entità diversa, template quasi identico) | Nessun cache hit | 26.5s, pipeline rieseguita, risposta corretta su Tim Burton |

---

## 9. Parallelizzazione dei worker

Per richieste che richiedono entrambi gli agenti (es. un pitch completo), il
coordinatore era istruito con un processo **numerato e sequenziale**, che
portava — nonostante Agno supporti nativamente l'esecuzione parallela di più
tool call nello stesso turno tramite `asyncio.gather` (`arun_function_calls`)
— a chiamate ai worker una dopo l'altra invece che in parallelo.

**Causa radice**: non tecnica/architetturale, ma di **prompt**: le istruzioni
del coordinatore descrivevano un processo sequenziale, e il modello lo seguiva
alla lettera. La correzione è stata quindi solo testuale, in `app/agent.py`:

```
"2. Se servono ENTRAMBI gli agenti, chiama 'delegate_task_to_member' due volte
nella STESSA risposta (due tool call nello stesso turno, non uno alla volta in
turni separati): le due richieste sono indipendenti tra loro [...] quindi
vengono eseguite in parallelo e la risposta arriva più velocemente."
```

**Verifica**: confrontati i timestamp `created_at` delle chiamate ai worker
nei raw SSE events (timestamp coincidenti = esecuzione parallela confermata) e
misurata la riduzione della durata end-to-end su richieste reali di pitch
completo: **108.6s (mediana, prima) → 91.3s, 65.2s, 51.2s (dopo, run
successive)**.

---

## 10. Osservabilità: logging e dashboard

### 10.1 — `QueryLoggerMiddleware` (`app/query_logger.py`)

Middleware Starlette che intercetta ogni `POST /teams/{team_id}/runs` (sia in
streaming SSE sia in risposta singola JSON) e registra una riga in
`tmp/query_log.csv` per ogni run, **prima** di verificare se servirla dalla
cache (§8).

Campi tracciati: timestamp, team/session/run id, domanda, agenti delegati,
risposta, status, modello e provider effettivi (per rilevare se è scattato il
fallback Groq), se la risposta viene dalla cache, token input/output/totali,
costo stimato, durata end-to-end.

**Scelte implementative rilevanti**:

- **Parsing manuale del multipart/form-data** e degli eventi SSE via regex
  (`_parse_multipart_text_fields`, `_FINAL_EVENT_RE`): necessario perché il
  middleware deve leggere il body della richiesta/risposta senza consumarlo in
  modo che l'handler successivo (o il client) possa comunque riceverlo —
  implementato con un **tee** del `body_iterator` di streaming (`tee_and_log`),
  che inoltra ogni chunk al client mentre lo accumula per il logging finale.
- **Calcolo di token e costo che somma coordinator + tutti i worker delegati**
  (`_compute_tokens_and_cost`): il campo `metrics` di primo livello della
  risposta copre **solo** il coordinator — sommare anche
  `member_responses[].metrics` è stato un fix critico dopo aver scoperto un
  caso reale di **sottostima di 30x** del consumo effettivo di token guardando
  solo il livello superiore.
- **Rilevamento del fallback** per confronto tra il provider osservato
  nell'ultima chiamata modello (`model_provider`) e la costante
  `_PRIMARY_PROVIDER = "Google"`: se diverso, il fallback Groq è scattato.
- **Costo stimato solo indicativo**: tariffe a pagamento (Gemini/Groq), usate
  per stimare "quanto costerebbe se non si fosse sul free tier", non un costo
  reale sostenuto.

### 10.2 — Dashboard Streamlit (`dashboard/`)

- **`data.py`**: carica e unisce due fonti — `tmp/query_log.csv` (una riga per
  run) e la tabella `agno_eval_runs` di `tmp/cinema_traces.db` (punteggi del
  giudice). Il join **non** può usare `run_id` come chiave: l'`AgentAsJudgeEval`
  genera un proprio `run_id` interno indipendente da quello della run del team
  (bug scoperto empiricamente) — la soluzione adottata abbina per **testo esatto
  della domanda + timestamp di valutazione più vicino** (successivo) al
  timestamp della run.
- **`app.py`**: tre istogrammi tripli (uno per "solo Graph Agent", uno per
  "solo Semantic Agent", uno per "entrambi") rispettivamente per:
  - **distribuzione dei token per domanda** (bin size arrotondato a
    1/2/5×10ⁿ per leggibilità, stesso numero di fasce sincronizzato tra i tre
    grafici della stessa riga per restare confrontabili pur avendo range di
    valori molto diversi);
  - **distribuzione dei tempi di risposta** (escluse le risposte servite dalla
    cache — tempi vicini a zero che distorcerebbero la distribuzione — e le
    domande oltre 180s, outlier dovuti a rate limit/blackout Gemini, non
    rappresentativi del comportamento "normale" del sistema);
  - **distribuzione dei punteggi del giudice** (asse x fisso 1-10 anche con
    pochi dati, linea verticale della media).
  - Ogni barra ha un tooltip con l'elenco delle domande in quella fascia.
- **Filtri**: combinazione **esatta** di agenti coinvolti (non "almeno uno") —
  deselezionare sia Graph sia Semantic Agent mostra **tutti** i log invece di
  filtrare a un insieme vuoto, comportamento esplicitamente richiesto per non
  rendere i due checkbox una trappola UX; più filtro per modello usato.
- **Tabella con drill-down**: selezione di riga singola (`st.dataframe` con
  `on_select="rerun"`) per vedere il dettaglio completo — risposta integrale,
  agenti coinvolti, modello/provider (con badge se il fallback è scattato),
  token/costo/tempo, motivazione testuale del giudice.
- **Palette colori validata** per accessibilità CVD/contrasto (skill
  `dataviz`): blu/arancione/acqua per le tre combinazioni di agenti, distinguibili
  in coppie e in tripla.
- **Timezone**: timestamp convertiti a `Europe/Rome` solo nella colonna
  visualizzata della tabella (i dati restano in UTC internamente).

### 10.3 — Osservabilità di versione_2

L'architettura a router (§13) ha uno stack di osservabilità **separato e
indipendente**, non una variante di quello sopra: log su un db SQLite dedicato
(non un CSV), una dashboard propria raggruppata per percorso invece che per
combinazione di agenti, e un'ulteriore dashboard di **confronto** tra le due
architetture. Dettagli in §13 e §14.

---

## 11. Valutazione (LLM-as-a-Judge)

Ogni run del coordinatore è sottoposta, in modo trasparente per l'utente, a
valutazione automatica tramite `AgentAsJudgeEval` (Agno), integrata come
**post-hook** (`quality_eval` in `app/agent.py`), eseguita **in background**
(`run_in_background=True`) per non introdurre latenza nella risposta.

**Criteri di giudizio**:
1. Lingua italiana e pertinenza alla domanda.
2. **Grounding fattuale**: nomi/titoli/numeri devono essere coerenti con dati
   realmente estraibili da Neo4j/ChromaDB, nessuna invenzione palese.
3. Assenza di sezioni meta-processuali ("Come abbiamo ottenuto questi dati").
4. Nei pitch, il cast proposto deve essere giustificato da collaborazioni reali,
   non scelto arbitrariamente (la componente narrativa/creativa non viene
   penalizzata).

Punteggio numerico 1-10, soglia di accettazione **7**, persistito in
`agno_eval_runs` (`tmp/cinema_traces.db`).

**Copertura su versione_2 (§13)**: il `post_hook` automatico esiste solo su
`Team`, non sui percorsi diretti del router (che chiamano `graph_agent`/
`semantic_agent` singolarmente, bypassando il coordinatore apposta per
risparmiarne il costo). Per non perdere la copertura di qualità, quei percorsi
richiamano `quality_eval` **esplicitamente**, in background
(`asyncio.create_task`, fire-and-forget, stessa logica del post-hook nativo) —
un costo extra di chiamate LLM accettato consapevolmente. Sul percorso
`ambiguous` (fallback al Team) la chiamata esplicita viene invece
deliberatamente **omessa**, perché scatterebbe comunque il post-hook nativo di
`cinema_team` e la duplicherebbe.

**Bug di libreria documentato**: il fallback Groq del giudice (stesso
meccanismo del §5) è stato collegato ma **non funziona**: `AgentAsJudgeEval`
richiede output strutturato (`output_schema`, una classe Pydantic), e il
client Groq di Agno crash con `Object of type ModelMetaclass is not JSON
serializable` quando prova a costruire il `response_format` — un bug reale
della libreria, non del codice del progetto (lo stesso modello Groq funziona
correttamente per gli altri agenti, che producono testo libero, non output
strutturato). Non è stato implementato un workaround (parsing di un output
testuale) per scelta esplicita: il giudice resta quindi **non disponibile**
quando Gemini è sia in quota sia in errore contemporaneamente.

**Limite strutturale, documentato con casi concreti**: il giudice vede solo
**input/output testuale** del run, non il trace delle chiamate ai tool — non
può distinguere un dato realmente recuperato da un'affermazione plausibile ma
allucinata. Due casi reali hanno confermato questo limite durante la
validazione manuale (dettagliati in §16): un'allucinazione sulla trama di
*Snatch* (Semantic Agent, innescata da un fallimento del tool) e la fabbricazione
di un attore/film inesistenti nel grafo (Graph Agent, *Lady Bird*/Timothée
Chalamet) — **nessuno dei due intercettato dal giudice automatico**.

---

## 12. Test set

Set di **43 domande di test** (`testing/domande_di_test.csv` +
`testing/domande_di_test.md`), categorizzate per tipo di capacità testata e
peso computazionale, con ground truth verificata direttamente su Neo4j/ChromaDB
dove disponibile.

| Categoria | N. domande | Peso tipico |
|---|---|---|
| Attore → film | 6 | leggera |
| Pitch completo | 6 | pesante |
| Regista → film | 5 | leggera |
| Collaboratori frequenti di un regista | 5 | leggera/media |
| Edge case / guardrail | 5 | leggera |
| Collaborazione tra attori | 4 | leggera |
| Ricerca semantica su trame | 4 | media |
| Conteggio film per genere | 3 | leggera |
| Ricerca biografie | 3 | media |
| Network genere-regista | 2 | media |

Cinque domande sono dedicate esplicitamente ai **guardrail** anziché alla
qualità narrativa: due su persone assenti dal dataset (TMDB 5000 si ferma al
2016-2017 — Zendaya, Timothée Chalamet), un tentativo di prompt injection, un
tentativo di query Cypher distruttiva, un caso di variante ortografica/accenti
nel nome di una persona.

Eseguibile in automatico con `testing/esegui_test.py` (N domande, filtrabili
per categoria/peso) e `testing/esegui_per_categoria.py` (K domande per ogni
categoria) — entrambi passano dal `QueryLoggerMiddleware` reale, quindi ogni
esecuzione produce anche i log/valutazioni completi in `tmp/query_log.csv`.

**Metodologia**: ogni risposta è stata verificata per **confronto diretto** con
i dati sorgente (query Cypher dirette o ispezione dei documenti ChromaDB), mai
per plausibilità — questo ha permesso di scoprire allucinazioni che un controllo
qualitativo o il solo giudice automatico non avrebbero intercettato.

### 12.1 — Altri due set di dati, per scopi diversi

Il progetto usa **tre** set di domande distinti, ciascuno con uno scopo
specifico — vanno tenuti separati per non invalidare le rispettive verifiche:

| Set | File | Scopo |
|---|---|---|
| Test set originale (sopra) | `testing/domande_di_test.csv` | Validare la qualità/correttezza dell'architettura principale |
| Esempi di classificazione | `architetture_alternative/versione_2/esempi_classificazione.csv` (112 domande) | Esemplari etichettati con cui il router di versione_2 classifica per similarità (§13) — **non** un test set di qualità, sono gli esempi noti contro cui si confronta ogni domanda nuova |
| Held-out per il confronto | `testing/domande_confronto_holdout.csv` (12 domande) | Confrontare v1/v2 su domande mai viste dal router — verificato programmaticamente **zero overlap** con gli esempi di classificazione, altrimenti il confronto sarebbe stato viziato (§14) |

---

## 13. Architettura alternativa: router deterministico (versione_2)

### 13.1 — Motivazione

Nell'architettura principale (§3), **ogni** domanda passa dal coordinatore
(`cinema_team`, `mode=coordinate`), che è esso stesso una chiamata LLM: anche
una domanda fattuale semplice ("Chi ha diretto Interstellar?") paga il costo
di orchestrazione del coordinatore oltre a quello del worker che risponde
davvero. `versione_2` testa un'ipotesi alternativa: **decidere il percorso
prima** di qualunque chiamata LLM, con un meccanismo deterministico e a costo
quasi nullo, così le domande semplici arrivano dritte al worker giusto, e solo
i casi davvero ambigui pagano il costo pieno del coordinatore.

Vive in `architetture_alternative/versione_2/`, **riusa direttamente** gli
stessi agenti/tool/guardrail di `app/agent.py` (stesso `graph_agent`,
`semantic_agent`, `cinema_team`, `quality_eval`) tramite `sys.path.append` —
non duplica la logica di dominio, solo la strategia di instradamento.

### 13.2 — I quattro percorsi

```python
class Percorso(str, Enum):
    GRAPH = "graph"
    SEMANTIC = "semantic"
    PITCH = "pitch"
    AMBIGUOUS = "ambiguous"
```

- **`GRAPH`/`SEMANTIC`**: chiamano direttamente `graph_agent`/`semantic_agent`,
  **bypassando il coordinatore**.
- **`PITCH`**: workflow deterministico dedicato (`pitch_workflow.py`) — esegue
  Graph e Semantic Agent **in parallelo** (`asyncio.gather`, nessuna istruzione
  testuale al coordinatore necessaria per ottenere il parallelismo, a
  differenza del §9) poi un agente di sintesi senza tool fonde i due risultati.
- **`AMBIGUOUS`**: fallback sull'architettura originale — `cinema_team.arun()`
  identico a `app/main.py`, con lo stesso post-hook del giudice nativo.

### 13.3 — Evoluzione del classificatore: da keyword a similarità

La prima versione instradava per **parole chiave fisse** (regex/pattern
matching). Scartata perché "ragiona solo per parole fisse": una domanda
semanticamente identica ma formulata diversamente da qualunque pattern noto
finiva mal classificata o in `AMBIGUOUS` senza necessità. Sostituita con un
classificatore basato su **similarità coseno su embedding**:

1. 112 domande etichettate a mano (`esempi_classificazione.csv`), con
   embedding **precalcolati offline** da un notebook dedicato
   (`ingestion/embedding_esempi_classificazione.ipynb`, stesso pattern di
   checkpointing/retry di `ingestion.ipynb`) — `router.py` non calcola mai
   embedding per gli esempi, solo per la domanda in arrivo, così un riavvio
   del server non richiede fino a 112 chiamate API per rimettersi in pari.
2. Per ogni domanda nuova: si calcola la similarità con tutti i 112 esempi, si
   prendono i **top-5** più simili (`_TOP_K = 5`).
3. **Tre condizioni**, tutte necessarie, altrimenti si ricade su `AMBIGUOUS`:
   - il più simile in assoluto supera la soglia di confidenza (`0.60`);
   - tra i 5 vicini, almeno 3 concordano sulla stessa categoria (maggioranza
     stretta, `_MIN_VOTI_MAGGIORANZA = 3`);
   - la categoria del vicino più simile in assoluto **coincide** con quella
     vincitrice della maggioranza.

**Nota terminologica esplicita**: il meccanismo (retrieval dei vicini più
simili + voto di maggioranza) è concettualmente uno schema k-NN, ma
**non c'è alcun training** — nessun modello viene addestrato, i "vicini" sono
gli stessi 112 esempi etichettati a mano, confrontati ogni volta da zero per
similarità coseno. Una alternativa con classificatore addestrato (es.
XGBoost) è stata scartata per dati insufficienti a un training affidabile.

**Perché la terza condizione**: senza di essa, un caso concreto veniva
instradato male. Esempio: *"Trovami qualcosa di simile a Il Padrino ma con un
tocco più moderno, magari con un regista che sappia gestire bene questo tipo
di atmosfera"* — genuinamente ambigua (mescola ricerca per atmosfera e
richiesta di un regista specifico). Il vicino più simile in assoluto era di
categoria `semantic`, ma 2 dei 3 vicini più prossimi erano `pitch` — la sola
maggioranza avrebbe fatto vincere `pitch` nonostante il segnale più forte (il
primo vicino) indicasse `semantic`. Aggiungendo il vincolo "il vincitore deve
coincidere col primo classificato", il caso ricade correttamente su
`AMBIGUOUS`.

**Calibrazione e validazione**: `k`, soglia e regola di maggioranza sono stati
scelti tramite **cross-validation leave-one-out** sui 112 esempi (ogni esempio
"finto nuovo" riclassificato usando gli altri 111 come pool), senza alcuna
chiamata API live — risultato finale: 96 corretti, 1 sbagliato, 15 lasciati
onestamente `AMBIGUOUS` su 112. Verificato anche con domande mai viste dal
classificatore (parafrasi ed entità nuove, non nell'insieme di esempi) e con i
tre esempi discussi nella presentazione (DiCaprio → graph, pitch su Nolan →
pitch, "Il Padrino" → ambiguous), tutti classificati correttamente dal vivo.

### 13.4 — Guardrail: perché resta a keyword fisse

Il controllo anti-prompt-injection (`controlla_prompt_injection`, stessa
lista di pattern del §6.1) **non** è stato convertito a similarità come il
resto del router — scelta deliberata: un controllo di sicurezza deve restare
**prevedibile**. Con una soglia di confidenza, un tentativo di attacco scritto
in modo che "assomigli poco" ai pattern noti (pur contenendo la stessa
istruzione malevola) potrebbe scivolare sotto soglia. Le keyword fisse non
hanno questa via di fuga per le formulazioni che riconoscono.

### 13.5 — Esposizione su AgentOS Web (`agentos_main.py`)

Oltre all'endpoint `/query` "misurato" (porta 8001, usato da test e
confronto), la stessa logica di routing è esposta anche sul pannello AgentOS
Web (porta 8002, come l'architettura principale), tramite un `Workflow` Agno
con uno step `Router` — il `selector` richiama direttamente
`router.classifica()`, la stessa funzione usata da `main.py` (nessuna
duplicazione della logica di instradamento). Ogni percorso registra comunque
una riga nel log strutturato e richiama il giudice in background, con la
stessa parità di comportamento descritta in §10.3/§11.

### 13.6 — Osservabilità dedicata

Il logging di versione_2 (`query_logger.py`) scrive nella tabella `query_log`
di un **db SQLite dedicato**
(`architetture_alternative/versione_2/tmp/cinema_traces_v2.db`), non un CSV
— scelta fatta per tenere separato lo storico di versione_2 da quello
dell'architettura principale, e per condividere lo stesso file db usato dalle
sessioni AgentOS di `agentos_main.py` (tabelle distinte per nome, stesso
file). Un'eccezione: il percorso `AMBIGUOUS` riusa l'oggetto `cinema_team` di
`app/agent.py`, che ha il proprio db (quello di versione_1) già cablato alla
costruzione — la **sessione AgentOS** di quel solo percorso finisce quindi nel
db originale, mentre la riga di `query_log` (scrittura SQL esplicita,
indipendente) va sempre nel db di versione_2 per tutti i percorsi.

Dashboard dedicata (`architetture_alternative/versione_2/dashboard/`),
raggruppata per **percorso** invece che per combinazione di agenti — con un
KPI esplicito su quanto spesso il router evita il percorso costoso
(`ambiguous`) rispetto ai percorsi diretti.

---

## 14. Confronto tra le due architetture

### 14.1 — Metodologia (`testing/FRAMEWORK_CONFRONTO.md`)

Sei principi, applicati rigorosamente per un confronto **equo**:

1. **Solo domande held-out**: mai domande già presenti negli esempi di
   classificazione del router (§12.1), altrimenti il confronto sarebbe viziato
   a favore di versione_2.
2. **Esecuzione interleaved**: stessa domanda su v1 e v2 **alternata**, non in
   blocco, per non far coincidere l'ordine con condizioni esterne diverse
   (quota, carico) tra le due architetture.
3. **Attesa obbligatoria della valutazione del giudice** prima di considerare
   completa una singola esecuzione (con una checklist pre-test che include il
   controllo della quota giornaliera Gemini e il bug noto del §11 sul
   fallback Groq del giudice).
4. **Mediana preferita alla media**: meno sensibile a outlier reali (es. un
   blackout di servizio di ~3h23m osservato durante lo sviluppo, §4).
5. **Fallback tracciato come variabile a parte**: se scatta per una sola
   architettura per puro timing di quota, quella domanda va segnalata o
   esclusa dal confronto principale.
6. **Un CSV dedicato per ogni esecuzione di confronto**, non solo i log
   condivisi — perché la dashboard di confronto, sulle "domande comuni",
   prende la run più recente per ciascuna domanda in ciascun log: se la stessa
   domanda fosse già stata posta in una sessione precedente, un confronto
   letto da lì rischierebbe di mescolare esecuzioni di sessioni diverse,
   vanificando l'alternanza del punto 2.

### 14.2 — Script ed esecuzione (`testing/esegui_confronto_interleaved.py`)

Automatizza i sei principi sopra: alterna le chiamate a v1/v2, attende (con
polling su `agno_eval_runs`) la valutazione del giudice per ciascuna con un
timeout esplicito, legge la riga appena scritta nel log di ciascuna
architettura (CSV per v1, tabella `query_log` del db dedicato per v2, §13.6),
e scrive un CSV di risultati **autosufficiente** per quella sola esecuzione,
con i delta già calcolati (token, durata, punteggio).

### 14.3 — Analisi storica (`architetture_alternative/confronto/analisi_storica.ipynb`)

Oltre al confronto controllato sopra, un notebook analizza **tutte** le
domande già eseguite in passato su entrambe le architetture (28+ domande in
comune al momento della stesura), per uno sguardo di insieme su un campione
più ampio di quello di un singolo run interleaved.

**Bug di data quality scoperto e corretto durante l'analisi**: la selezione
"esecuzione più recente per ciascuna domanda" su v1 poteva selezionare una
riga servita dalla **cache delle risposte** (§8, durata ≈ 0s), facendo
apparire v1 falsamente istantanea su quella domanda. Fix: escludere le righe
`risposta_da_cache = "Sì"` prima di scegliere la "più recente" — la
correzione ha cambiato il delta medio di durata da **-8.6%** a **+12.0%** a
favore di v1 (un risultato meno favorevole a v2, riportato comunque
onestamente perché era quello vero).

**Divergenza mediana/media**: sulle stesse domande, la mediana della durata
mostra v2 **~33% più veloce** di v1, mentre la media (dopo il fix sopra) è
**+12.0%** più lenta — divergenza reale dovuta a poche domande con tempi
molto alti (es. pitch, che richiedono più chiamate LLM in sequenza), non un
errore di calcolo: da qui il punto 4 della metodologia (§14.1), preferire la
mediana per non farsi distorcere da questi outlier reali.

### 14.4 — Dashboard di confronto (`architetture_alternative/confronto/app.py`)

KPI aggregati, tabella delle domande comuni con delta per singola domanda
(token/durata/**numero di chiamate LLM**, non solo tempo/token — il numero di
chiamate cattura direttamente il risparmio strutturale del router: v1 paga
sempre `1 + N worker delegati` per il coordinatore, v2 paga `N worker` sui
percorsi diretti, `1 + N` solo su `ambiguous`), grafici di composizione e
distribuzione sovrapposti, confronto dei punteggi del giudice.

---

## 15. Organizzazione del repository

### Struttura delle directory

```
app/                            # Architettura principale (produzione): Team coordinate
├── agent.py                    #   team, agenti worker, tool, guardrail, memoria, giudice
├── main.py                     #   FastAPI/AgentOS (porta 8000)
├── query_logger.py             #   middleware di logging + cache
└── response_cache.py           #   cache semantica delle risposte

dashboard/                      # Dashboard Streamlit dell'architettura principale
testing/                        # Suite di test e framework di confronto tra architetture
├── domande_di_test.csv/.md       #   43 domande categorizzate con ground truth (§12)
├── domande_confronto_holdout.csv #   12 domande held-out per il confronto v1/v2 (§14)
├── esegui_confronto_interleaved.py
└── FRAMEWORK_CONFRONTO.md        #   metodologia del confronto (§14.1)

architetture_alternative/       # Esperimenti architetturali per il confronto
├── versione_1/                 #   snapshot statico di riferimento (non eseguibile standalone)
├── versione_2/                 #   router deterministico + workflow parallelo (§13)
│   ├── router.py                 #   classificazione per similarità su embedding
│   ├── pitch_workflow.py          #   workflow parallelo per i pitch
│   ├── esempi_classificazione.csv #   112 esempi etichettati per il router (§13.3)
│   ├── main.py, query_logger.py   #   endpoint /query "misurato" (porta 8001) + log su db dedicato
│   ├── agentos_main.py            #   stesso router esposto su AgentOS Web (porta 8002, §13.5)
│   ├── dashboard/                 #   dashboard dedicata, raggruppata per percorso
│   └── testing/                   #   stessa suite di test, adattata all'endpoint /query
└── confronto/                  #   dashboard + notebook di analisi storica tra le due architetture (§14)

ingestion/                      # Notebook di popolamento Neo4j + ChromaDB
├── ingestion.ipynb               #   pipeline principale (§2.3)
└── embedding_esempi_classificazione.ipynb  # embedding offline dei 112 esempi del router (§13.3)
neo4j_export/                   # Export del grafo in CSV/JSON
data/                           # Dataset sorgente (TMDB CSV, biografie)
Presentazione/                  # Materiale per la presentazione d'esame
tmp/                             # Runtime: log, cache, db (in gran parte gitignored)
```

I quattro file dell'architettura principale (`agent.py`, `main.py`,
`query_logger.py`, `response_cache.py`) vivono insieme in `app/` perché si
importano a vicenda nella stessa directory; i percorsi verso `tmp/` sono
ancorati esplicitamente alla radice del progetto tramite `Path(__file__)`
(non alla working directory di invocazione), cosicché l'app funzioni
identicamente sia lanciata come `uv run python app/main.py` dalla radice sia,
in linea di principio, da una posizione diversa. Le architetture sperimentali
sotto `architetture_alternative/` riusano direttamente gli agenti e i tool
definiti in `app/agent.py` (stessa configurazione, stesso fallback, stesso
knowledge base) tramite un `sys.path.append` mirato a quella directory — non
duplicano la logica del dominio, solo la strategia di orchestrazione.

- **Branching**: `master` (stabile) ← `develop` ← `sviluppo_N` (feature branch
  attivo, es. `sviluppo_2`), con workflow esplicito feature → develop → master
  a doppio merge (fast-forward) a ogni rilascio, poi ritorno sul branch di
  sviluppo per continuare. Pattern pensato per essere esteso con nuovi branch
  `sviluppo_N` per sviluppi futuri.
- **`.gitignore`**, con enfasi sulla sicurezza: `.env` escluso fin dal primo
  commit (verificato con grep sui diff staged per pattern di chiavi API prima
  di ogni commit: `AIza...`, `gsk_...`, `sk-...`), file di lock di PowerPoint
  (`~$*`), dati runtime pesanti e rigenerabili (`tmp/chromadb/`,
  `tmp/cinema_traces.db`, il suo equivalente per versione_2
  `architetture_alternative/versione_2/tmp/cinema_traces_v2.db`,
  `tmp/query_log.xlsx`) esclusi perché non adatti a git e ricostruibili
  dall'ingestion/dall'uso del sistema. Il file della presentazione d'esame
  resta deliberatamente fuori dalla cronologia dei commit (aggiornato e
  condiviso separatamente, non tramite il repository).
- **Pulizia del codice**: rimossi durante lo sviluppo `models.py`,
  `recipebot_app.py`, un notebook di ingestion duplicato/obsoleto
  (`da_csv_a_neo4jaa.ipynb`) e una copia orfana del database vettoriale sotto
  `ingestion/tmp/chromadb/` (il DB vettoriale reale vive sotto `tmp/chromadb/`
  alla radice del progetto, con percorso assoluto risolto da
  `_find_project_root()`, vedi §2.3).

---

## 16. Limiti noti del sistema

### Dati
- **Dataset non aggiornato** (TMDB 5000, fermo al 2016-2017 circa): persone e
  film recenti sono assenti — comportamento corretto verificato (risposta
  "nessun risultato", non invenzione), ma è un limite di copertura strutturale.
- **Nessuna capacità di scrittura**: sistema solo in lettura (blocklist Cypher);
  aggiornare il grafo richiede una nuova esecuzione manuale della pipeline di
  ingestion (vedi proposta di Builder Workflow, §17) — resta vero anche per
  versione_2 (§13), che instrada in modo diverso ma è comunque solo in lettura.
- **Nessun chunking** degli embedding: adeguato alla dimensione attuale dei
  testi, non scalerebbe a documenti molto più lunghi.

### Modello / allucinazioni
Nonostante istruzioni esplicite anti-invenzione, la validazione empirica ha
individuato **casi concreti e riproducibili**:
- **Semantic Agent**: a seguito di un fallimento del tool di ricerca, ha
  generato una trama per *Snatch* non presente nella knowledge base.
- **Graph Agent**: ha fabbricato sia la presenza di un attore (Timothée
  Chalamet) sia il titolo di un film (*Lady Bird*), in modo incoerente rispetto
  a una domanda di controllo quasi identica che ha invece prodotto la risposta
  corretta.
- **Pitch su Tim Burton**: cast generato includeva un film (*Sleepy Hollow*)
  non presente nei dati restituiti dal Graph Agent.

### Valutazione
- Il giudice automatico non vede il trace dei tool — non intercetta nessuno dei
  casi sopra.
- Validazione empirica manuale su un set fisso di 43 domande, non ancora una
  suite di regressione automatica con metriche aggregate (RAGAS, §17).

### Architettura e conversazione
- **Memoria conversazionale** limitata a `num_history_runs=5`: in conversazioni
  molto lunghe, il contesto iniziale viene perso silenziosamente.
- **Parallelizzazione dei worker** basata su istruzioni testuali al
  coordinatore, non su un vincolo architetturale imposto dal framework —
  comportamento atteso ma non garantito in modo deterministico.
- **Guardrail basati su pattern/keyword fissi** (pre-hook anti-injection,
  blocklist Cypher): aggirabili in linea di principio con parafrasi,
  traduzione o offuscamento — prima linea di difesa, non garanzia assoluta.
- **Blocklist Cypher applicativa, non a livello di permessi DB**: non esiste un
  utente Neo4j dedicato in sola lettura.
- **Cache delle risposte**: risolto il limite del match esatto (§8) con un
  match semantico protetto da un gate sulle entità citate — resta comunque
  vincolata al riconoscimento di nomi propri capitalizzati (euristica, non
  un NER completo).

### Operativi
- **Dipendenza dal free tier di Gemini**: quote instabili (429/503, incluso un
  disservizio reale di ~3h23m), mitigata ma non eliminata da retry + fallback +
  cache + timeout.
- **Modello di fallback scelto tramite confronto empirico** tra pochi
  candidati disponibili sul free tier, non una valutazione esaustiva.
- **API senza autenticazione**: accettabile in un contesto d'esame, non per un
  deployment reale.

### Architettura router (versione_2, §13)
- **Classificatore per similarità, non un NER/parser completo**: si basa su
  112 esempi etichettati a mano — copre bene i pattern di domanda osservati
  durante lo sviluppo, ma un dominio molto più ampio richiederebbe un set di
  esempi più grande per restare accurato.
- **Nessuna cache delle risposte**: a differenza dell'architettura principale
  (§8), versione_2 non implementa una cache — ogni domanda, anche ripetuta,
  riesegue la pipeline completa.
- **Il fallback `AMBIGUOUS` riusa `cinema_team` così com'è**: eredita quindi
  anche i suoi limiti (memoria conversazionale, parallelizzazione basata su
  prompt) descritti sopra.
- **Esposizione AgentOS (`agentos_main.py`) senza cache/autenticazione**,
  stesso livello di maturità del resto del progetto — pensata per
  l'esplorazione interattiva, non per un uso in produzione.

---

## 17. Sviluppi futuri

### Builder Workflow per la scrittura (non implementato)

Il sistema resta **solo in lettura** su entrambe le architetture (blocklist
Cypher, §6.2). Per superare questo limite, era stata disegnata — insieme al
Router Agent poi effettivamente costruito come versione_2 (§13) — l'idea di
un **Builder Workflow** dedicato all'ingestion conversazionale (es.
*"Aggiungi al grafo il film Inception e tutte le informazioni sugli attori
principali"*), che validerebbe i dati, popolerebbe Neo4j e genererebbe gli
embedding ChromaDB corrispondenti.

**Deliberatamente non implementato**: a differenza della parte di
instradamento (costruita, testata e confrontata empiricamente in versione_2),
la parte di scrittura non è mai stata realizzata. Un comando di scrittura
implicherebbe chiamate LLM aggiuntive (estrazione strutturata, validazione,
generazione embedding) con un consumo di token e una pressione sui limiti di
quota free-tier — già un vincolo stretto per tutto il resto dello sviluppo —
non sostenibile nei tempi del progetto. Resta una proposta motivata
architetturalmente ma non implementata.

### RAGAS

Per superare il limite del giudice LLM-as-a-judge (cieco rispetto al trace dei
tool), è stato individuato il framework **RAGAS** (Retrieval-Augmented
Generation Assessment) come estensione naturale della componente di
valutazione: permetterebbe di misurare metriche specifiche dei sistemi RAG —
*faithfulness* (fedeltà della risposta ai documenti/dati recuperati),
*context precision/recall* — confrontando esplicitamente risposta, domanda e
contesto recuperato dai tool, invece di valutare solo il testo finale in
isolamento.
