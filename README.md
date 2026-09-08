# Cinema GraphRAG

Sistema di intelligenza artificiale cinematografica basato su un'architettura GraphRAG ibrida: un knowledge graph Neo4j (film/attori/registi/generi) combinato con un database vettoriale ChromaDB (trame/biografie), orchestrati da un team multi-agente (framework [Agno](https://github.com/agno-agi/agno)) che risponde a domande fattuali e a richieste creative (pitch, casting motivato da dati reali).

Il progetto include **due architetture alternative** (per confrontare un approccio "sempre attraverso il coordinator" contro un router deterministico) e gli strumenti per confrontarle. Per l'approfondimento completo di ogni scelta progettuale vedi [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md).

## Struttura del progetto

```
app/                              Architettura principale: agenti, FastAPI, logging, cache
├── agent.py                        Team, worker (Graph/Semantic Agent), tool, guardrail, giudice
├── main.py                         Server FastAPI/AgentOS (porta 8000)
├── query_logger.py                 Middleware di logging + integrazione cache
└── response_cache.py               Cache delle risposte (match esatto + semantico)

dashboard/                        Dashboard Streamlit dell'architettura principale
testing/                          Suite di test e framework di confronto
├── domande_di_test.csv/.md         43 domande categorizzate con ground truth
├── esegui_test.py                  Esegue N domande contro l'architettura principale
├── domande_confronto_holdout.csv   Domande held-out per confrontare le due architetture
├── esegui_confronto_interleaved.py Esegue le stesse domande su entrambe alternando
└── FRAMEWORK_CONFRONTO.md          Metodologia per un confronto equo

architetture_alternative/         Esperimenti architetturali
├── versione_1/                     Snapshot statico di riferimento (non eseguibile)
├── versione_2/                     Router deterministico (porta 8001)
│   ├── router.py                     Classificazione per similarità su embedding
│   ├── pitch_workflow.py             Workflow parallelo per i pitch
│   ├── esempi_classificazione.csv    Esempi etichettati per il router
│   ├── main.py, query_logger.py      Server FastAPI + logging dedicati
│   ├── agentos_main.py               Stesso router esposto su AgentOS Web (porta 8002)
│   └── dashboard/                    Dashboard dedicata
└── confronto/                      Dashboard + notebook di confronto tra le architetture

ingestion/                        Popolamento Neo4j + ChromaDB (notebook)
neo4j_export/                     Export del grafo in CSV/JSON
data/                             Dataset sorgente (TMDB CSV, biografie)
Presentazione/                    Materiale per la presentazione d'esame
```

## Prerequisiti

- Python 3.13 e [`uv`](https://docs.astral.sh/uv/) installati (`uv sync` per creare l'ambiente dalle dipendenze in `pyproject.toml`).
- Un'istanza Neo4j in esecuzione, raggiungibile dalle credenziali in `.env`.
- Un file `.env` in radice con almeno: `GEMINI_API_KEY`, `GROQ_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- Dati già popolati in Neo4j/ChromaDB — se non ancora fatto, eseguire prima `ingestion/ingestion.ipynb` (popolamento film/biografie) e poi `ingestion/embedding_esempi_classificazione.ipynb` (embedding degli esempi del router).

## Come eseguire

### Architettura principale (root)

```bash
uv run python app/main.py
```

API su `http://127.0.0.1:8000` (endpoint AgentOS, es. `POST /teams/cinema-creative-team/runs`).

### Architettura sperimentale a router (versione_2)

```bash
uv run python architetture_alternative/versione_2/main.py
```

API su `http://127.0.0.1:8001` (`POST /query`, body JSON `{"message": "..."}`). Può girare in parallelo alla principale per confrontarle.

La stessa logica di routing è esplorabile anche dal pannello AgentOS Web (come versione_1), tramite un secondo entry-point che espone il router come `Workflow` con uno step `Router`:

```bash
uv run python architetture_alternative/versione_2/agentos_main.py
```

API su `http://127.0.0.1:8002`. Riusa `router.classifica()`, quindi instrada esattamente come `main.py` — ma è pensato solo per l'esplorazione interattiva: non scrive nella tabella `query_log` (quella letta dalla dashboard e dal confronto tra architetture) e non richiama il giudice (`quality_eval`) in automatico (i `Workflow` di Agno non supportano un `post_hook` come i `Team`). Il percorso "misurato", usato da tutta la suite di test e dal confronto tra architetture, resta l'endpoint `/query` di `main.py`.

Tutto il tracing di versione_2 — sia le sessioni/run di AgentOS sia i log strutturati di `main.py` (tabella `query_log`, letta dalla dashboard di versione_2 e dal confronto tra architetture) — va su un db dedicato, `architetture_alternative/versione_2/tmp/cinema_traces_v2.db`, separato da quello di versione_1 (`tmp/cinema_traces.db`). Un'eccezione: il fallback per i casi ambigui riusa l'oggetto `cinema_team` di `app/agent.py`, che ha il proprio db (e il giudice `quality_eval`) già cablati alla costruzione — quelle run, e tutte le valutazioni del giudice di entrambe le architetture, finiscono quindi comunque nel db originale.

### Dashboard

Ogni dashboard è un'app Streamlit indipendente — porte diverse per tenerle aperte insieme:

```bash
uv run streamlit run dashboard/app.py                                          # architettura principale, porta 8501 (default)
uv run streamlit run architetture_alternative/versione_2/dashboard/app.py --server.port 8502   # versione_2
uv run streamlit run architetture_alternative/confronto/app.py --server.port 8504              # confronto tra le due
```

### Suite di test

```bash
uv run python testing/esegui_test.py 10                          # 10 domande dal test set, sull'architettura principale
uv run python testing/esegui_per_categoria.py --k 2              # 2 domande per ciascuna delle 9 categorie
uv run python testing/esegui_confronto_interleaved.py            # stesse domande su entrambe le architetture, alternate
```

Richiedono che i rispettivi server (`app/main.py` e/o `architetture_alternative/versione_2/main.py`) siano già avviati. Vedi `testing/FRAMEWORK_CONFRONTO.md` per la metodologia del confronto tra architetture.

### Analisi e notebook

```bash
uv run jupyter notebook architetture_alternative/confronto/analisi_storica.ipynb   # confronto sulle domande già eseguite su entrambe
uv run jupyter notebook ingestion/ingestion.ipynb                                  # popolamento Neo4j + ChromaDB
```

## Documentazione

- [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md) — riferimento tecnico completo di ogni scelta progettuale
- [`testing/FRAMEWORK_CONFRONTO.md`](testing/FRAMEWORK_CONFRONTO.md) — metodologia per confrontare le due architetture
- [`Presentazione/architettura.md`](Presentazione/architettura.md) — sintesi per la presentazione d'esame
