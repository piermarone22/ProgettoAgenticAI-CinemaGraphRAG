# Framework per il confronto equo tra versione_1 e versione_2

Metodologia per confrontare token, tempi e valutazioni del giudice tra
l'architettura originale (`app/`, sempre attraverso il Creative Agent) e il
router deterministico (`architetture_alternative/versione_2/`), senza che il
confronto sia falsato da fattori esterni alle due architetture stesse.

Nasce da problemi concreti riscontrati durante lo sviluppo (non ipotetici):
un classificatore che si autovaluta sui propri esempi, un giudice che fallisce
silenziosamente per quota esaurita facendo sembrare un'architettura "senza
valutazioni" quando in realtà è solo sfortuna di tempistica, un confronto
falsato perché le due architetture sono state testate in momenti diversi con
condizioni di rete/quota diverse.

---

## 1. Usa solo domande held-out, mai quelle nell'esemplario del router

Le domande in `architetture_alternative/versione_2/esempi_classificazione.csv`
(112 righe, consolidate da `testing/domande_di_test.csv` più un set di esempi
aggiuntivi dedicato) sono gli esempi etichettati con cui il classificatore di
versione_2 si confronta per similarità coseno (vedi `router.py`). Usarle per il confronto
tra le due architetture **favorirebbe artificialmente versione_2**: il
router riconoscerebbe se stesso quasi alla perfezione, non perché generalizza
bene, ma perché sta letteralmente confrontando la domanda con se stessa.

**Regola pratica**: usa `testing/domande_confronto_holdout.csv` (o un nuovo
file di domande mai incluso nei due esemplari sopra) per qualunque confronto
tra architetture. Il classificatore va validato separatamente, sulle sue
stesse metriche (cross-validation leave-one-out, vedi il docstring in cima a
`router.py`), non mischiando le due valutazioni.

## 2. Esecuzione ALTERNATA (interleaved), mai in blocco

Non testare "prima tutte le domande su versione_1, poi tutte su versione_2".
Le condizioni esterne — soprattutto la quota Gemini, che si esaurisce
progressivamente nel corso della giornata e può azzerarsi del tutto (limite
giornaliero `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, non solo un
rate limit al minuto) — cambiano nel tempo. Se un'architettura viene testata
per intero con quota fresca e l'altra con quota quasi esaurita, il confronto
misura le condizioni esterne, non le architetture.

**Regola pratica**: per ogni domanda, esegui prima la chiamata a versione_1,
poi subito quella a versione_2 (o viceversa, l'importante è alternare), prima
di passare alla domanda successiva. Lo script
`testing/esegui_confronto_interleaved.py` fa esattamente questo.

## 3. Aspetta sempre la valutazione del giudice — obbligatorio

Il giudice (`quality_eval`, `AgentAsJudgeEval`) gira in background
(`run_in_background=True` sul post_hook nativo per l'architettura originale;
`asyncio.create_task` esplicito per i percorsi diretti di versione_2) — **non
blocca la risposta all'utente**, ma questo significa anche che può non essere
ancora finito quando lo script passa alla domanda successiva o quando si
riavvia un server.

**Scoperto concretamente in sessione**: riavviare il processo server subito
dopo aver ricevuto la risposta HTTP, senza aspettare che il task di
valutazione in background finisca, fa perdere quella valutazione per sempre
— non c'è modo di recuperarla. Inoltre, un fallimento del giudice (es. quota
Gemini esaurita) non solleva un errore visibile: `AgentAsJudgeEval.arun()`
logga comunque una riga nel db con `input` e `score` a `None` invece di
sollevare un'eccezione — bisogna controllare esplicitamente che lo score non
sia `None`, non solo che la riga esista.

**Regola pratica**: dopo ogni chiamata completata con successo, interroga
`tmp/cinema_traces.db` (tabella `agno_eval_runs`) finché non compare una
valutazione con `score` valido per quella esatta domanda, prima di procedere.
Non riavviare né killare i processi server finché non sei sicuro che tutte le
valutazioni in sospeso siano arrivate. Lo script gestisce questo in automatico
con `attendi_valutazione()` (timeout configurabile, default 90s a chiamata).

### Checklist pre-test (prima di lanciare qualunque batch di confronto)

- [ ] Verifica che la quota Gemini sia disponibile: lancia una singola domanda
      di prova e controlla i log del server per eventuali errori
      `RESOURCE_EXHAUSTED` con `GenerateRequestsPerDayPerProjectPerModel-FreeTier`.
      Se presente, la quota giornaliera è esaurita: aspetta il reset (di solito
      a mezzanotte Pacific Time) prima di procedere — nessun fix lato codice
      risolve una quota giornaliera esaurita.
- [ ] Verifica che entrambi i server siano avviati: `main.py` (root, porta
      8000) e `architetture_alternative/versione_2/main.py` (porta 8001).
- [ ] Se il fallback Gemini→Groq dovesse scattare durante il test, ricorda il
      limite noto: il giudice ha un `fallback_config` configurato (stesso
      Qwen degli altri agenti), ma la combinazione Groq + output strutturato
      (`output_schema`) va in crash per un bug di libreria in questa versione
      di agno (`Object of type ModelMetaclass is not JSON serializable`) — se
      Gemini fallisce E il fallback del giudice viene tentato, la valutazione
      fallirà comunque. Gli altri agenti (Graph/Semantic/coordinator) non
      hanno questo problema perché non usano output strutturato.

## 4. Guarda la mediana e la tabella per-domanda, non solo la media aggregata

Con un campione di 10-40 domande in comune, un singolo caso estremo può
spostare parecchio una media (es. una domanda di network genere-regista ha
mostrato -84% di token in un confronto, un outlier reale non rappresentativo
dell'andamento tipico). La dashboard di confronto
(`architetture_alternative/confronto/app.py`) calcola già i delta per singola
domanda nella tabella "Confronto diretto" — guarda l'intera distribuzione,
non solo la metrica aggregata in cima alla pagina.

## 5. Traccia il fallback come variabile a parte

Se durante un test il fallback Groq scatta per un'architettura e non per
l'altra (per puro timing di quota, non per una differenza di design), quella
singola domanda ha token/qualità diversi per un motivo esterno alle due
architetture messe a confronto. La colonna `fallback_usato` è già presente in
entrambi i log (`tmp/query_log.csv`, `architetture_alternative/versione_2/tmp/query_log_v2.csv`)
— quando confronti, segnala o escludi le righe con `fallback_usato = Sì` dal
confronto principale, e riportale a parte se rilevanti.

---

## 6. Un CSV dedicato per ogni esecuzione, non solo i log condivisi

I log persistenti (`tmp/query_log.csv`, `architetture_alternative/versione_2/tmp/query_log_v2.csv`)
accumulano tutte le domande mai eseguite, in sessioni diverse, con condizioni
diverse. La dashboard di confronto, per le "domande comuni", prende la run
più recente di ciascuna domanda in ciascun log — se la stessa domanda fosse
mai stata posta anche in una sessione precedente, un confronto letto da lì
rischierebbe di mescolare un'esecuzione di oggi con una di ieri, vanificando
l'alternanza del punto 2.

**Regola pratica**: `esegui_confronto_interleaved.py` scrive un CSV dedicato
per ogni esecuzione (`testing/risultati_confronto/confronto_<timestamp>.csv`),
leggendo — subito dopo ogni chiamata — esattamente la riga appena scritta nel
log dell'architettura interessata (per testo della domanda + timestamp
successivo all'invio), non "l'ultima in assoluto". Il risultato è un CSV
autosufficiente con le colonne `v1_*`/`v2_*` affiancate (token, costo, durata,
punteggio del giudice) e i delta già calcolati, limitato esattamente alle
domande di quella singola esecuzione — pronto per un confronto diretto senza
passare dalla dashboard, se preferisci un foglio di calcolo.

## Come eseguire il confronto

```bash
# 1. Verifica quota Gemini (checklist sopra), poi avvia entrambi i server:
uv run python app/main.py                                    # porta 8000
uv run python architetture_alternative/versione_2/main.py    # porta 8001

# 2. Esegui il confronto interleaved (in un altro terminale):
uv run python testing/esegui_confronto_interleaved.py

# Varianti utili:
uv run python testing/esegui_confronto_interleaved.py --n 5                    # solo le prime 5 domande
uv run python testing/esegui_confronto_interleaved.py --timeout-giudice 120    # piu' pazienza per il giudice
uv run python testing/esegui_confronto_interleaved.py --file testing/domande_confronto_holdout.csv --pausa 8

# 3. Analizza i risultati:
uv run streamlit run architetture_alternative/confronto/app.py
```

Ogni domanda del file `testing/domande_confronto_holdout.csv` è pensata per
usare attori/registi/film con dati vettoriali già presenti in ChromaDB
(verificato al momento della creazione del set) e presenti in Neo4j, così da
poter essere eseguita end-to-end senza il rischio di "nessun risultato
trovato" per assenza di dati — l'unica eccezione voluta è la domanda
`edge_case`, che testa deliberatamente il comportamento anti-invenzione su
una persona inesistente.
