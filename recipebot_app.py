from dotenv import load_dotenv

# Carica le variabili da .env (in particolare GOOGLE_API_KEY).
# Importante: deve avvenire PRIMA degli import di Agno.
load_dotenv()

from pydantic import BaseModel, Field
import json
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.learn import (
    LearningMachine,
    LearningMode,
    UserProfileConfig,
    UserMemoryConfig,
)


# region Costanti e setup

RECIPES_DB = [
    {
        "name": "Insalata di ceci e zucchine",
        "ingredients": ["ceci", "zucchine", "limone", "olio"],
        "minutes": 15,
        "tags": ["vegetariano", "vegano", "senza-glutine"],
    },
    {
        "name": "Riso allo yogurt e zucchine",
        "ingredients": ["riso", "zucchine", "yogurt-greco", "olio"],
        "minutes": 25,
        "tags": ["vegetariano", "senza-glutine"],
    },
    {
        "name": "Pasta al pesto di rucola",
        "ingredients": ["pasta", "rucola", "parmigiano", "noci"],
        "minutes": 20,
        "tags": ["vegetariano", "contiene-frutta-secca"],
    },
    {
        "name": "Hummus express",
        "ingredients": ["ceci", "limone", "tahini", "aglio"],
        "minutes": 10,
        "tags": ["vegetariano", "vegano", "senza-glutine"],
    },
]

SHOPPING_LIST_FILE = Path("tmp/shopping_list.json")

# endregion

# region Tools


def _find_recipes(
    available_ingredients: list[str],
    dietary_constraints: list[str],
    max_minutes: int,
) -> list[dict]:
    """Ricerca pura riusabile da tool e workflow."""
    available_set = set(i.lower() for i in available_ingredients)
    constraints_set = set(c.lower() for c in dietary_constraints)

    results = []
    for recipe in RECIPES_DB:
        if recipe["minutes"] > max_minutes:
            continue
        if constraints_set and not constraints_set.issubset(set(recipe["tags"])):
            continue
        recipe_ingredients = set(i.lower() for i in recipe["ingredients"])
        if not recipe_ingredients.issubset(available_set):
            missing = recipe_ingredients - available_set
            if len(missing) > 2:
                continue
        results.append(recipe)

    return results


@tool
def find_recipes_by_constraints(
    available_ingredients: list[str],
    dietary_constraints: list[str] = [],
    max_minutes: int = 60,
) -> list[dict]:
    """Trova ricette compatibili con ingredienti disponibili, vincoli e tempo."""
    return _find_recipes(available_ingredients, dietary_constraints, max_minutes)


@tool
def save_shopping_list(items: list[str]) -> str:
    """Salva la lista della spesa su file."""
    SHOPPING_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHOPPING_LIST_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return f"Lista della spesa salvata con {len(items)} ingredienti in {SHOPPING_LIST_FILE}."


# endregion

# region Models


class RecipeRecommendation(BaseModel):
    """Risposta strutturata di RecipeBot."""

    selected_recipe: str = Field(description="Nome della ricetta scelta.")
    estimated_minutes: int = Field(description="Tempo di preparazione in minuti.")
    missing_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredienti che l'utente deve comprare per fare questa ricetta.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Avvisi rilevanti per l'utente: allergeni potenziali, "
            "vincoli a rischio, conflitti dietetici."
        ),
    )
    reason: str = Field(
        description="Breve spiegazione del perché questa ricetta è stata scelta."
    )


class Constraints(BaseModel):
    """Vincoli alimentari normalizzati estratti dall'input dell'utente."""

    available_ingredients: list[str] = Field(default_factory=list)
    dietary_constraints: list[str] = Field(default_factory=list)
    excluded_ingredients: list[str] = Field(default_factory=list)
    max_minutes: int = Field(default=60)


# endregion

# Knowledge Base

knowledge = Knowledge(
    vector_db=LanceDb(
        table_name="recipebot_knowledge",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(),
    ),
)

# Inserimento idempotente: se i file non sono cambiati, è veloce.
try:
    knowledge.insert(path="docs/ricettario.md")
    knowledge.insert(path="docs/allergeni.md")
except Exception:
    # Non bloccare l'avvio se i file non esistono durante lo sviluppo
    pass

# endregion

db = SqliteDb(db_file="tmp/recipebot.db")

recipebot = Agent(
    name="RecipeBot",
    description="Assistente di cucina che propone ricette e gestisce la lista della spesa.",
    model=Gemini(id="gemini-2.5-flash"),
    db=db,
    instructions=[
        "Sei RecipeBot, un assistente di cucina.",
        "Distingui i vincoli HARD (allergie, diete) dalle preferenze SOFT (gusti).",
        "Non suggerire mai ingredienti che violino le allergie dichiarate dall'utente.",
        "Quando l'utente non specifica un vincolo, chiedi prima di assumere.",
        "Per richieste con più vincoli, usa think() per decomporli prima di agire.",
        "Dopo aver chiamato find_recipes_by_constraints, usa analyze() per valutare i risultati.",
        "Se nessuna ricetta soddisfa i vincoli, segnala il conflitto in warnings.",
        "Usa find_recipes_by_constraints per cercare ricette compatibili.",
        "Usa save_shopping_list quando l'utente chiede di salvare gli ingredienti mancanti.",
        "Restituisci sempre una raccomandazione strutturata secondo lo schema RecipeRecommendation.",
        "Consulta la knowledge base per: tecniche di cottura, allergeni, sostituzioni di ingredienti.",
        "Quando l'utente menziona allergie, preferenze o caratteristiche personali, salva queste informazioni nei tool di learning per ricordarle nelle conversazioni future.",
    ],
    tools=[
        ReasoningTools(add_instructions=True),
        find_recipes_by_constraints,
        save_shopping_list,
    ],
    knowledge=knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
    ),
    add_learnings_to_context=False,
    add_history_to_context=True,
    num_history_runs=3,
    debug_mode=True,
    markdown=True,
)


# === NUOVO: Workflow a 3 step ============================================

constraint_extractor = Agent(
    name="ConstraintExtractor",
    description="Estrae vincoli alimentari strutturati dal messaggio dell'utente.",
    model=Gemini(id="gemini-2.5-flash"),
    instructions=[
        "Sei un estrattore di vincoli alimentari.",
        "REGOLE:",
        "- excluded_ingredients: TUTTO cio' che l'utente non vuole (allergie, intolleranze, avversioni).",
        "- dietary_constraints: solo tag positivi: vegetariano, vegano, senza-glutine.",
        "Quando in dubbio, includi nelle esclusioni: meglio scartare una ricetta valida che proporne una pericolosa.",
    ],
    output_schema=Constraints,
    use_json_mode=True,
)


def search_recipes_step(step_input: StepInput) -> StepOutput:
    """Step deterministico: query al RECIPES_DB."""
    constraints: Constraints = step_input.previous_step_content
    candidates = _find_recipes(
        available_ingredients=constraints.available_ingredients,
        dietary_constraints=constraints.dietary_constraints,
        max_minutes=constraints.max_minutes,
    )
    return StepOutput(
        content={
            "candidates": candidates,
            "constraints": constraints.model_dump(),
        }
    )


def filter_and_format_step(step_input: StepInput) -> StepOutput:
    """Step deterministico: filtra esclusioni e formatta."""
    data = step_input.previous_step_content
    candidates = data["candidates"]
    excluded = set(i.lower() for i in data["constraints"]["excluded_ingredients"]) 

    safe = [
        r for r in candidates
        if not any(ing.lower() in excluded for ing in r["ingredients"])
    ]

    if not safe:
        return StepOutput(content=f"Nessuna ricetta soddisfa i vincoli (escluse {len(excluded)}).")

    chosen = safe[0]
    return StepOutput(
        content=(
            f"**Ricetta consigliata: {chosen['name']}**\n\n"
            f"- Ingredienti: {', '.join(chosen['ingredients'])}\n"
            f"- Tempo: {chosen['minutes']} minuti\n"
            f"- Esclusioni applicate: {', '.join(sorted(excluded)) or 'nessuna'}"
        )
    )


recipe_workflow = Workflow(
    name="RecipeWorkflow",
    description="Pipeline a 3 step: estrai vincoli, cerca, filtra esclusioni.",
    db=db,
    steps=[
        Step(name="extract_constraints", agent=constraint_extractor),
        Step(name="search_recipes", executor=search_recipes_step),
        Step(name="filter_and_format", executor=filter_and_format_step),
    ],
)

agent_os = AgentOS(
    name="RecipeBot OS",
    agents=[recipebot],
    workflows=[recipe_workflow],
    db=db,
    tracing=True,
)

app = agent_os.get_app()


if __name__ == "__main__":
    print("RecipeBot app created. Use an ASGI server (uvicorn) to run `app` if desired.")
