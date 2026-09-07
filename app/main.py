from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from agno.os import AgentOS
from agent import cinema_team, db
from query_logger import QueryLoggerMiddleware

agent_os = AgentOS(
    name="Cinema GraphRAG",
    teams=[cinema_team],
    db=db,
    tracing=True,
)

app = agent_os.get_app()
app.add_middleware(QueryLoggerMiddleware)

if __name__ == "__main__":
    import uvicorn
    # app_dir esplicito: con reload=True uvicorn rilancia "main:app" in un
    # sottoprocesso che risolve i moduli rispetto alla working directory di
    # avvio, non rispetto a questo file — se lanciato da fuori app/ (es. `uv
    # run python app/main.py` dalla radice) senza questo, il sottoprocesso non
    # troverebbe il modulo "main".
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(Path(__file__).resolve().parent))
