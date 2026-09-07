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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
