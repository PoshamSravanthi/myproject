from fastapi import FastAPI
from pydantic import BaseModel
from app import run_multi_agent_workflow, WorkflowInput

app = FastAPI(title="Multi-Agent Research API")


# ---------- REQUEST MODEL ----------
class QueryRequest(BaseModel):
    query: str


# ---------- API ENDPOINT ----------
@app.post("/run-workflow")
async def run_workflow(request: QueryRequest):
    """
    Runs the Research → Summarizer → Email agent workflow
    """
    input_data = WorkflowInput(query=request.query)
    result = run_multi_agent_workflow(input_data)
    return result
