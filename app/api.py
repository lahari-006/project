"""
Minimal FastAPI wrapper around the RAG pipeline, so your future Streamlit
frontend (or Postman/curl for the demo) can hit a clean HTTP endpoint
instead of importing python modules directly.

Run:
    uvicorn app.api:app --reload --port 8000

Then:
    POST http://localhost:8000/query   {"question": "..."}
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import answer_query

app = FastAPI(title="Internal Knowledge Navigator API")


class QueryRequest(BaseModel):
    question: str
    k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    blocked: bool
    block_reason: str | None
    latency_seconds: float
    cost_usd: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = answer_query(req.question, k=req.k)
    result.pop("_contexts", None)  # internal-only field, not part of the API contract
    return result
