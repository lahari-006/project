"""
End-to-end RAG (lightly agentic) pipeline for the Internal Knowledge Navigator.
 
Flow for one query:
    1. Guardrails INPUT check (block empty/injection/toxic input)
    2. Retrieve top-k chunks from ChromaDB
    3. Call the MCP tool (get_page_status) for every distinct page_id in the
       retrieved set, to confirm freshness "live" rather than trusting the
       status captured at index time — this is the agentic tool-use step.
    4. Build a grounded prompt (context + freshness flags) and call the LLM
       via AWS Bedrock (amazon.nova-micro-v1:0 by default)
    5. Guardrails OUTPUT check (groundedness / PII) on the generated answer
    6. Log everything to Langfuse (latency, cost, errors) and return
 
This module exposes one clean sync entrypoint, `answer_query()`, so it can be
called from a CLI (app/main.py), a Streamlit UI (app/streamlit_app.py), a
FastAPI layer (app/api.py), or the eval harness (app/generate_answers.py)
without duplicating logic.
 
Every returned dict includes `trace_id` / `trace_url` (None if Langfuse
isn't configured) so a UI can link directly to the trace in the Langfuse
dashboard.
"""
import asyncio
import time
 
import boto3
 
from app import config
from app.retriever import Retriever
from app.mcp_client import MCPToolClient
from app.guardrails_setup import check_input, check_output
from app.observability import QueryTrace
 
# ---------------------------------------------------------------------
# Bedrock client (replaces the old `anthropic.Anthropic(...)` client).
# Uses the standard Bedrock Runtime `converse()` API, which works the
# same way across model families (Nova, Claude, Llama, etc.) so swapping
# GENERATION_MODEL later doesn't require touching this code again.
# ---------------------------------------------------------------------
_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=config.AWS_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
)
 
_retriever = None  # lazy singleton, ChromaDB/embedder init is not free
 
 
def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
 
 
SYSTEM_PROMPT = """You are the Internal Knowledge Navigator, an assistant that answers \
employee questions using ONLY the internal wiki excerpts provided below.
 
Rules:
- Answer strictly from the provided context. If the context doesn't contain \
the answer, say so plainly instead of guessing.
- If any of the provided excerpts are flagged OUTDATED or DEPRECATED, do not \
present their content as current policy/process. Prefer CURRENT excerpts, \
and explicitly warn the user when the only relevant information available is \
outdated or deprecated.
- Cite the page title and page ID for every claim, like: (PTO and Leave \
Policy, HR-001).
- Be concise and direct."""
 
 
def _build_prompt(query: str, chunks: list, freshness: dict) -> str:
    context_blocks = []
    for c in chunks:
        live_status = freshness.get(c["page_id"], {}).get("status", c["status"])
        flag = "" if live_status == "current" else f" [{live_status.upper()}]"
        context_blocks.append(
            f"[{c['page_id']}{flag}] {c['title']} > {c['heading']}\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT EXCERPTS:\n\n{context}\n\nQUESTION: {query}"
 
 
async def _check_freshness(chunks: list) -> dict:
    """Call the MCP server once per distinct page_id in the retrieved set."""
    page_ids = sorted({c["page_id"] for c in chunks if c["page_id"]})
    freshness = {}
    try:
        async with MCPToolClient() as tools:
            for pid in page_ids:
                freshness[pid] = await tools.get_page_status(pid)
    except Exception as e:
        # MCP server unreachable shouldn't take the whole app down — fall back
        # to the status captured at index time and surface the error.
        print(f"[mcp] freshness check failed, falling back to indexed status: {e}")
    return freshness
 
 
def _call_bedrock(system_prompt: str, prompt: str, max_tokens: int):
    """Call the Bedrock Runtime `converse` API and return (answer_text, usage_dict).
 
    `usage_dict` has keys: inputTokens, outputTokens, totalTokens.
    Works uniformly across Bedrock model families (Nova, Claude, Llama, etc.).
    """
    response = _bedrock.converse(
        modelId=config.GENERATION_MODEL,
        system=[{"text": system_prompt}],
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={"maxTokens": max_tokens},
    )
    answer_text = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    return answer_text, usage
 
 
def answer_query(query: str, k: int = None, user_id: str = "demo-user") -> dict:
    """Synchronous entrypoint. Returns a dict:
        {answer, sources, blocked, block_reason, latency_seconds, cost_usd,
         trace_id, trace_url}
    """
    trace = QueryTrace(query, user_id=user_id)
 
    # 1. INPUT guardrail --------------------------------------------------
    guard_in = check_input(query)
    if not guard_in.passed:
        latency = trace.finish(guard_in.message, guard_blocked=True)
        return {
            "answer": guard_in.message,
            "sources": [],
            "blocked": True,
            "block_reason": guard_in.category,
            "latency_seconds": latency,
            "cost_usd": 0.0,
            "trace_id": trace.trace_id,
            "trace_url": trace.trace_url,
        }
 
    # 2. Retrieval ----------------------------------------------------------
    with trace.span("retrieval", query=query, k=k or config.TOP_K):
        chunks = _get_retriever().retrieve(query, k=k)
 
    if not chunks:
        answer = "I couldn't find anything relevant to that in the knowledge base."
        latency = trace.finish(answer)
        return {
            "answer": answer, "sources": [], "blocked": False,
            "block_reason": None, "latency_seconds": latency, "cost_usd": 0.0,
            "trace_id": trace.trace_id, "trace_url": trace.trace_url,
        }
 
    # 3. Agentic MCP tool call: live freshness check --------------------
    with trace.span("mcp_freshness_check", page_ids=list({c["page_id"] for c in chunks})):
        freshness = asyncio.run(_check_freshness(chunks))
 
    # 4. Generation -------------------------------------------------------
    prompt = _build_prompt(query, chunks, freshness)
    t0 = time.time()
    try:
        with trace.span("generation", model=config.GENERATION_MODEL):
            answer_text, usage = _call_bedrock(
                SYSTEM_PROMPT, prompt, config.MAX_TOKENS
            )
        cost = trace.log_generation(
            config.GENERATION_MODEL, prompt, answer_text,
            usage["inputTokens"], usage["outputTokens"],
        )
    except Exception as e:
        latency = trace.finish(f"Generation failed: {e}")
        return {
            "answer": "Something went wrong generating an answer. Please try again.",
            "sources": [], "blocked": True, "block_reason": "generation_error",
            "latency_seconds": latency, "cost_usd": 0.0,
            "trace_id": trace.trace_id, "trace_url": trace.trace_url,
        }
 
    # 5. OUTPUT guardrail ---------------------------------------------------
    guard_out = check_output(
        answer_text, [c["text"] for c in chunks], threshold=config.GROUNDEDNESS_THRESHOLD
    )
    final_answer = answer_text if guard_out.passed else guard_out.message
 
    latency = trace.finish(final_answer, guard_blocked=not guard_out.passed)
 
    sources = [
        {
            "page_id": c["page_id"],
            "title": c["title"],
            "heading": c["heading"],
            "status": freshness.get(c["page_id"], {}).get("status", c["status"]),
            "last_updated": freshness.get(c["page_id"], {}).get("last_updated", c["last_updated"]),
        }
        for c in chunks
    ]
 
    return {
        "answer": final_answer,
        "sources": sources,
        "blocked": not guard_out.passed,
        "block_reason": None if guard_out.passed else guard_out.category,
        "latency_seconds": round(latency, 3),
        "cost_usd": cost,
        "trace_id": trace.trace_id,
        "trace_url": trace.trace_url,
        # raw context kept around for the eval harness (RAGAS needs it)
        "_contexts": [c["text"] for c in chunks],
    }
 
 
if __name__ == "__main__":
    result = answer_query("How many PTO days do I get after 3 years?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(" -", s)
    print(f"\nLatency: {result['latency_seconds']}s | Cost: ${result['cost_usd']}")
    if result.get("trace_url"):
        print(f"Trace: {result['trace_url']}")
 
