# """
# "What's New" feature — proactive use of the MCP `list_recent_updates` tool.
 
# This is a DIFFERENT agentic pattern from the freshness check in
# rag_pipeline.py:
#     - rag_pipeline.py's MCP use is REACTIVE: the user asks a question, chunks
#       are retrieved, THEN we verify their freshness before answering.
#     - This module's MCP use is PROACTIVE: independent of any user question,
#       we go ask "what changed recently?" and surface it directly — then
#       optionally summarize it with one LLM call.
 
# Used by app/streamlit_app.py's sidebar.
# """
# import asyncio
# from datetime import date, timedelta
 
# import boto3
 
# from app import config
# from app.mcp_client import MCPToolClient
 
# _bedrock = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=config.AWS_REGION,
#     aws_access_key_id=config.AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
# )
 
# SUMMARY_SYSTEM = """You summarize recent internal wiki changes for an employee.
# Given a list of updated pages (title, space, status, last_updated), write a
# short, friendly summary as 3-5 bullet points, grouped naturally by topic or
# space where it makes sense. Explicitly flag any page marked OUTDATED or
# DEPRECATED so the reader knows not to treat it as current policy. Be concise
# — this is a quick "what changed" digest, not a full report."""
 
 
# async def _fetch_recent_updates(space: str, days: int) -> list:
#     since_date = (date.today() - timedelta(days=days)).isoformat()
#     async with MCPToolClient() as tools:
#         return await tools.list_recent_updates(space=space, since_date=since_date)
 
 
# def get_recent_updates(space: str = "", days: int = None) -> list:
#     """Sync wrapper safe to call directly from Streamlit.
 
#     Returns a list of dicts (page_id, title, space, status, last_updated,
#     tags) as produced by mcp_server/server.py's list_recent_updates tool.
#     Returns [] (never raises) if the MCP server is unreachable — this
#     sidebar feature should degrade gracefully, not crash the app.
#     """
#     days = days if days is not None else config.RECENT_UPDATES_DAYS
#     try:
#         return asyncio.run(_fetch_recent_updates(space, days))
#     except Exception as e:
#         print(f"[whats_new] failed to fetch recent updates: {e}")
#         return []
 
 
# def summarize_updates(pages: list) -> str:
#     """One LLM call over the recent-updates list -> short human-readable
#     summary. Assumes `pages` is non-empty (caller should check first).
#     """
#     if not pages:
#         return "No recent updates to summarize."
 
#     lines = []
#     for p in pages:
#         status = (p.get("status") or "").lower()
#         flag = f" [{status.upper()}]" if status not in ("current", "", "unknown") else ""
#         lines.append(
#             f"- [{p.get('page_id', '?')}] {p.get('title', 'Untitled')} "
#             f"(space: {p.get('space', '?')}, updated: {p.get('last_updated', '?')}){flag}"
#         )
#     prompt = "RECENTLY UPDATED PAGES:\n\n" + "\n".join(lines)
 
#     response = _bedrock.converse(
#         modelId=config.GENERATION_MODEL,
#         system=[{"text": SUMMARY_SYSTEM}],
#         messages=[{"role": "user", "content": [{"text": prompt}]}],
#         inferenceConfig={"maxTokens": 400},
#     )
#     return response["output"]["message"]["content"][0]["text"]
"""
"What's New" feature — proactive use of the MCP `list_recent_updates` tool.
 
This is a DIFFERENT agentic pattern from the freshness check in
rag_pipeline.py:
    - rag_pipeline.py's MCP use is REACTIVE: the user asks a question, chunks
      are retrieved, THEN we verify their freshness before answering.
    - This module's MCP use is PROACTIVE: independent of any user question,
      we go ask "what changed recently?" and surface it directly — then
      optionally summarize it with one LLM call.
 
Used by app/streamlit_app.py's sidebar.
"""
import asyncio
from datetime import date, timedelta
 
import boto3
 
from app import config
from app.mcp_client import MCPToolClient
 
_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=config.AWS_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
)
 
SUMMARY_SYSTEM = """You summarize recent internal wiki changes for an employee.
Given a list of updated pages (title, space, status, last_updated), write a
short, friendly summary as 3-5 bullet points, grouped naturally by topic or
space where it makes sense. Explicitly flag any page marked OUTDATED or
DEPRECATED so the reader knows not to treat it as current policy. Be concise
— this is a quick "what changed" digest, not a full report."""
 
 
async def _fetch_recent_updates(space: str, days: int) -> list:
    since_date = (date.today() - timedelta(days=days)).isoformat()
    async with MCPToolClient() as tools:
        return await tools.list_recent_updates(space=space, since_date=since_date)
 
 
def get_recent_updates(space: str = "", days: int = None) -> list:
    """Sync wrapper safe to call directly from Streamlit.
 
    Returns a list of dicts (page_id, title, space, status, last_updated,
    tags) as produced by mcp_server/server.py's list_recent_updates tool.
    Returns [] (never raises) if the MCP server is unreachable — this
    sidebar feature should degrade gracefully, not crash the app.
    """
    days = days if days is not None else config.RECENT_UPDATES_DAYS
    try:
        result = asyncio.run(_fetch_recent_updates(space, days))
    except Exception as e:
        print(f"[whats_new] failed to fetch recent updates: {e}")
        return []
 
    print(f"[whats_new] DEBUG raw result type={type(result).__name__} "
          f"value_preview={str(result)[:200]}")
 
    if isinstance(result, dict):
        # Defensive: some MCP client paths hand back the raw tool-result
        # envelope (e.g. {"content": [...]}) or a dict keyed by page_id
        # instead of the parsed list. Try the common shapes before giving up.
        if "content" in result:
            print("[whats_new] got raw MCP envelope instead of parsed list; "
                  "check MCPToolClient.list_recent_updates() unwrapping")
            return []
        result = list(result.values())
    elif isinstance(result, set):
        result = list(result)
    elif not isinstance(result, list):
        print(f"[whats_new] unexpected type from list_recent_updates: {type(result)}")
        return []
 
    return result
 
 
def summarize_updates(pages: list) -> str:
    """One LLM call over the recent-updates list -> short human-readable
    summary. Assumes `pages` is non-empty (caller should check first).
    """
    if not pages:
        return "No recent updates to summarize."
 
    lines = []
    for p in pages:
        status = (p.get("status") or "").lower()
        flag = f" [{status.upper()}]" if status not in ("current", "", "unknown") else ""
        lines.append(
            f"- [{p.get('page_id', '?')}] {p.get('title', 'Untitled')} "
            f"(space: {p.get('space', '?')}, updated: {p.get('last_updated', '?')}){flag}"
        )
    prompt = "RECENTLY UPDATED PAGES:\n\n" + "\n".join(lines)
 
    response = _bedrock.converse(
        modelId=config.GENERATION_MODEL,
        system=[{"text": SUMMARY_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400},
    )
    return response["output"]["message"]["content"][0]["text"]
 
 








