# """
# Custom MCP server for the Internal Knowledge Navigator.

# Why this exists: the RAG pipeline retrieves chunks from a *static* vector
# index that was built at ingest time. In a real Confluence/Notion workspace,
# page status (current / outdated / deprecated) and last-updated dates change
# continuously. This MCP server simulates the "live" source-of-truth check a
# real agent would do against the Confluence/Notion API before trusting a
# retrieved chunk, and also powers simple query-analytics style lookups
# (e.g. "what changed recently in ENG?").

# Exposed tools:
#     - get_page_status(page_id)      -> live status/last_updated/title for one page
#     - list_recent_updates(space, since_date) -> pages touched after a date
#     - search_pages(keyword)         -> pages whose title/tags match a keyword

# Run standalone for debugging:
#     python mcp_server/server.py
# It is normally launched automatically (as a subprocess, over stdio) by
# app/mcp_client.py.
# """
# import csv
# import json
# import os
# import sys
# from pathlib import Path

# from mcp.server.fastmcp import FastMCP

# ROOT_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# from app import config  # noqa: E402

# mcp = FastMCP("knowledge-navigator-manifest")


# def _load_manifest():
#     """Load the corpus manifest, trying JSON first, then CSV.

#     Returns a list of dicts. Works whichever shape the export tool produced
#     (list-of-records JSON, {"pages": [...]}-style JSON, or a flat CSV).
#     """
#     manifest_json = Path(config.CORPUS_DIR) / "manifest.json"
#     manifest_csv = Path(config.CORPUS_DIR) / "manifest.csv"

#     if manifest_json.exists():
#         data = json.loads(manifest_json.read_text(encoding="utf-8"))
#         if isinstance(data, dict):
#             for key in ("pages", "items", "data"):
#                 if key in data:
#                     return data[key]
#             return list(data.values())
#         return data

#     if manifest_csv.exists():
#         with open(manifest_csv, newline="", encoding="utf-8") as f:
#             return list(csv.DictReader(f))

#     raise FileNotFoundError(
#         f"No manifest.json or manifest.csv found under {config.CORPUS_DIR}"
#     )


# def _normalize(record):
#     """Map whatever the export's column names are onto a common shape."""
#     def get(*keys, default=""):
#         for k in keys:
#             if k in record and record[k] not in (None, ""):
#                 return record[k]
#         return default

#     return {
#         "page_id": get("page_id", "confluence-page-id", "id"),
#         "title": get("title", "name"),
#         "space": get("space", "confluence-space"),
#         "status": get("status", default="unknown"),
#         "last_updated": get("last_updated", "last-updated", default=""),
#         "tags": get("tags", default=""),
#     }


# @mcp.tool()
# def get_page_status(page_id: str) -> dict:
#     """Look up the live status of a single wiki page by its page ID.

#     Use this after retrieving a chunk to double-check whether the page it
#     came from is still 'current', or has since been marked 'outdated' /
#     'deprecated', before presenting the answer as authoritative.
#     """
#     records = [_normalize(r) for r in _load_manifest()]
#     for r in records:
#         if r["page_id"] == page_id:
#             return r
#     return {"error": f"No page found with id '{page_id}'"}


# @mcp.tool()
# def list_recent_updates(space: str = "", since_date: str = "1900-01-01") -> list:
#     """List pages updated on/after `since_date` (YYYY-MM-DD), optionally
#     filtered to one space (e.g. 'ENG', 'HR', 'ITS', 'OPS', 'PROD').

#     Useful for query-analytics style questions like
#     'what changed in Engineering docs recently?'.
#     """
#     records = [_normalize(r) for r in _load_manifest()]
#     out = []
#     for r in records:
#         if space and r["space"].lower() != space.lower():
#             continue
#         if r["last_updated"] and r["last_updated"] >= since_date:
#             out.append(r)
#     out.sort(key=lambda r: r["last_updated"], reverse=True)
#     return out


# @mcp.tool()
# def search_pages(keyword: str) -> list:
#     """Search page titles and tags for a keyword. Returns matching page
#     records with their current status, useful for disambiguating which
#     page(s) a vague query might be referring to."""
#     keyword = keyword.lower()
#     records = [_normalize(r) for r in _load_manifest()]
#     return [
#         r for r in records
#         if keyword in r["title"].lower() or keyword in r["tags"].lower()
#     ]


# if __name__ == "__main__":
#     mcp.run(transport="stdio")
"""
Custom MCP server for the Internal Knowledge Navigator.
 
Why this exists: the RAG pipeline retrieves chunks from a *static* vector
index that was built at ingest time. In a real Confluence/Notion workspace,
page status (current / outdated / deprecated) and last-updated dates change
continuously. This MCP server simulates the "live" source-of-truth check a
real agent would do against the Confluence/Notion API before trusting a
retrieved chunk, and also powers simple query-analytics style lookups
(e.g. "what changed recently in ENG?").
 
Exposed tools:
    - get_page_status(page_id)      -> live status/last_updated/title for one page
    - list_recent_updates(space, since_date) -> pages touched after a date
    - search_pages(keyword)         -> pages whose title/tags match a keyword
 
Run standalone for debugging:
    python mcp_server/server.py
It is normally launched automatically (as a subprocess, over stdio) by
app/mcp_client.py.
"""
import csv
import json
import os
import sys
from pathlib import Path
 
from mcp.server.fastmcp import FastMCP
 
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
 
from app import config  # noqa: E402
 
mcp = FastMCP("knowledge-navigator-manifest")
 
 
def _load_manifest():
    """Load the corpus manifest, trying JSON first, then CSV.
 
    Returns a list of dicts. Works whichever shape the export tool produced
    (list-of-records JSON, {"pages": [...]}-style JSON, or a flat CSV).
    """
    manifest_json = Path(config.CORPUS_DIR) / "manifest.json"
    manifest_csv = Path(config.CORPUS_DIR) / "manifest.csv"
 
    if manifest_json.exists():
        data = json.loads(manifest_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("pages", "items", "data"):
                if key in data:
                    return data[key]
            return list(data.values())
        return data
 
    if manifest_csv.exists():
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
 
    raise FileNotFoundError(
        f"No manifest.json or manifest.csv found under {config.CORPUS_DIR}"
    )
 
 
def _normalize(record):
    """Map whatever the export's column names are onto a common shape."""
    def get(*keys, default=""):
        for k in keys:
            if k in record and record[k] not in (None, ""):
                return record[k]
        return default
 
    return {
        "page_id": get("page_id", "confluence-page-id", "id"),
        "title": get("title", "name"),
        "space": get("space", "confluence-space"),
        "status": get("status", default="unknown"),
        "last_updated": get("last_updated", "last-updated", default=""),
        "tags": get("tags", default=""),
    }
 
 
@mcp.tool()
def get_page_status(page_id: str) -> dict:
    """Look up the live status of a single wiki page by its page ID.
 
    Use this after retrieving a chunk to double-check whether the page it
    came from is still 'current', or has since been marked 'outdated' /
    'deprecated', before presenting the answer as authoritative.
    """
    records = [_normalize(r) for r in _load_manifest()]
    for r in records:
        if r["page_id"] == page_id:
            return r
    return {"error": f"No page found with id '{page_id}'"}
 
 
@mcp.tool()
def list_recent_updates(space: str = "", since_date: str = "1900-01-01") -> list[dict]:
    """List pages updated on/after `since_date` (YYYY-MM-DD), optionally
    filtered to one space (e.g. 'ENG', 'HR', 'ITS', 'OPS', 'PROD').
 
    Useful for query-analytics style questions like
    'what changed in Engineering docs recently?'.
    """
    records = [_normalize(r) for r in _load_manifest()]
    out = []
    for r in records:
        if space and r["space"].lower() != space.lower():
            continue
        if r["last_updated"] and r["last_updated"] >= since_date:
            out.append(r)
    out.sort(key=lambda r: r["last_updated"], reverse=True)
    return out
 
 
@mcp.tool()
def search_pages(keyword: str) -> list[dict]:
    """Search page titles and tags for a keyword. Returns matching page
    records with their current status, useful for disambiguating which
    page(s) a vague query might be referring to."""
    keyword = keyword.lower()
    records = [_normalize(r) for r in _load_manifest()]
    return [
        r for r in records
        if keyword in r["title"].lower() or keyword in r["tags"].lower()
    ]
 
 
if __name__ == "__main__":
    mcp.run(transport="stdio")
 








