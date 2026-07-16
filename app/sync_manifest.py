"""
Regenerates CORPUS_DIR/manifest.json directly from the HTML pages' own
<meta> tags (and <title>).
 
WHY THIS EXISTS: mcp_server/server.py's get_page_status / list_recent_updates
tools read page status/last-updated data from manifest.json — NOT from the
HTML files directly. Adding, editing, or removing a .html page in the corpus
(or running `python -m app.ingest`, which only rebuilds the ChromaDB vector
index used for chat retrieval) does NOT update manifest.json on its own.
That mismatch is exactly what causes a newly-added page to show up in chat
answers but never appear in the "What's New" sidebar / freshness checks.
 
Run this any time you add/edit/remove a page, in addition to app.ingest:
    python -m app.sync_manifest
 
This OVERWRITES manifest.json with a fresh scan of every .html file in
CORPUS_DIR — it is not additive. That's intentional: the HTML files are the
single source of truth, manifest.json is a derived cache for the MCP server.
"""
import glob
import json
import os
 
from bs4 import BeautifulSoup
 
from app import config
 
 
def build_manifest() -> list:
    files = glob.glob(os.path.join(config.CORPUS_DIR, "*", "*.html"))
    records = []
    skipped = 0
 
    for fp in sorted(files):
        try:
            with open(fp, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
 
            meta = {
                m.get("name"): m.get("content", "")
                for m in soup.find_all("meta") if m.get("name")
            }
            # <title> isn't a <meta> tag, so pull it separately — this is
            # the field mcp_server.py's _normalize() looks for via
            # get("title", "name").
            title_tag = soup.find("title")
            meta["title"] = title_tag.text.strip() if title_tag else ""
 
            if not meta.get("confluence-page-id"):
                print(f"  [skip] {fp}: no confluence-page-id meta tag found")
                skipped += 1
                continue
 
            records.append(meta)
        except Exception as e:
            print(f"  [skip] {fp}: {e}")
            skipped += 1
 
    print(f"Scanned {len(files)} HTML files -> {len(records)} manifest records "
          f"({skipped} skipped)")
    return records
 
 
def main():
    print(f"Corpus dir    : {config.CORPUS_DIR}")
    print(f"Manifest path : {config.MANIFEST_PATH}")
 
    if not os.path.exists(config.CORPUS_DIR):
        raise SystemExit(f"CORPUS_DIR does not exist: {config.CORPUS_DIR}")
 
    records = build_manifest()
 
    os.makedirs(os.path.dirname(config.MANIFEST_PATH), exist_ok=True)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
 
    print(f"\nWrote {len(records)} page records to {config.MANIFEST_PATH}")
    print("The MCP server (get_page_status / list_recent_updates / search_pages) "
          "will pick this up on its next call — no restart needed, since it "
          "re-reads the file on every tool call.")
 
 
if __name__ == "__main__":
    main()
 











































