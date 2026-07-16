"""
Ingestion pipeline for the Confluence/Notion HTML export.

This is a refactor of the user's original notebook cells into a reusable,
re-runnable module:
    - parse_page()     : one HTML file -> (metadata, sections)
    - chunk_page()      : one HTML file -> list[chunk dict]
    - ingest_corpus()   : whole corpus dir -> list[chunk dict]
    - build_index()     : chunks -> persisted ChromaDB collection (+ vectorizer if TF-IDF)

Run directly to (re)build the index from scratch:
    python -m app.ingest
"""
import glob
import os
import shutil

import chromadb
import pandas as pd
from bs4 import BeautifulSoup

from app import config
from app.embeddings import Embedder


# ----------------------------------------------------------------------
# Parsing (unchanged logic from the notebook, just moved into a module)
# ----------------------------------------------------------------------
def parse_page(filepath):
    """Read one HTML wiki page and return (metadata_dict, list_of_(heading, text)_sections)."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    meta = {m.get("name"): m.get("content", "") for m in soup.find_all("meta") if m.get("name")}

    content_div = soup.find("div", class_="page-content")
    sections = []
    current_heading = soup.find("title").text
    current_parts = []

    def flush():
        if current_parts:
            sections.append((current_heading, "\n".join(current_parts).strip()))

    for el in content_div.children:
        tag = getattr(el, "name", None)
        if tag in ("h1", "h2"):
            flush()
            current_heading = el.get_text(strip=True)
            current_parts = []
        elif tag == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                rows.append(" | ".join(cells))
            current_parts.append("\n".join(rows))
        elif tag == "pre":
            current_parts.append(f"[CODE]\n{el.get_text().strip()}\n[/CODE]")
        elif tag in ("p", "ul", "ol"):
            text = el.get_text(" ", strip=True)
            if text:
                current_parts.append(text)
    flush()

    return meta, sections


def chunk_page(filepath, max_chars=1800, overlap_ratio=0.15):
    """Turn one HTML page into a list of chunk dicts, each carrying full metadata."""
    meta, sections = parse_page(filepath)
    with open(filepath, encoding="utf-8") as f:
        title = BeautifulSoup(f.read(), "html.parser").title.text

    raw_chunks = []
    for heading, text in sections:
        if not text:
            continue
        if len(text) <= max_chars:
            raw_chunks.append((heading, text))
        else:
            step = int(max_chars * (1 - overlap_ratio))
            for i in range(0, len(text), step):
                raw_chunks.append((heading, text[i:i + max_chars]))

    chunks = []
    for i, (heading, text) in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{meta.get('confluence-page-id', '?')}::chunk{i}",
            "page_id": meta.get("confluence-page-id", ""),
            "title": title,
            "space": meta.get("confluence-space", ""),
            "status": meta.get("status", ""),  # current | outdated | deprecated
            "last_updated": meta.get("last-updated", ""),
            "tags": meta.get("tags", ""),
            "heading_path": heading,
            "text": text,
        })
    return chunks


def ingest_corpus(corpus_dir):
    files = glob.glob(os.path.join(corpus_dir, "*", "*.html"))
    all_chunks = []
    errors = 0
    for fp in files:
        try:
            all_chunks.extend(chunk_page(fp))
        except Exception as e:
            errors += 1
            print(f"  [skip] {fp}: {e}")
    print(f"Ingested {len(files)} pages -> {len(all_chunks)} chunks ({errors} errors)")
    return all_chunks


# ----------------------------------------------------------------------
# Indexing: chunks -> ChromaDB (embedding backend is pluggable)
# ----------------------------------------------------------------------
def build_index(chunks, chroma_path=None, collection_name=None):
    chroma_path = chroma_path or config.CHROMA_PATH
    collection_name = collection_name or config.CHROMA_COLLECTION

    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)  # fresh start each rebuild, same as the notebook

    embedder = Embedder(backend=config.EMBEDDING_BACKEND)
    texts = [f"{c['title']}. {c['heading_path']}. {c['text']}" for c in chunks]
    embedder.fit(texts)  # no-op for sbert, fits+saves TF-IDF vocabulary otherwise
    embedding_matrix = embedder.embed_documents(texts)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embedding_matrix,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "page_id": c["page_id"],
                "title": c["title"],
                "space": c["space"],
                "status": c["status"],
                "heading": c["heading_path"],
                "last_updated": c["last_updated"],
                "tags": c["tags"],
            }
            for c in chunks
        ],
    )
    print(f"Stored in ChromaDB: {collection.count()} chunks "
          f"(collection='{collection_name}', backend='{config.EMBEDDING_BACKEND}')")
    return collection


def main():
    print(f"Corpus dir : {config.CORPUS_DIR}")
    print(f"Chroma path: {config.CHROMA_PATH}")
    print(f"Backend    : {config.EMBEDDING_BACKEND}")

    if not os.path.exists(config.CORPUS_DIR):
        raise SystemExit(f"CORPUS_DIR does not exist: {config.CORPUS_DIR}")

    chunks = ingest_corpus(config.CORPUS_DIR)
    df = pd.DataFrame(chunks)[["chunk_id", "page_id", "space", "status", "heading_path"]]
    print(df.head(10).to_string(index=False))

    build_index(chunks)
    print("\nIndex build complete. You can now run: python -m app.main")


if __name__ == "__main__":
    main()
