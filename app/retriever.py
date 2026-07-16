"""
Retrieval layer: turns a natural-language query into a ranked list of
chunks from ChromaDB, annotated with freshness status so the generation
layer (and the user) can tell current vs. outdated/deprecated content apart.
"""
import chromadb

from app import config
from app.embeddings import Embedder


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(config.CHROMA_COLLECTION)
        self.embedder = Embedder.load_for_query()

    def retrieve(self, query: str, k: int = None, only_current: bool = False):
        k = k or config.TOP_K
        query_vec = self.embedder.embed_query(query)

        where = {"status": "current"} if only_current else None
        # Over-fetch a bit so that after de-duplicating near-identical chunks
        # from the same page we still end up with k useful, diverse results.
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=max(k * 2, k),
            where=where,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            md = results["metadatas"][0][i]
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "page_id": md["page_id"],
                "title": md["title"],
                "space": md["space"],
                "status": md["status"],
                "heading": md["heading"],
                "last_updated": md.get("last_updated", ""),
            })

        hits.sort(key=lambda h: h["distance"])
        return hits[:k]


if __name__ == "__main__":
    r = Retriever()
    for q in [
        "How many PTO days do I get after 3 years?",
        "What is our current deployment process?",
        "Is the Jenkins deployment process still in use?",
    ]:
        print(f"QUERY: {q}")
        for h in r.retrieve(q):
            flag = "  <-- OUTDATED/DEPRECATED" if h["status"] != "current" else ""
            print(f"  [{h['page_id']}] {h['title']} > {h['heading']} "
                  f"(status={h['status']}, dist={h['distance']:.3f}){flag}")
        print()
