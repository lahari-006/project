"""
Embedding backend abstraction.

Two interchangeable backends so the project can run with zero heavy
dependencies (TF-IDF, exactly what the original notebook used) or with
much better semantic recall (sentence-transformers) if you have the
bandwidth to download a model.

Whichever backend is used at ingest time MUST be used at query time too,
which is why the fitted TF-IDF vectorizer is pickled to disk and the
sbert model name is fixed in config.
"""
import pickle
from pathlib import Path

from app import config


class Embedder:
    def __init__(self, backend: str = "tfidf"):
        self.backend = backend
        self._vectorizer = None
        self._sbert_model = None

        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME)
        elif backend == "tfidf":
            pass  # built lazily in fit() or load()
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    # -- build-time -----------------------------------------------------
    def fit(self, texts):
        """Fit (TF-IDF only) and persist the vectorizer. No-op for sbert."""
        if self.backend != "tfidf":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=4000, stop_words="english")
        self._vectorizer.fit(texts)
        Path(config.TFIDF_VECTORIZER_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.TFIDF_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self._vectorizer, f)

    def embed_documents(self, texts):
        if self.backend == "sbert":
            return self._sbert_model.encode(texts, show_progress_bar=False).tolist()
        # tfidf
        if self._vectorizer is None:
            self.fit(texts)
        return self._vectorizer.transform(texts).toarray().tolist()

    # -- query-time -------------------------------------------------------
    def _load_tfidf(self):
        if self._vectorizer is not None:
            return
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            self._vectorizer = pickle.load(f)

    def embed_query(self, text):
        if self.backend == "sbert":
            return self._sbert_model.encode([text], show_progress_bar=False).tolist()[0]
        self._load_tfidf()
        return self._vectorizer.transform([text]).toarray().tolist()[0]

    @classmethod
    def load_for_query(cls, backend: str = None):
        """Convenience constructor used by the retriever at query time."""
        backend = backend or config.EMBEDDING_BACKEND
        emb = cls(backend=backend)
        if backend == "tfidf":
            emb._load_tfidf()
        return emb
