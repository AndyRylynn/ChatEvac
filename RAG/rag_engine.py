"""
ChatEvac Code RAG Engine
Uses OpenAI text-embedding-3-small for embeddings and FAISS for vector search.
Provides retrieval of relevant building code provisions for LLM prompt injection.
"""

import json
import os
import numpy as np
import faiss
from openai import OpenAI

# ==================== Config ====================
HERE = os.path.dirname(os.path.abspath(__file__))
PROVISIONS_PATH = os.path.join(HERE, "code_provisions.json")
INDEX_PATH = os.path.join(HERE, "code_index.faiss")
EMBEDDINGS_PATH = os.path.join(HERE, "code_embeddings.npy")
QUERY_CACHE_PATH = os.path.join(HERE, "query_cache.json")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DEFAULT_TOP_K = 3

# ==================== Engine ====================


class CodeRAG:
    """Retrieval engine for building code provisions."""

    def __init__(
        self,
        api_key,
        api_base="https://api.openai.com/v1",
        provisions_path=PROVISIONS_PATH,
        index_path=INDEX_PATH,
        embeddings_path=EMBEDDINGS_PATH,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.index_path = index_path
        self.embeddings_path = embeddings_path

        # Load provisions
        with open(provisions_path, "r", encoding="utf-8") as f:
            self.provisions = json.load(f)

        # Load or initialize index
        self.index = None
        self.corpus_embeddings = None
        self._ready = False

        if os.path.exists(index_path) and os.path.exists(embeddings_path):
            self.index = faiss.read_index(index_path)
            self.corpus_embeddings = np.load(embeddings_path)
            self._ready = True

        # Query cache to avoid re-embedding identical queries
        self._query_cache = {}
        if os.path.exists(QUERY_CACHE_PATH):
            with open(QUERY_CACHE_PATH, "r", encoding="utf-8") as f:
                self._query_cache = json.load(f)

    @property
    def is_ready(self):
        return self._ready

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, batch_size=100):
        """Embed all provisions and build FAISS index. Call once offline."""
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        texts = [p["output"] for p in self.provisions]
        n = len(texts)
        embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        print(f"Embedding {n} provisions in batches of {batch_size}...")
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            for j, item in enumerate(resp.data):
                embeddings[i + j] = np.array(item.embedding, dtype=np.float32)
            print(f"  {min(i + batch_size, n)}/{n} done")

        # Normalize for cosine similarity (inner product)
        faiss.normalize_L2(embeddings)

        # Build index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine on normalized vectors
        self.index.add(embeddings)
        self.corpus_embeddings = embeddings

        # Persist
        faiss.write_index(self.index, self.index_path)
        np.save(self.embeddings_path, embeddings)
        self._ready = True
        print(f"Index built and saved: {self.index_path}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _embed_query(self, query):
        """Embed a single query string via OpenAI API."""
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def retrieve(self, query, top_k=DEFAULT_TOP_K):
        """
        Retrieve top-k relevant code provisions for a query.
        Returns list of provision dicts with added 'similarity' field.
        """
        if not self._ready:
            return []

        # Check cache first
        cache_key = query.strip().lower()[:200]
        if cache_key in self._query_cache:
            cached_ids = self._query_cache[cache_key]
            results = []
            for cid in cached_ids[:top_k]:
                for p in self.provisions:
                    if p.get("id") == cid:
                        results.append({**p, "similarity": 1.0})
                        break
            return results

        # Embed query
        q_vec = self._embed_query(query).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        # Search
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.provisions):
                prov = dict(self.provisions[idx])
                prov["similarity"] = round(float(score), 4)
                results.append(prov)

        # Update cache (store top-10 ids)
        self._query_cache[cache_key] = [p["id"] for p in results[:10]]
        if len(self._query_cache) > 1000:
            # Prune oldest entries
            keys = list(self._query_cache.keys())
            for k in keys[:200]:
                del self._query_cache[k]
            self._save_cache()

        return results

    def _save_cache(self):
        with open(QUERY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._query_cache, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, provisions):
        """Format retrieved provisions as a compact prompt injection block."""
        if not provisions:
            return ""

        lines = ["", "[RELEVANT CODE PROVISIONS — Use these to ground your response:]"]
        for i, p in enumerate(provisions):
            source = "NFPA 101" if "NFPA 101" in p.get("output", "") else "IBC"
            lines.append(f"  [{source}] {p['output']}")
        lines.append("[END CODE PROVISIONS]")
        return "\n".join(lines)

    def retrieve_and_format(self, query, top_k=DEFAULT_TOP_K):
        """Convenience: retrieve + format in one call."""
        provisions = self.retrieve(query, top_k=top_k)
        return self.format_for_prompt(provisions), provisions


# ==================== Singleton helper ====================

_rag_instance = None


def get_rag(api_key, api_base="https://api.openai.com/v1"):
    """Get or create the singleton CodeRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = CodeRAG(api_key=api_key, api_base=api_base)
    return _rag_instance
