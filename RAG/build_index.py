"""
One-time script to build FAISS index from code_provisions.json.
Uses OpenAI text-embedding-3-small to embed all provisions.
Run once before using RAG in Agent.py.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import CodeRAG, PROVISIONS_PATH, INDEX_PATH, EMBEDDINGS_PATH
from config import CHAT_API_KEY, CHAT_API_BASE

if __name__ == "__main__":
    print("=" * 50)
    print("Building Code RAG Index")
    print("=" * 50)

    rag = CodeRAG(
        api_key=CHAT_API_KEY,
        api_base=CHAT_API_BASE,
        provisions_path=PROVISIONS_PATH,
        index_path=INDEX_PATH,
        embeddings_path=EMBEDDINGS_PATH,
    )

    print(f"Provisions loaded: {len(rag.provisions)}")
    print(f"Embedding model: text-embedding-3-small")
    print(f"Output dim: 1536")

    rag.build_index(batch_size=100)

    print(f"\nIndex: {INDEX_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print("Done!")
