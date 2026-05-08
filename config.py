# ============================================================
# ChatEvac API Configuration
# ============================================================
# Edit this file to configure API keys and endpoints before
# running any script in this project.
#
# For OpenAI-compatible endpoints, you may use:
#   - https://api.openai.com/v1          (OpenAI official)
#   - https://api.deepseek.com           (DeepSeek)
#   - Any other OpenAI-compatible endpoint
#
# IMPORTANT: Never commit this file with real API keys to git.
#
# ============================================================
# Script-to-API mapping overview:
#   Agent.py                         → CHAT_API (chat + RAG)
#   LLM agent.py                     → CHAT_API (chat only)
#   RAG/build_index.py               → CHAT_API (embedding only)
#   RAG/rag_engine.py                → receives key from caller
#   Validate/FSM/fsm_stress_test.py  → DEEPSEEK_API (chat only)
#   Validate/RAG/rag_stress_test.py  → CHAT_API + JUDGE_API
# ============================================================

import os
import sys

# ------------------------------------------------------------------
# CHAT_API — primary API key and endpoint
# ------------------------------------------------------------------
# Used by:
#   Agent.py                         — Chat (gpt-4o) + RAG embeddings
#   LLM agent.py                     — Chat (gpt-4o)
#   RAG/build_index.py               — Embedding (text-embedding-3-small)
#   Validate/RAG/rag_stress_test.py  — Answer generation
#
# Required capabilities:
#   - Chat Completions   (gpt-4o / gpt-4o-mini / equivalent)
#   - Embeddings         (text-embedding-3-small / equivalent)
# ------------------------------------------------------------------

CHAT_API_KEY = os.environ.get("CHATEVAC_CHAT_API_KEY", "your-chat-api-key-here")
CHAT_API_BASE = os.environ.get("CHATEVAC_CHAT_API_BASE", "https://api.openai.com/v1")

# Model selection — change to match your provider's available models
CHAT_MODEL = os.environ.get("CHATEVAC_CHAT_MODEL", "gpt-4o")
CONCLUSION_MODEL = os.environ.get("CHATEVAC_CONCLUSION_MODEL", "gpt-4o")
EMBEDDING_MODEL = "text-embedding-3-small"  # RAG embedding model (not user-configurable)

# ------------------------------------------------------------------
# DEEPSEEK_API — DeepSeek API key and endpoint
# ------------------------------------------------------------------
# Used by:
#   Validate/FSM/fsm_stress_test.py  — FSM transition accuracy tests
#
# Required capabilities:
#   - Chat Completions only
# ------------------------------------------------------------------

DEEPSEEK_API_KEY = os.environ.get("CHATEVAC_DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
DEEPSEEK_API_BASE = os.environ.get("CHATEVAC_DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("CHATEVAC_DEEPSEEK_MODEL", "deepseek-chat")

# ------------------------------------------------------------------
# JUDGE_API — LLM-as-Judge API key and endpoint
# ------------------------------------------------------------------
# Used by:
#   Validate/RAG/rag_stress_test.py  — Scoring RAG answer quality
#
# Required capabilities:
#   - Chat Completions only
# Can point to the same provider as CHAT_API to save cost.
# ------------------------------------------------------------------

JUDGE_API_KEY = os.environ.get("CHATEVAC_JUDGE_API_KEY", "your-judge-api-key-here")
JUDGE_API_BASE = os.environ.get("CHATEVAC_JUDGE_API_BASE", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("CHATEVAC_JUDGE_MODEL", "gpt-4o")

# ============================================================
# Quick-start checklist:
#   1. Set CHAT_API_KEY to your OpenAI / compatible API key
#   2. Set CHAT_API_BASE if using a non-OpenAI endpoint
#   3. (Optional) Set DEEPSEEK_API_KEY if running FSM stress tests
#   4. (Optional) Set JUDGE_API_KEY if running RAG stress tests
# ============================================================
