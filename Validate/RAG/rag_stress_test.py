"""
RAG Evaluation Stress Test — 1000-case independent evaluation.
Compares RAG-on (provisions injected) vs RAG-off (baseline, no provisions).
Uses LLM-as-Judge to score citation accuracy, numerical precision, and hallucination.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI

# Ensure project root is on path for RAG imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ==================== Config ====================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # ChatEvac/
TEST_DATA = os.path.join(HERE, "rag_test_cases_1k.json")
PROVISIONS_PATH = os.path.join(PROJECT_ROOT, "RAG", "code_provisions.json")
OUTPUT_DIR = HERE
REPORT_JSON = os.path.join(HERE, "rag_test_report.json")
SUMMARY_TXT = os.path.join(HERE, "rag_test_summary.txt")

# API configuration — edit config.py in the project root
from config import (
    CHAT_API_KEY, CHAT_API_BASE, CHAT_MODEL,
    JUDGE_API_KEY, JUDGE_API_BASE, JUDGE_MODEL,
)

# Answering API (GPT-4o — production model)
ANSWER_API_BASE = CHAT_API_BASE
ANSWER_API_KEY = CHAT_API_KEY
ANSWER_MODEL = CHAT_MODEL

# Judge API
JUDGE_API_BASE = JUDGE_API_BASE
JUDGE_API_KEY = JUDGE_API_KEY
JUDGE_MODEL = JUDGE_MODEL

# RAG embedding API (same endpoint as answer model)
RAG_API_BASE = CHAT_API_BASE
RAG_API_KEY = CHAT_API_KEY

MAX_WORKERS = 12
DEFAULT_TOP_K = 3

# ==================== RAG System Prompt ====================

RAG_SYSTEM_PROMPT = """You are an AI assistant specialized in building evacuation safety assessment.
You have access to relevant building code provisions that will be provided in a [RELEVANT CODE PROVISIONS] block.
Use these provisions to answer the user's question with precise numerical values and code references.
If the question falls outside the scope of the provided provisions, honestly state that you do not have the relevant code information.
Keep responses concise and professional."""

BASELINE_SYSTEM_PROMPT = """You are an AI assistant specialized in building evacuation safety assessment.
Answer the user's question based on your knowledge of building codes (NFPA 101, IBC).
Provide precise numerical values and code references where possible.
Keep responses concise and professional."""

# ==================== Judge Prompt ====================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a building code RAG system.
Your task is to score an AI response against a ground truth answer on specific criteria.
Output ONLY a valid JSON object. No markdown, no commentary."""


def build_judge_prompt(question, ground_truth, ai_response, is_in_scope, category):
    """Build the judge evaluation prompt."""
    if not is_in_scope:
        return f"""Evaluate this AI response for an OUT-OF-SCOPE question. The knowledge base does NOT cover this topic.

QUESTION: {question}

AI RESPONSE: {ai_response}

Score on these criteria (0 or 1):
- correctly_refused (1 = acknowledged knowledge gap, did not fabricate an answer; 0 = made up a false answer)

Output JSON:
{{"correctly_refused": 0 or 1, "brief_reason": "one short sentence"}}"""

    return f"""Evaluate this AI response for a {category} building code question.

QUESTION: {question}

GROUND TRUTH: {ground_truth}

AI RESPONSE: {ai_response}

Score on these criteria (0 or 1):
- citation_correct (1 = cited the correct code [NFPA 101 or IBC], AND the regulation is approximately correct; 0 = wrong code, fabricated section, or missing citation)
- numerical_match (1 = key numerical values match ground truth within 1% tolerance; 0 = wrong number, missing number, or vague number)
- no_hallucination (1 = no fabricated code section numbers or entirely invented values; 0 = contains hallucinated references)

Note: If the AI response gives the correct numerical value but says "approximately" or rounds slightly (e.g. 32\" vs \"not less than 32 inches\"), count numerical_match as 1.

Output JSON:
{{"citation_correct": 0 or 1, "numerical_match": 0 or 1, "no_hallucination": 0 or 1, "brief_reason": "one short sentence"}}"""


# ==================== Core Functions ====================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(api_base, api_key, model, system_prompt, user_message, max_tokens=512, temperature=0.0):
    """Generic LLM call."""
    client = OpenAI(base_url=api_base, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"__ERROR__: {e}"


# ==================== RAG Retrieval (reuses CodeRAG from production) ====================

from RAG.rag_engine import CodeRAG

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = CodeRAG(api_key=RAG_API_KEY, api_base=RAG_API_BASE)
    return _rag_instance


# ==================== Single Test Runner ====================

def run_single_test(case):
    """Run one test case in both RAG-on and RAG-off modes, then judge both."""
    question = case["question"]
    is_in_scope = case.get("is_in_scope", True)
    category = case["category"]

    # --- RAG-on ---
    rag = get_rag()
    rag_block, _ = rag.retrieve_and_format(question, top_k=DEFAULT_TOP_K) if is_in_scope else ("", [])

    rag_prompt = RAG_SYSTEM_PROMPT + (rag_block if rag_block else "")
    rag_response = call_llm(ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                            rag_prompt, question)

    # --- RAG-off (Baseline) ---
    baseline_response = call_llm(ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                                 BASELINE_SYSTEM_PROMPT, question)

    # --- Judge both ---
    judge_prompt_rag = build_judge_prompt(question, case["ground_truth"], rag_response, is_in_scope, category)
    judge_prompt_base = build_judge_prompt(question, case["ground_truth"], baseline_response, is_in_scope, category)

    judge_raw_rag = call_llm(JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL,
                             JUDGE_SYSTEM_PROMPT, judge_prompt_rag, max_tokens=256)
    judge_raw_base = call_llm(JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL,
                              JUDGE_SYSTEM_PROMPT, judge_prompt_base, max_tokens=256)

    rag_scores = parse_judge_output(judge_raw_rag, is_in_scope)
    base_scores = parse_judge_output(judge_raw_base, is_in_scope)

    return {
        "case_id": case["id"],
        "question": question[:200],
        "category": category,
        "is_in_scope": is_in_scope,
        "ground_truth": case["ground_truth"][:200],
        "rag_response": rag_response[:500],
        "baseline_response": baseline_response[:500],
        "rag_scores": rag_scores,
        "baseline_scores": base_scores,
    }


def parse_judge_output(raw, is_in_scope):
    """Parse LLM judge output, with fallback defaults."""
    try:
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        scores = json.loads(cleaned.strip())
        return scores
    except (json.JSONDecodeError, AttributeError):
        if is_in_scope:
            return {"citation_correct": 0, "numerical_match": 0, "no_hallucination": 0, "brief_reason": "parse error"}
        return {"correctly_refused": 0, "brief_reason": "parse error"}


# ==================== Statistics ====================

def compute_stats(results, mode="rag"):
    """Compute aggregated statistics."""
    prefix = "rag" if mode == "rag" else "baseline"
    scores_key = f"{prefix}_scores"

    total = len(results)
    in_scope_results = [r for r in results if r["is_in_scope"]]
    out_scope_results = [r for r in results if not r["is_in_scope"]]

    stats = {"total": total}

    if in_scope_results:
        n = len(in_scope_results)
        citation_ok = sum(1 for r in in_scope_results if r[scores_key].get("citation_correct", 0) == 1)
        numerical_ok = sum(1 for r in in_scope_results if r[scores_key].get("numerical_match", 0) == 1)
        no_halluc = sum(1 for r in in_scope_results if r[scores_key].get("no_hallucination", 0) == 1)

        stats["in_scope"] = {
            "total": n,
            "citation_accuracy": round(citation_ok / n * 100, 2),
            "numerical_precision": round(numerical_ok / n * 100, 2),
            "no_hallucination_rate": round(no_halluc / n * 100, 2),
            "citation_ok": citation_ok,
            "numerical_ok": numerical_ok,
            "no_halluc_ok": no_halluc,
        }

        # Overall score (all three criteria met)
        all_ok = sum(1 for r in in_scope_results
                     if r[scores_key].get("citation_correct", 0) == 1
                     and r[scores_key].get("numerical_match", 0) == 1
                     and r[scores_key].get("no_hallucination", 0) == 1)
        stats["in_scope"]["all_criteria_met"] = round(all_ok / n * 100, 2)

    if out_scope_results:
        n = len(out_scope_results)
        refused = sum(1 for r in out_scope_results if r[scores_key].get("correctly_refused", 0) == 1)
        stats["out_of_scope"] = {
            "total": n,
            "correct_refusal_rate": round(refused / n * 100, 2),
            "refused": refused,
        }

    # By category
    by_cat = defaultdict(lambda: {"total": 0, "citation_ok": 0, "numerical_ok": 0, "no_halluc_ok": 0})
    for r in in_scope_results:
        cat = r["category"]
        by_cat[cat]["total"] += 1
        if r[scores_key].get("citation_correct", 0) == 1:
            by_cat[cat]["citation_ok"] += 1
        if r[scores_key].get("numerical_match", 0) == 1:
            by_cat[cat]["numerical_ok"] += 1
        if r[scores_key].get("no_hallucination", 0) == 1:
            by_cat[cat]["no_halluc_ok"] += 1

    stats["by_category"] = {}
    for cat, v in sorted(by_cat.items()):
        t = v["total"]
        stats["by_category"][cat] = {
            "total": t,
            "citation_accuracy": round(v["citation_ok"] / t * 100, 2) if t > 0 else 0,
            "numerical_precision": round(v["numerical_ok"] / t * 100, 2) if t > 0 else 0,
            "no_hallucination_rate": round(v["no_halluc_ok"] / t * 100, 2) if t > 0 else 0,
        }

    return stats


def generate_summary(rag_stats, base_stats):
    lines = []
    lines.append("=" * 70)
    lines.append("ChatEvac RAG Evaluation Report — RAG-on vs Baseline")
    lines.append(f"Answer Model: {ANSWER_MODEL} | Judge Model: {JUDGE_MODEL} | {rag_stats['total']} test cases")
    lines.append("=" * 70)

    for label, stats, mode in [("RAG-ON", rag_stats, "rag"), ("BASELINE", base_stats, "baseline")]:
        lines.append(f"\n--- {label} ---")
        if "in_scope" in stats:
            s = stats["in_scope"]
            lines.append(f"  In-scope cases: {s['total']}")
            lines.append(f"  Citation Accuracy:     {s['citation_accuracy']}% ({s['citation_ok']}/{s['total']})")
            lines.append(f"  Numerical Precision:   {s['numerical_precision']}% ({s['numerical_ok']}/{s['total']})")
            lines.append(f"  No-Hallucination Rate: {s['no_hallucination_rate']}% ({s['no_halluc_ok']}/{s['total']})")
            lines.append(f"  All Criteria Met:      {s['all_criteria_met']}%")
        if "out_of_scope" in stats:
            lines.append(f"  Out-of-scope cases: {stats['out_of_scope']['total']}")
            lines.append(f"  Correct Refusal Rate:  {stats['out_of_scope']['correct_refusal_rate']}%")

        lines.append(f"\n  By Category:")
        for cat, v in stats.get("by_category", {}).items():
            lines.append(f"    {cat:20s}: Citation {v['citation_accuracy']:.1f}% | "
                         f"Numerical {v['numerical_precision']:.1f}% | "
                         f"No-Halluc {v['no_hallucination_rate']:.1f}%")

    # Comparison
    lines.append(f"\n{'=' * 70}")
    lines.append("COMPARISON (RAG-on — Baseline)")
    lines.append("=" * 70)
    for metric in ["citation_accuracy", "numerical_precision", "no_hallucination_rate"]:
        rag_val = rag_stats.get("in_scope", {}).get(metric, 0)
        base_val = base_stats.get("in_scope", {}).get(metric, 0)
        delta = round(rag_val - base_val, 2)
        lines.append(f"  {metric}: RAG {rag_val}% vs Baseline {base_val}% (Δ {delta:+.1f} pp)")

    if "out_of_scope" in rag_stats and "out_of_scope" in base_stats:
        rag_ref = rag_stats["out_of_scope"].get("correct_refusal_rate", 0)
        base_ref = base_stats["out_of_scope"].get("correct_refusal_rate", 0)
        lines.append(f"  correct_refusal_rate: RAG {rag_ref}% vs Baseline {base_ref}% (Δ {round(rag_ref - base_ref, 1):+.1f} pp)")

    lines.append("=" * 70)
    return "\n".join(lines)


# ==================== Main ====================

def main():
    print("=" * 60)
    print("RAG Evaluation Stress Test")
    print(f"Answer: {ANSWER_MODEL} | Judge: {JUDGE_MODEL} | Workers: {MAX_WORKERS}")
    print("=" * 60)

    # Load data
    print("\nLoading test cases...")
    test_cases = load_json(TEST_DATA)
    print(f"Test cases: {len(test_cases)}")

    # Init RAG engine (loads FAISS index)
    print("Initializing RAG engine...")
    rag = get_rag()
    print(f"RAG ready: {rag.is_ready}")

    # Run tests
    print(f"\nTesting {len(test_cases)} cases "
          f"({len(test_cases) * 2} answer calls + {len(test_cases) * 2} judge calls)...")
    start = time.time()
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for case in test_cases:
            f = executor.submit(run_single_test, case)
            futures[f] = case["id"]

        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 200 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    eta = (len(test_cases) - completed) / rate if rate > 0 else 0
                    print(f"  {completed}/{len(test_cases)} — {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
            except Exception as e:
                print(f"  Error [{cid}]: {e}")
                completed += 1

    elapsed = time.time() - start
    print(f"  Done. {elapsed:.0f}s total")

    # Compute stats
    rag_stats = compute_stats(results, "rag")
    base_stats = compute_stats(results, "baseline")

    # Save detailed report
    report = {
        "metadata": {
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "total_cases": len(test_cases),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rag_mode": rag_stats,
        "baseline_mode": base_stats,
        "sample_results": results[:50],  # First 50 for inspection
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Generate and save summary
    summary = generate_summary(rag_stats, base_stats)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{summary}")
    print(f"\nReports saved:")
    print(f"  {REPORT_JSON}")
    print(f"  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
