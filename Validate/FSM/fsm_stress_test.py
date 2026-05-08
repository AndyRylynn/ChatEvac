"""
ChatEvac FSM State Machine Automated Transition Accuracy Test Script
- FSM mode: Inject state machine context, test LLM symbol output accuracy
- Baseline mode: No state machine context, pure intent classification comparison
- Sequence mode: Simulate full session with continuous state propagation
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI

# Ensure project root is on path for StateMachine imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ==================== Configuration ====================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # ChatEvac/
CONFIG_PATH = os.path.join(PROJECT_ROOT, "StateMachine", "workflow_config.json")
TEST_DATA_PATH = os.path.join(HERE, "test_cases_5k.json")
REPORT_PATH = os.path.join(HERE, "test_report.json")
SUMMARY_PATH = os.path.join(HERE, "test_summary.txt")

# API configuration — edit config.py in the project root
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

API_BASE = DEEPSEEK_API_BASE
API_KEY = DEEPSEEK_API_KEY
MODEL = DEEPSEEK_MODEL
MAX_WORKERS = 20

# ==================== System Prompts ====================

FSM_BASE_PROMPT = """You are an AI assistant specialized in building evacuation safety assessment. You help users conduct simulation evaluations of architectural floor plans through a structured workflow.

WORKFLOW CONTROL MECHANISM:
- The workflow is managed by a state machine. Each state has a single-letter symbol (A through O).
- At the end of each turn, you will receive a [WORKFLOW STATE CONTEXT] block describing your current state and available transitions.
- When the user's intent matches a transition condition, include the corresponding tag in your response: <WORKFLOW_ACTION>SYMBOL</WORKFLOW_ACTION>
- Use ONLY the single-letter symbols listed in the context block. Do not invent new symbols or use old action names.
- Evaluate user intent SEMANTICALLY — do not rely on exact keyword matching. Understand what the user means, not just what they literally say.
- You may include at most ONE workflow action tag per response.

PARAMETER CONFIGURATION:
- Supported parameters: space width (m), space height (m), number of people, max speed factor, pedestrian radius (m), delta time (s), people mass (kg), people-people repulsion, people-wall repulsion.
- Default values: width=15m, height=15m, people=30, speed=1.6, radius=0.35m, dt=0.1s, mass=100kg, repulsion=100, wall-repulsion=100.
- When the user provides numbers, extract and apply them. When no numbers are given, proceed with defaults.

GENERAL GUIDELINES:
- Keep responses concise and professional.
- Focus on evacuation safety assessment tasks.
- After the conclusion is generated, answer any follow-up questions about the analysis."""

BASELINE_PROMPT = """You are an evacuation safety assessment assistant. Your task is to classify the user's intent into exactly one category.

Available intent categories (output the corresponding letter):
  D — User wants to extract features from a floor plan, analyze a building layout, or restart the whole process.
  G — User wants to run an evacuation simulation with given or default parameters.
  J — User wants to run data analysis on simulation results, generate heatmaps or visualizations.
  M — User wants to generate an expert conclusion report based on all analysis results.
  Z — User is off-topic: asking about weather, cooking, sports, movies, programming, firewall(network), or anything unrelated to evacuation safety assessment.
  B — User is providing a floor plan image for the first time and requesting evacuation assessment.
  E — User wants to configure or confirm simulation parameters.
  I — User confirms they want to proceed with data analysis after simulation.
  L — User confirms they want to generate a conclusion report after data analysis.
  O — User indicates the workflow is complete or they are satisfied with the results.

INSTRUCTIONS:
- Analyze the user's message and determine which single intent best matches.
- Output ONLY: <WORKFLOW_ACTION>X</WORKFLOW_ACTION> where X is one of the letters above.
- Do NOT output any other text. Just the tag."""


# ==================== Helpers ====================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_symbol(text):
    """Extract <WORKFLOW_ACTION>X</WORKFLOW_ACTION> from LLM output"""
    match = re.search(r'<WORKFLOW_ACTION>\s*([A-Za-z])\s*</WORKFLOW_ACTION>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: look for any single capital letter near "action" or "symbol"
    match2 = re.search(r'(?:action|symbol)\s*(?:is|:)?\s*["\']?([A-Za-z])', text, re.IGNORECASE)
    if match2:
        return match2.group(1).upper()
    # Last resort: find any isolated capital letter A-Z in the response
    match3 = re.findall(r'\b([A-OZ])\b', text)
    if match3:
        return match3[-1].upper()
    return None


def call_llm(system_prompt, user_message, max_tokens=256, temperature=0.0):
    """Call LLM and return text response"""
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"__ERROR__: {e}"


# ==================== FSM Mode ====================

def test_fsm_single(case, workflow_mgr):
    """FSM mode: Inject state machine context, test a single case"""
    # Force transition to the test case's specified state
    workflow_mgr.force_transition_to(case["current_state"])
    prompt_injection = workflow_mgr.get_prompt_injection(user_text=case["user_input"])
    full_prompt = FSM_BASE_PROMPT + prompt_injection

    llm_output = call_llm(full_prompt, case["user_input"])
    actual_symbol = parse_symbol(llm_output)
    expected = case["expected_symbol"]
    success = (actual_symbol == expected)

    return {
        "case_id": case["case_id"],
        "current_state": case["current_state"],
        "user_input": case["user_input"],
        "expected_symbol": expected,
        "actual_symbol": actual_symbol,
        "intent_type": case["intent_type"],
        "success": success,
        "mode": "FSM"
    }


# ==================== Baseline Mode ====================

def test_baseline_single(case):
    """Baseline mode: No state machine context, pure intent classification"""
    llm_output = call_llm(BASELINE_PROMPT, case["user_input"])
    actual_symbol = parse_symbol(llm_output)
    expected = case["expected_symbol"]
    success = (actual_symbol == expected)

    return {
        "case_id": case["case_id"],
        "current_state": case["current_state"],
        "user_input": case["user_input"],
        "expected_symbol": expected,
        "actual_symbol": actual_symbol,
        "intent_type": case["intent_type"],
        "success": success,
        "mode": "Baseline"
    }


# ==================== Sequence Mode ====================

def test_sequence_golden_path(workflow_mgr, all_cases):
    """
    Sequence mode: Simulate a full session (idle -> completed).
    Select cases matching the current state from the test pool; LLM output symbols drive state transitions.
    """
    # Index test cases by current_state
    cases_by_state = defaultdict(list)
    for c in all_cases:
        cases_by_state[c["current_state"]].append(c)

    golden_path = [
        "idle",
        "processing_initial",
        "running_feature_extraction",
        "awaiting_parameter_config",
        "running_simulation",
        "awaiting_data_analysis",
        "running_data_analysis",
        "awaiting_conclusion",
        "generating_conclusion",
        "conclusion_ready",
    ]

    results = []
    workflow_mgr.force_transition_to("idle")
    case_idx = 0

    for expected_state in golden_path:
        current = workflow_mgr.current_state
        candidates = cases_by_state.get(current, [])
        if not candidates:
            print(f"  [Sequence] No cases for state {current}, skipping...")
            continue

        # Select Normal-type cases for this state
        normals = [c for c in candidates if c["intent_type"] == "Normal"]
        picks = normals[:3] if normals else candidates[:3]

        for pick in picks:
            case_idx += 1
            prompt_injection = workflow_mgr.get_prompt_injection(user_text=pick["user_input"])
            full_prompt = FSM_BASE_PROMPT + prompt_injection
            llm_output = call_llm(full_prompt, pick["user_input"])
            actual_symbol = parse_symbol(llm_output)
            expected = pick["expected_symbol"]
            success = (actual_symbol == expected)

            results.append({
                "case_id": f"SEQ_{case_idx:04d}",
                "current_state": current,
                "user_input": pick["user_input"],
                "expected_symbol": expected,
                "actual_symbol": actual_symbol,
                "intent_type": pick["intent_type"],
                "success": success,
                "mode": "Sequence"
            })

            # Drive state transition using the symbol actually output by the LLM
            if actual_symbol:
                target = workflow_mgr.get_state_by_symbol(actual_symbol)
                if target:
                    workflow_mgr.force_transition_to(target)
                elif actual_symbol == "Z":
                    pass  # Z leaves state unchanged
            break  # Only take one case per state to advance

    return results


# ==================== Statistics ====================

def compute_stats(results):
    """Compute statistics"""
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    accuracy = success_count / total * 100 if total > 0 else 0

    # Count failures by state
    failure_by_state = defaultdict(lambda: {"total": 0, "failures": 0})
    for r in results:
        st = r["current_state"]
        failure_by_state[st]["total"] += 1
        if not r["success"]:
            failure_by_state[st]["failures"] += 1

    # Count by intent_type
    by_intent = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        it = r["intent_type"]
        by_intent[it]["total"] += 1
        if r["success"]:
            by_intent[it]["success"] += 1

    # Confusion matrix: expected -> actual
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["expected_symbol"]][r["actual_symbol"] or "?"] += 1

    return {
        "total": total,
        "success_count": success_count,
        "accuracy": round(accuracy, 2),
        "failure_by_state": {s: {"total": v["total"], "failures": v["failures"],
                                  "failure_rate": round(v["failures"] / v["total"] * 100, 2) if v["total"] > 0 else 0}
                             for s, v in sorted(failure_by_state.items())},
        "by_intent": {i: {"total": v["total"], "success": v["success"],
                          "accuracy": round(v["success"] / v["total"] * 100, 2) if v["total"] > 0 else 0}
                      for i, v in sorted(by_intent.items())},
        "confusion_matrix": {s: dict(v) for s, v in sorted(confusion.items())}
    }


# ==================== Report Generation ====================

def generate_summary(fsm_stats, baseline_stats, sequence_stats):
    """Generate human-readable test summary text"""
    lines = []
    lines.append("=" * 70)
    lines.append("ChatEvac FSM Stress Test Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {MODEL}")
    lines.append("=" * 70)

    # FSM Mode
    lines.append("\n" + "-" * 50)
    lines.append("[FSM MODE] With State Machine Context Injection")
    lines.append("-" * 50)
    lines.append(f"  Total Tests:    {fsm_stats['total']}")
    lines.append(f"  Success Count:  {fsm_stats['success_count']}")
    lines.append(f"  Accuracy:       {fsm_stats['accuracy']}%")

    lines.append("\n  Failure by State:")
    for state, info in fsm_stats["failure_by_state"].items():
        if info["failures"] > 0:
            lines.append(f"    {state:35s}: {info['failures']:4d}/{info['total']:4d} failures ({info['failure_rate']:.1f}%)")

    lines.append("\n  Accuracy by Intent Type:")
    for intent, info in fsm_stats["by_intent"].items():
        lines.append(f"    {intent:15s}: {info['accuracy']:.1f}% ({info['success']}/{info['total']})")

    # Baseline Mode
    lines.append("\n" + "-" * 50)
    lines.append("[BASELINE MODE] Without State Machine Context (Pure Intent)")
    lines.append("-" * 50)
    lines.append(f"  Total Tests:    {baseline_stats['total']}")
    lines.append(f"  Success Count:  {baseline_stats['success_count']}")
    lines.append(f"  Accuracy:       {baseline_stats['accuracy']}%")

    lines.append("\n  Failure by State:")
    for state, info in baseline_stats["failure_by_state"].items():
        if info["failures"] > 0:
            lines.append(f"    {state:35s}: {info['failures']:4d}/{info['total']:4d} failures ({info['failure_rate']:.1f}%)")

    lines.append("\n  Accuracy by Intent Type:")
    for intent, info in baseline_stats["by_intent"].items():
        lines.append(f"    {intent:15s}: {info['accuracy']:.1f}% ({info['success']}/{info['total']})")

    # Sequence Mode
    if sequence_stats:
        lines.append("\n" + "-" * 50)
        lines.append("[SEQUENCE MODE] Golden Path Continuous Session")
        lines.append("-" * 50)
        lines.append(f"  Total Steps:    {sequence_stats['total']}")
        lines.append(f"  Success Count:  {sequence_stats['success_count']}")
        lines.append(f"  Accuracy:       {sequence_stats['accuracy']}%")

    # Comparison
    lines.append("\n" + "=" * 70)
    lines.append("COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  FSM Mode Accuracy:      {fsm_stats['accuracy']}%")
    lines.append(f"  Baseline Accuracy:      {baseline_stats['accuracy']}%")
    delta = fsm_stats['accuracy'] - baseline_stats['accuracy']
    lines.append(f"  Improvement from FSM:   {delta:+.1f}%")
    lines.append(f"\n  Interpretation: The state machine context injection {'improves' if delta > 0 else 'degrades'} accuracy by {abs(delta):.1f} percentage points.")
    lines.append("=" * 70)

    return "\n".join(lines)


# ==================== Main ====================

def run_parallel_tests(test_func, cases, workflow_mgr=None, desc="Testing", max_workers=MAX_WORKERS):
    """Run tests in parallel"""
    results = []
    total = len(cases)
    completed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, case in enumerate(cases):
            if workflow_mgr:
                # Each thread needs an independent workflow_mgr instance to avoid races
                from StateMachine.workflow_engine import EvacWorkflowManager
                local_mgr = EvacWorkflowManager(CONFIG_PATH, verbose=False)
                future = executor.submit(test_func, case, local_mgr)
            else:
                future = executor.submit(test_func, case)
            futures[future] = i

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{desc}] {completed}/{total} ({completed/total*100:.1f}%) "
                          f"— {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
            except Exception as e:
                completed += 1
                print(f"  [{desc}] Error at index {futures[future]}: {e}")

    return results


def main():
    print("=" * 60)
    print("ChatEvac FSM Stress Test Runner")
    print(f"Model: {MODEL} | Workers: {MAX_WORKERS}")
    print("=" * 60)

    # Load test data
    print("\nLoading test cases...")
    test_cases = load_json(TEST_DATA_PATH)
    print(f"Loaded {len(test_cases)} test cases")

    # ---- FSM Mode ----
    print(f"\n{'=' * 60}")
    print("PHASE 1: FSM Mode (with state machine context)")
    print(f"{'=' * 60}")
    fsm_results = run_parallel_tests(
        test_fsm_single, test_cases,
        workflow_mgr=True,
        desc="FSM"
    )
    fsm_stats = compute_stats(fsm_results)
    print(f"\nFSM Mode Complete: {fsm_stats['accuracy']}% accuracy "
          f"({fsm_stats['success_count']}/{fsm_stats['total']})")

    # ---- Baseline Mode ----
    print(f"\n{'=' * 60}")
    print("PHASE 2: Baseline Mode (no state machine context)")
    print(f"{'=' * 60}")
    baseline_results = run_parallel_tests(
        test_baseline_single, test_cases,
        desc="Baseline"
    )
    baseline_stats = compute_stats(baseline_results)
    print(f"\nBaseline Mode Complete: {baseline_stats['accuracy']}% accuracy "
          f"({baseline_stats['success_count']}/{baseline_stats['total']})")

    # ---- Sequence Mode ----
    print(f"\n{'=' * 60}")
    print("PHASE 3: Sequence Mode (golden path continuous session)")
    print(f"{'=' * 60}")
    from StateMachine.workflow_engine import EvacWorkflowManager
    seq_mgr = EvacWorkflowManager(CONFIG_PATH, verbose=False)
    seq_results = test_sequence_golden_path(seq_mgr, test_cases)
    seq_stats = compute_stats(seq_results) if seq_results else None
    if seq_stats:
        print(f"Sequence Mode Complete: {seq_stats['accuracy']}% accuracy "
              f"({seq_stats['success_count']}/{seq_stats['total']})")

    # ---- Save Reports ----
    print(f"\n{'=' * 60}")
    print("Saving reports...")

    report = {
        "metadata": {
            "model": MODEL,
            "test_data": TEST_DATA_PATH,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_cases": len(test_cases)
        },
        "fsm_mode": fsm_stats,
        "baseline_mode": baseline_stats,
        "sequence_mode": seq_stats,
        "comparison": {
            "fsm_accuracy": fsm_stats["accuracy"],
            "baseline_accuracy": baseline_stats["accuracy"],
            "delta": round(fsm_stats["accuracy"] - baseline_stats["accuracy"], 2),
            "interpretation": (
                "State machine context injection improves LLM workflow adherence "
                f"by {abs(round(fsm_stats['accuracy'] - baseline_stats['accuracy'], 2))} percentage points."
                if fsm_stats["accuracy"] > baseline_stats["accuracy"]
                else "State machine context did not improve accuracy in this test run."
            )
        }
    }

    # Detailed results (first 200 failures from each mode for analysis)
    fsm_failures = [r for r in fsm_results if not r["success"]][:200]
    baseline_failures = [r for r in baseline_results if not r["success"]][:200]
    report["fsm_mode"]["sample_failures"] = fsm_failures
    report["baseline_mode"]["sample_failures"] = baseline_failures

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to: {REPORT_PATH}")

    # Summary text
    summary_text = generate_summary(fsm_stats, baseline_stats, seq_stats)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary saved to: {SUMMARY_PATH}")

    print(f"\n{summary_text}")
    print("\nDone!")


if __name__ == "__main__":
    main()
