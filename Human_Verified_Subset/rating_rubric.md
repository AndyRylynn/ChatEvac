# Rating Rubric for Human Verification

## FSM Dataset: Expected Symbol Consistency

| Level | Label | Definition |
|-------|-------|------------|
| 5 | Fully consistent | The assigned symbol is the uniquely correct transition; no plausible alternative exists. |
| 4 | Largely consistent | The symbol correctly captures the user intent; a minor alternative could be argued. |
| 3 | Adequate | The symbol is a legally valid transition, but ambiguity exists (another symbol is also reasonable). |
| 2 | Partially inconsistent | The symbol is technically permitted but clearly suboptimal for the given context. |
| 1 | Inconsistent | The symbol is either structurally invalid or semantically unrelated to the user intent. |

## RAG Dataset: Ground Truth Factual Consistency

| Level | Label | Definition |
|-------|-------|------------|
| 5 | Fully consistent | Numerical values, code reference, and phrasing exactly match the source provision. |
| 4 | Largely consistent | Values and reference correct; minor stylistic deviation from provision wording. |
| 3 | Adequate | Values correct, but key qualifiers or conditions from the source provision are omitted. |
| 2 | Partially inconsistent | A key numerical value or the code reference (NFPA 101 vs. IBC) is incorrect. |
| 1 | Inconsistent | Both numerical values and code reference contradict the source provision. |

## Notes

- For FSM cases, refer to the workflow state machine specification (workflow_config.json) when judging transition validity.
- For RAG cases, refer to the source code provisions (code_provisions.json) when judging factual accuracy.
- Ratings below 5 should be driven by observable issues in the test data, not by personal preference for alternative phrasing.
