# Human-Verified Subset for ChatEvac Evaluation Data

This directory contains the stratified random samples used for the independent human audit described in Appendix B of the manuscript.

## Contents

| File | Description |
|------|-------------|
| `fsm_sample_200.json` | 200 FSM dialogue turns (stratified by 5 intent types), with rating slots for 4 raters |
| `rag_sample_200.json` | 200 RAG questions (stratified by 5 question categories), with rating slots for 4 raters |
| `anchor_items_20.json` | 20 anchor items (10 FSM + 10 RAG) rated by all 4 raters for inter-rater reliability |
| `rating_rubric.md` | Five-level rating scale definitions for both FSM and RAG evaluation |
| `inter_rater_analysis.py` | Reproducible script that reads `anchor_items_20.json` and computes Gwet's AC1 |

## Usage

Each JSON file contains the original test cases plus a `ratings` field with slots for four independent raters (Rater_1 through Rater_4). Ratings use the 1-5 scale defined in `rating_rubric.md`. The anchor items are a subset of the 200-sample files and were rated by all four raters to compute Gwet's AC1.

After all four raters have completed their ratings and the scores have been filled into `anchor_items_20.json`, run the analysis script to obtain the inter-rater reliability coefficient:

```bash
python inter_rater_analysis.py
```

The script computes Gwet's AC1 separately for the FSM and RAG anchor subsets and reports the interpretation against the conventional 0.80 threshold for almost perfect agreement.

## Citation

If you use this dataset, please cite the corresponding manuscript.
