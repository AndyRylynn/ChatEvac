"""
Reproducible Gwet's AC1 computation for the ChatEvac human verification subset.
Reads anchor_items_20.json and computes inter-rater reliability.

Usage: python inter_rater_analysis.py
"""
import json
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def gwet_ac1(ratings_matrix):
    """
    Compute Gwet's AC1 for a ratings matrix.

    Args:
        ratings_matrix: numpy array of shape (n_items, n_raters), values 1-5.

    Returns:
        ac1: Gwet's AC1 coefficient.
    """
    n_items, n_raters = ratings_matrix.shape
    n_categories = 5

    counts = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for r in range(n_raters):
            cat = int(ratings_matrix[i, r]) - 1  # 1-5 -> 0-4
            counts[i, cat] += 1

    # Category marginal proportions
    pi_k = np.sum(counts, axis=0) / (n_items * n_raters)
    pi_star = np.sum(pi_k * (1 - pi_k))

    # Agreement for each item
    A_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_a = np.mean(A_i)

    # Chance agreement under Gwet
    gamma_star = pi_star / (n_categories - 1)

    ac1 = (P_a - gamma_star) / (1 - gamma_star) if gamma_star < 1 else 1.0
    return ac1


def compute_from_anchors(anchor_path):
    """Load anchor items and compute AC1 for FSM and RAG separately."""
    with open(anchor_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    for subset_name, key in [('FSM', 'fsm_anchors'), ('RAG', 'rag_anchors')]:
        items = data[key]
        n_items = len(items)
        n_raters = 4

        # Build ratings matrix: items with all 4 ratings
        matrix = np.zeros((n_items, n_raters))
        for i, item in enumerate(items):
            ratings = item.get('ratings', {})
            for r_idx, rater_key in enumerate(['Rater_1', 'Rater_2', 'Rater_3', 'Rater_4']):
                val = ratings.get(rater_key)
                if val is not None and 1 <= val <= 5:
                    matrix[i, r_idx] = val
                else:
                    matrix[i, r_idx] = np.nan

        # Keep only items with complete ratings
        complete = ~np.isnan(matrix).any(axis=1)
        matrix_complete = matrix[complete]
        n_complete = matrix_complete.shape[0]

        if n_complete < 2:
            results[subset_name] = {
                'n_items': n_items,
                'n_complete': n_complete,
                'ac1': None,
                'warning': 'Insufficient complete ratings for AC1 computation.'
            }
        else:
            ac1 = gwet_ac1(matrix_complete)
            results[subset_name] = {
                'n_items': n_items,
                'n_complete': n_complete,
                'ac1': round(float(ac1), 4)
            }

    return results


def main():
    anchor_path = os.path.join(HERE, 'anchor_items_20.json')

    if not os.path.exists(anchor_path):
        print(f'Error: {anchor_path} not found.')
        print('Ensure anchor_items_20.json contains ratings from all 4 raters before running this script.')
        return

    results = compute_from_anchors(anchor_path)

    print('=' * 50)
    print("Gwet's AC1 Inter-Rater Reliability")
    print('=' * 50)
    for subset, res in results.items():
        print(f'\n{subset}:')
        print(f'  Anchor items: {res["n_items"]}')
        print(f'  Complete ratings: {res["n_complete"]}')
        if res['ac1'] is not None:
            print(f"  Gwet's AC1: {res['ac1']}")
            if res['ac1'] >= 0.80:
                print('  Interpretation: Almost perfect agreement (AC1 >= 0.80)')
            elif res['ac1'] >= 0.60:
                print('  Interpretation: Substantial agreement (0.60 <= AC1 < 0.80)')
            else:
                print('  Interpretation: Moderate or lower agreement (AC1 < 0.60)')
        else:
            print(f'  Warning: {res.get("warning", "N/A")}')
    print()


if __name__ == '__main__':
    main()
