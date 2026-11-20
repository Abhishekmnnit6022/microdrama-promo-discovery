"""
Evaluation metrics for hook quality.

Metrics:
  precision_at_k, recall_at_k, ndcg_at_k  — standard IR metrics
  time_iou_match   — IoU >= 0.5 to match a predicted window to a label
  cohen_kappa      — inter-annotator agreement for 2 annotators
  fleiss_kappa     — inter-annotator agreement for N annotators
"""

import math


# ── Type aliases ──────────────────────────────────────────────────────────
# Label: {"episode": str, "start": float, "end": float, "grade": int (0-2)}
# Pred:  {"start": float, "end": float, "score": float}


def time_iou(a_start, a_end, b_start, b_end) -> float:
    inter_s = max(a_start, b_start)
    inter_e = min(a_end, b_end)
    if inter_s >= inter_e:
        return 0.0
    inter = inter_e - inter_s
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _match(pred: dict, labels: list, iou_thresh: float = 0.5) -> int:
    """Return grade of best-matching label (0 if no match)."""
    best = 0
    for lb in labels:
        if lb.get("grade", 0) > 0:
            iou = time_iou(pred["start"], pred["end"], lb["start"], lb["end"])
            if iou >= iou_thresh:
                best = max(best, lb["grade"])
    return best


def precision_at_k(preds: list, labels: list, k: int, iou_thresh: float = 0.5) -> float:
    top = preds[:k]
    hits = sum(1 for p in top if _match(p, labels, iou_thresh) > 0)
    return hits / k if k else 0.0


def recall_at_k(preds: list, labels: list, k: int, iou_thresh: float = 0.5) -> float:
    pos = [lb for lb in labels if lb.get("grade", 0) > 0]
    if not pos:
        return 0.0
    top = preds[:k]
    hits = sum(1 for p in top if _match(p, pos, iou_thresh) > 0)
    return hits / len(pos)


def _dcg(grades: list) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(preds: list, labels: list, k: int, iou_thresh: float = 0.5) -> float:
    top = preds[:k]
    pred_grades = [_match(p, labels, iou_thresh) for p in top]
    ideal = sorted([lb.get("grade", 0) for lb in labels], reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(pred_grades) / idcg if idcg > 0 else 0.0


def cohen_kappa(ann1: list, ann2: list) -> float:
    """
    Cohen's kappa for two annotators.
    ann1, ann2: lists of integer grades (same length).
    """
    n = len(ann1)
    if n == 0 or len(ann2) != n:
        return 0.0
    classes = sorted(set(ann1) | set(ann2))
    # Observed agreement
    p_o = sum(a == b for a, b in zip(ann1, ann2)) / n
    # Expected agreement
    p_e = sum(
        (ann1.count(c) / n) * (ann2.count(c) / n)
        for c in classes
    )
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def fleiss_kappa(ratings: list, n_cats: int = 3) -> float:
    """
    Fleiss' kappa for N annotators.
    ratings: list of dicts {grade: count} per item,
             or list of lists where each inner list is one item's ratings.
    """
    if not ratings:
        return 0.0
    # Normalise to list-of-lists
    if isinstance(ratings[0], dict):
        rows = [[r.get(c, 0) for c in range(n_cats)] for r in ratings]
    else:
        rows = ratings
    n_items = len(rows)
    n_j = sum(sum(row) for row in rows) // n_items  # raters per item
    if n_j == 0:
        return 0.0
    # P_i
    P_i = [(sum(r[c]**2 for c in range(n_cats)) - n_j) / (n_j * (n_j - 1)) for r in rows]
    P_bar = sum(P_i) / n_items
    # P_j
    total = n_items * n_j
    p_j = [sum(row[c] for row in rows) / total for c in range(n_cats)]
    P_e = sum(p**2 for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0
