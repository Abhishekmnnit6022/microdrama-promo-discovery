"""Tests for eval/metrics.py using hand-computed cases."""
import pytest
from eval.metrics import (
    time_iou, precision_at_k, recall_at_k, ndcg_at_k,
    cohen_kappa, fleiss_kappa,
)

LABELS = [
    {"start": 0.0,  "end": 15.0, "grade": 2},
    {"start": 30.0, "end": 50.0, "grade": 1},
    {"start": 70.0, "end": 90.0, "grade": 0},
]
PREDS_PERFECT = [
    {"start": 0.0,  "end": 15.0, "score": 0.9},
    {"start": 30.0, "end": 50.0, "score": 0.7},
]
PREDS_WRONG = [
    {"start": 100.0, "end": 115.0, "score": 0.9},
    {"start": 200.0, "end": 215.0, "score": 0.7},
]


def test_time_iou_perfect():
    assert time_iou(0, 15, 0, 15) == pytest.approx(1.0)


def test_time_iou_no_overlap():
    assert time_iou(0, 10, 20, 30) == pytest.approx(0.0)


def test_time_iou_partial():
    # [0,10] vs [5,15] → intersection 5, union 15
    assert time_iou(0, 10, 5, 15) == pytest.approx(5 / 15)


def test_precision_perfect():
    assert precision_at_k(PREDS_PERFECT, LABELS, k=2) == pytest.approx(1.0)


def test_precision_zero():
    assert precision_at_k(PREDS_WRONG, LABELS, k=2) == pytest.approx(0.0)


def test_recall_perfect():
    # 2 positive labels (grade > 0); both matched
    assert recall_at_k(PREDS_PERFECT, LABELS, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_perfect():
    # Both predictions match grade-2 and grade-1 labels
    score = ndcg_at_k(PREDS_PERFECT, LABELS, k=2)
    assert score > 0.5


def test_ndcg_at_k_zero():
    assert ndcg_at_k(PREDS_WRONG, LABELS, k=2) == pytest.approx(0.0)


def test_cohen_kappa_perfect():
    ann = [0, 1, 2, 1, 0]
    assert cohen_kappa(ann, ann) == pytest.approx(1.0)


def test_cohen_kappa_zero():
    # Completely disagreeing annotators
    a1 = [0, 0, 0]
    a2 = [2, 2, 2]
    k = cohen_kappa(a1, a2)
    assert k <= 0.0


def test_fleiss_kappa_perfect():
    # 3 items, 3 raters each, all agree on grade 1
    ratings = [[0, 3, 0], [0, 3, 0], [0, 3, 0]]
    assert fleiss_kappa(ratings, n_cats=3) == pytest.approx(1.0)


def test_fleiss_kappa_range():
    ratings = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
    k = fleiss_kappa(ratings, n_cats=3)
    assert -1.0 <= k <= 1.0
