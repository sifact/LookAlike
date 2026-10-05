import numpy as np
import pytest

from visual_search.evaluation import metrics as M


def test_ndcg_perfect_and_reversed():
    ideal = np.array([2, 2, 1, 0])
    assert M.ndcg(np.array([2, 2, 1]), ideal) == pytest.approx(1.0)
    worse = M.ndcg(np.array([1, 2, 2]), ideal)
    assert 0 < worse < 1


def test_ndcg_hand_computed():
    # DCG = 1/log2(2) + 0 + 2/log2(4) = 1 + 1 = 2 ; IDCG = 2/1 + 1/log2(3) + 0
    expected = 2 / (2 + 1 / np.log2(3))
    assert M.ndcg(np.array([1, 0, 2]), np.array([2, 1, 0, 0])) == pytest.approx(expected)


def test_ndcg_uses_all_candidates_for_ideal():
    # Relevant items exist in the gallery but none were retrieved -> 0, not undefined.
    assert M.ndcg(np.array([0, 0]), np.array([2])) == 0.0
    assert M.ndcg(np.array([0, 0]), np.array([0])) == 0.0


def test_binary_metrics():
    hits = np.array([0, 1, 1, 0])
    assert M.reciprocal_rank(hits) == 0.5
    assert M.precision_at_k(hits) == 0.5
    assert M.recall_at_k(hits, total_relevant=4) == 0.5
    # AP: (1/2 + 2/3) / min(4, 3)
    assert M.average_precision(hits, total_relevant=3) == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert M.reciprocal_rank(np.zeros(3)) == 0.0
