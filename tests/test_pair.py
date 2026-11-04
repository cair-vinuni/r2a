import numpy as np
import pytest

from robot_ood.runtime.pair import PairedRetrieval, pair_table


def _pairs():
    # demo 0: a = [0,4),[4,8),[8,10)  b = [0,5),[5,10);  demo 1: a = [10,20)  b = [10,13),[13,20)
    spans_a = np.array([[0, 4], [4, 8], [8, 10], [10, 20]])
    owner_a = np.array([0, 0, 0, 1])
    spans_b = np.array([[0, 5], [5, 10], [10, 13], [13, 20]])
    owner_b = np.array([0, 0, 1, 1])
    return spans_a, owner_a, spans_b, owner_b


def test_pair_table_overlaps_within_demo():
    sa, oa, sb, ob = _pairs()
    indptr, indices = pair_table(sa, oa, sb, ob)
    rows = [indices[indptr[i]:indptr[i + 1]].tolist() for i in range(len(sa))]
    assert rows == [[0], [0, 1], [1], [2, 3]]
    indptr, indices = pair_table(sb, ob, sa, oa)
    rows = [indices[indptr[i]:indptr[i + 1]].tolist() for i in range(len(sb))]
    assert rows == [[0, 1], [1, 2], [3], [3]]


def test_pair_table_refuses_unpaired_segment():
    sa, oa, sb, ob = _pairs()
    with pytest.raises(ValueError):
        pair_table(sa, np.array([0, 0, 0, 2]), sb, ob)


def test_prior_is_best_partner_score_over_pairs():
    sa, oa, sb, ob = _pairs()
    paired = PairedRetrieval({"left": pair_table(sa, oa, sb, ob), "right": pair_table(sb, ob, sa, oa)})
    s_right = np.array([1.0, 5.0, -2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(paired.prior("left", s_right), [1.0, 5.0, 5.0, 3.0])
    s_left = np.array([0.5, -1.0, 2.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(paired.prior("right", s_left), [0.5, 2.0, 7.0, 7.0])
    assert paired.partner == {"left": "right", "right": "left"}
