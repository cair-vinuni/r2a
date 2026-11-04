"""Tests for robot_ood.eval.phase_skew's strand-lock statistic.

The replay itself needs real artifacts, so it is not exercised end-to-end here. What IS pure and
synthetic-data-testable is the correlation statistic -- _strand_lock_stats and summarize's use of
it -- whether losing track means locking onto whichever strand it seeded on.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.eval.phase_skew import ERR_TAIL_THRESHOLD, Sample, _strand_lock_stats, summarize


def _sample(owner: int, phase_gt: float, phase_hat: float, top1_owner: int) -> Sample:
    return Sample(owner=owner, phase_gt=phase_gt, phase_hat=phase_hat, top1_owner=top1_owner)


def test_strand_lock_stats_all_on_strand_reports_zero_off_strand_frac():
    off_strand = np.zeros(10, dtype=bool)
    err = np.linspace(-0.5, 0.5, 10)
    stats = _strand_lock_stats(off_strand, err)
    assert stats["overall"]["off_strand_frac"] == 0.0
    assert stats["small_err"]["off_strand_frac"] == 0.0
    assert stats["large_err"]["off_strand_frac"] == 0.0


def test_strand_lock_stats_isolates_large_error_off_strand_checks():
    # 4 well-tracked on-strand checks, 4 lost-track off-strand checks: the correlation should be
    # perfectly visible in the split even though the pooled ("overall") frac is 0.5, hiding it.
    off_strand = np.array([False, False, False, False, True, True, True, True])
    err = np.array([0.01, -0.02, 0.03, -0.01, 0.5, -0.4, 0.6, -0.55])
    stats = _strand_lock_stats(off_strand, err)
    assert stats["overall"]["off_strand_frac"] == pytest.approx(0.5)
    assert stats["small_err"]["off_strand_frac"] == pytest.approx(0.0)
    assert stats["large_err"]["off_strand_frac"] == pytest.approx(1.0)
    assert stats["small_err"]["n"] == 4
    assert stats["large_err"]["n"] == 4


def test_strand_lock_stats_empty_bucket_reports_none_not_nan():
    off_strand = np.array([False, False])
    err = np.array([0.01, -0.02])  # nothing crosses ERR_TAIL_THRESHOLD
    stats = _strand_lock_stats(off_strand, err)
    assert stats["large_err"]["n"] == 0
    assert stats["large_err"]["off_strand_frac"] is None


def test_summarize_reports_off_strand_frac_per_phase_bin():
    samples = [
        _sample(owner=0, phase_gt=0.01, phase_hat=0.02, top1_owner=1),   # entry, off-strand, tiny err
        _sample(owner=0, phase_gt=0.9, phase_hat=0.9 - 2 * ERR_TAIL_THRESHOLD, top1_owner=7),
        _sample(owner=0, phase_gt=0.9, phase_hat=0.9 + 0.001, top1_owner=0),  # on-strand, tracked
    ]
    report = summarize(samples)
    late_bin = report["by_phase_bin"]["0.75-1.0"]
    assert late_bin["off_strand_frac"] == pytest.approx(1.0 / 2.0)
    # the lost-track (large-error) check in the late bin is exactly the off-strand one
    assert late_bin["strand_lock"]["large_err"]["off_strand_frac"] == pytest.approx(1.0)
    assert late_bin["strand_lock"]["small_err"]["off_strand_frac"] == pytest.approx(0.0)
    assert report["strand_lock"]["stats"]["large_err"]["n"] == 1
