"""Phase bucketing — the one place a phase in [0, 1] becomes a bin index.

Align, readout and locator must all agree on boundaries or they silently diverge.
"""

from __future__ import annotations

import numpy as np

from robot_ood.data import schema


def phase_bins(phase: np.ndarray, nbins: int = schema.NBINS_ALIGN) -> np.ndarray:
    """phase [N] f64 in [0, 1] -> [N] i64 in [0, nbins), equal width."""
    # phase == 1.0 folds into the last bin via this clip; not the same binning as
    # train/loss/train/metrics (float32 torch, no lower clamp) -- do not merge the two.
    return np.clip((np.asarray(phase, dtype=np.float64) * nbins).astype(np.int64), 0, nbins - 1)


def phase_hist(phase: np.ndarray, p_node: np.ndarray, nbins: int = schema.NBINS_ALIGN) -> np.ndarray:
    """Rebin a per-node weight vector by phase: phase [V], p_node [V] -> [nbins]."""
    bins = phase_bins(phase, nbins)
    return np.bincount(bins, weights=np.asarray(p_node, dtype=np.float64), minlength=nbins).astype(np.float64)
