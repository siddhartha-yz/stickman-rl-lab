from __future__ import annotations

import numpy as np
import pytest

from scripts.distill_successful_rollouts import epoch_batch_indices


def test_natural_batches_cover_each_sample_once() -> None:
    rng = np.random.default_rng(3)
    final_mask = np.array([False, False, True, True, False, True])
    batches = epoch_batch_indices(rng, 6, 4, final_mask, None)
    combined = np.concatenate(batches)
    assert sorted(combined.tolist()) == list(range(6))


def test_balanced_batches_use_requested_final_fraction() -> None:
    rng = np.random.default_rng(4)
    final_mask = np.array([False] * 8 + [True] * 2)
    batches = epoch_batch_indices(rng, 10, 6, final_mask, 0.5)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 6
        assert int(np.count_nonzero(final_mask[batch])) == 3


def test_balanced_batches_reject_invalid_fraction() -> None:
    rng = np.random.default_rng(5)
    final_mask = np.array([False, True])
    with pytest.raises(ValueError, match="strictly between"):
        epoch_batch_indices(rng, 2, 2, final_mask, 1.0)
