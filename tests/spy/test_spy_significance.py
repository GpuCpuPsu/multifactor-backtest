"""Significance test for the SPY low-vs-high result: block bootstrap and thinning must
treat overlapping, clustered touches as dependent, not independent."""
import numpy as np
import pytest

import spy_levels_stats as st


def test_thinning_keeps_only_touches_at_least_horizon_apart():
    assert st.thin([0, 5, 10, 25, 30, 50], 20) == [0, 25, 50]


def test_bootstrap_of_identical_groups_is_zero():
    rng = np.random.default_rng(0)
    fwd = rng.normal(0, 0.02, 500)
    flags = (rng.random(500) < 0.2).astype(float)
    d = st.block_bootstrap(fwd, flags, flags, block=20, reps=200, rng=rng)
    assert np.allclose(d, 0.0)


def test_bootstrap_recovers_a_constant_gap():
    rng = np.random.default_rng(0)
    lo = np.zeros(400); hi = np.zeros(400)
    lo[::10] = 1; hi[5::10] = 1
    fwd = np.where(lo == 1, 0.03, 0.01)                       # low touches always +2% better
    d = st.block_bootstrap(fwd, lo, hi, block=40, reps=200, rng=rng)
    assert np.allclose(d, 0.02)


def test_blocks_keep_clustered_days_together():
    """A gap carried by one 30-day cluster must show wide bootstrap spread with long
    blocks -- i.e. the cluster is resampled as one unit, not as 30 independent draws."""
    rng = np.random.default_rng(1)
    n = 3000
    lo = np.zeros(n); hi = np.zeros(n)
    lo[::50] = 1; hi[25::50] = 1
    fwd = rng.normal(0, 0.01, n)
    lo[1000:1030] = 1; fwd[1000:1030] = 0.10                  # one lucky clustered episode
    wide = st.block_bootstrap(fwd, lo, hi, block=60, reps=400, rng=rng).std()
    narrow = st.block_bootstrap(fwd, lo, hi, block=1, reps=400, rng=rng).std()
    assert wide > 1.5 * narrow


def test_welch_two_sided_p_value():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.01, 0.02, 400), rng.normal(0.0, 0.02, 400)
    diff, z, p = st.welch(a, b)
    assert diff == pytest.approx(a.mean() - b.mean()) and z > 3 and p < 0.01
