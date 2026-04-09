"""Regression tests for fit_single_gaussian and fit_bayesian_gmm.

These tests cover the float32 catastrophic cancellation failure mode that
caused the first sweep run (20260409T080607Z) to crash. sklearn's diagonal
variance computation E[X²] - mean² + reg_covar produces tiny negative values
in float32 when the per-dimension variance is small relative to the squared
mean — exactly the regime of small UMAP-tier firms in our data.

The fix in scripts/run_kmax_sweep.py is to cast inputs to float64 and bump
reg_covar to 1e-4. These tests verify that fix holds.

The primary fixture (`degenerate_firm_actual.npy`) is the real production
data from a single-Gaussian firm that deterministically crashes sklearn under
the old configuration (float32 + reg_covar=1e-6). It was extracted from the
v3 production data on 2026-04-09 during diagnosis of the first failed AWS run.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from scripts.run_kmax_sweep import (
    fit_single_gaussian,
    fit_bayesian_gmm,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures: degenerate inputs from real production data
# ---------------------------------------------------------------------------

@pytest.fixture
def real_degenerate_firm_X():
    """Production data from a single-Gaussian firm that crashes sklearn.

    Loaded from `tests/unit/fixtures/degenerate_firm_actual.npy`.
    Shape: (n, 50), dtype: float32. This is the real 50D UMAP output for
    one of the firms that triggered the first sweep crash on AWS.
    """
    return np.load(FIXTURE_DIR / "degenerate_firm_actual.npy")


def test_fixture_actually_reproduces_old_bug(real_degenerate_firm_X):
    """Sanity check: the fixture must actually crash sklearn under the OLD config.

    If this test ever fails, it means the fixture no longer reproduces the bug
    we are protecting against. Either the fixture is stale, or sklearn's
    behavior has changed and we need a new fixture.
    """
    X = real_degenerate_firm_X
    assert X.dtype == np.float32, "Fixture must be float32 to reproduce the bug"
    gm = GaussianMixture(
        n_components=1, covariance_type="diag", reg_covar=1e-6, random_state=42
    )
    with pytest.raises(ValueError, match="ill-defined empirical covariance"):
        gm.fit(X)  # noqa: should crash, that's the point


@pytest.fixture
def gmm_tier_float32_X():
    """Realistic GMM-tier input: 100 patents, 50D, multimodal."""
    rng = np.random.RandomState(7)
    n, d = 100, 50
    # Two clusters
    cluster1 = rng.normal(loc=5.0, scale=0.4, size=(n // 2, d)).astype(np.float32)
    cluster2 = rng.normal(loc=7.0, scale=0.4, size=(n - n // 2, d)).astype(np.float32)
    X = np.vstack([cluster1, cluster2])
    return X


@pytest.fixture
def fake_config():
    return {
        "portfolio": {
            "weight_concentration_prior": 1.0,
            "mean_precision_prior": 1.0,
            "degrees_of_freedom_prior": 52,
            "max_iter": 200,
            "n_init": 1,  # Speed up tests
            "random_state": 42,
        }
    }


# ---------------------------------------------------------------------------
# fit_single_gaussian regression tests
# ---------------------------------------------------------------------------

def test_fit_single_gaussian_handles_real_degenerate_firm(real_degenerate_firm_X):
    """The actual production crash: real firm data that breaks sklearn under float32.

    This is the bug fix verification: real production data that deterministically
    crashed the original sweep run must now succeed under fit_single_gaussian.
    """
    result = fit_single_gaussian("test_real_firm", real_degenerate_firm_X)

    assert result["gvkey"] == "test_real_firm"
    assert result["tier"] == "single_gaussian"
    assert result["n_components"] == 1
    assert result["n_patents"] == real_degenerate_firm_X.shape[0]
    assert result["means"].shape == (1, 50)
    assert result["covariances"].shape == (1, 50)
    assert result["weights"].shape == (1,)
    assert np.isclose(result["weights"].sum(), 1.0)
    # Stored in float64 for downstream BC computation
    assert result["means"].dtype == np.float64
    assert result["covariances"].dtype == np.float64
    # All per-dim covariances should be strictly positive (no degenerate ones)
    assert (result["covariances"] > 0).all()


def test_fit_single_gaussian_handles_trivially_degenerate():
    """Even nearly-identical rows should not crash."""
    rng = np.random.RandomState(0)
    n, d = 5, 50
    base = rng.uniform(3.0, 10.0, size=d).astype(np.float32)
    X = np.tile(base, (n, 1)) + (1e-7 * rng.randn(n, d)).astype(np.float32)

    result = fit_single_gaussian("trivial_firm", X)
    assert result["n_components"] == 1
    assert result["covariances"].shape == (1, 50)
    # Variance should be at least reg_covar (1e-4) since data has no spread
    assert (result["covariances"] >= 1e-5).all()


def test_fit_single_gaussian_preserves_input(real_degenerate_firm_X):
    """The input array should not be mutated (defensive)."""
    X_copy = real_degenerate_firm_X.copy()
    fit_single_gaussian("preserve_test", real_degenerate_firm_X)
    np.testing.assert_array_equal(real_degenerate_firm_X, X_copy)


# ---------------------------------------------------------------------------
# fit_bayesian_gmm regression tests
# ---------------------------------------------------------------------------

def test_fit_bayesian_gmm_handles_float32_input(gmm_tier_float32_X, fake_config):
    """Bayesian GMM should also accept float32 input without crashing."""
    global_mean = np.full(50, 6.0, dtype=np.float32)
    global_var = np.full(50, 1.0, dtype=np.float32)

    result = fit_bayesian_gmm(
        "test_firm_gmm",
        gmm_tier_float32_X,
        k_max=10,
        global_mean=global_mean,
        global_var=global_var,
        config=fake_config,
    )

    assert result["gvkey"] == "test_firm_gmm"
    assert result["tier"] == "gmm"
    assert result["n_components"] >= 1
    assert result["means"].shape == (result["n_components"], 50)
    assert result["covariances"].shape == (result["n_components"], 50)
    assert np.isclose(result["weights"].sum(), 1.0)
    assert (result["covariances"] > 0).all()


def test_fit_bayesian_gmm_kmax_guard_when_n_small(fake_config):
    """When n_samples < k_max, the actual_kmax guard should prevent sklearn ValueError."""
    rng = np.random.RandomState(0)
    X = rng.normal(loc=5.0, scale=0.5, size=(8, 50)).astype(np.float32)  # n=8
    global_mean = np.full(50, 5.0, dtype=np.float32)
    global_var = np.full(50, 0.5, dtype=np.float32)

    # k_max=15 but n_samples=8 → actual_kmax should be 7
    result = fit_bayesian_gmm(
        "small_gmm_firm", X, k_max=15,
        global_mean=global_mean, global_var=global_var, config=fake_config,
    )
    assert result["n_components"] >= 1
    assert result["n_components"] <= 7  # Cannot exceed n_samples - 1
