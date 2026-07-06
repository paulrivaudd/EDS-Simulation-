"""Step 1 & 2 — Euler-Maruyama and Milstein on GBM (exact solution known)."""

import numpy as np
import pytest

from sde_lab import GBM, euler_maruyama, milstein

MU, SIGMA, X0, T = 0.05, 0.4, 100.0, 1.0


def _run(scheme_name, n_steps, n_paths, seed):
    """Run a scheme against the exact GBM solution on the SAME Brownian path."""
    gbm = GBM(MU, SIGMA)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    W_T = dW.sum(axis=1)
    exact = gbm.exact_terminal(X0, T, W_T)

    if scheme_name == "euler":
        X = euler_maruyama(X0, gbm.drift, gbm.diffusion, T, n_steps, n_paths, dW=dW)
    else:
        X = milstein(X0, gbm.drift, gbm.diffusion, gbm.diffusion_dx,
                     T, n_steps, n_paths, dW=dW)
    return X, exact


def test_shapes_and_initial_value():
    X, _ = _run("euler", n_steps=16, n_paths=50, seed=0)
    assert X.shape == (50, 17)
    assert np.allclose(X[:, 0], X0)


def test_euler_close_to_exact_for_fine_grid():
    X, exact = _run("euler", n_steps=512, n_paths=20_000, seed=42)
    err = np.mean(np.abs(X[:, -1] - exact))
    # strong error ~ C sqrt(h); must be small relative to X0 on a fine grid
    assert err < 1.5


def test_milstein_close_to_exact_for_fine_grid():
    X, exact = _run("milstein", n_steps=512, n_paths=20_000, seed=42)
    err = np.mean(np.abs(X[:, -1] - exact))
    assert err < 0.1


def test_milstein_beats_euler_pathwise():
    """On the same Brownian increments, Milstein must be strictly closer."""
    n_steps, n_paths, seed = 32, 20_000, 7
    X_e, exact = _run("euler", n_steps, n_paths, seed)
    X_m, exact_m = _run("milstein", n_steps, n_paths, seed)
    np.testing.assert_allclose(exact, exact_m)  # same driving noise
    err_e = np.mean(np.abs(X_e[:, -1] - exact))
    err_m = np.mean(np.abs(X_m[:, -1] - exact))
    assert err_m < 0.5 * err_e


def test_euler_mean_matches_exact_mean():
    gbm = GBM(MU, SIGMA)
    rng = np.random.default_rng(3)
    X = euler_maruyama(X0, gbm.drift, gbm.diffusion, T, 64, 200_000, rng=rng)
    assert np.mean(X[:, -1]) == pytest.approx(gbm.exact_mean(X0, T), rel=5e-3)
