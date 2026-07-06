"""Step 3 — empirical convergence orders on GBM.

Expected: Euler-Maruyama strong order ~0.5, Milstein strong order ~1.0,
Euler-Maruyama weak order ~1.0. Order estimation from finite samples is
noisy, so the assertions use generous but discriminating brackets: 0.5
and 1.0 cannot both fit in the same bracket.
"""

import numpy as np

from sde_lab import GBM, euler_maruyama, milstein, strong_error, weak_error, estimate_order

MU, SIGMA, X0, T = 0.05, 0.4, 1.0, 1.0
STEP_COUNTS = [8, 16, 32, 64, 128]


def _strong_errors(scheme_name, n_paths=50_000, seed=123):
    gbm = GBM(MU, SIGMA)
    rng = np.random.default_rng(seed)
    hs, errs = [], []
    # Finest grid first; coarser grids reuse the SAME Brownian motion by
    # summing consecutive fine increments (essential for strong error).
    n_fine = max(STEP_COUNTS)
    dt_fine = T / n_fine
    dW_fine = rng.standard_normal((n_paths, n_fine)) * np.sqrt(dt_fine)
    W_T = dW_fine.sum(axis=1)
    exact = gbm.exact_terminal(X0, T, W_T)

    for n_steps in STEP_COUNTS:
        factor = n_fine // n_steps
        dW = dW_fine.reshape(n_paths, n_steps, factor).sum(axis=2)
        if scheme_name == "euler":
            X = euler_maruyama(X0, gbm.drift, gbm.diffusion, T, n_steps,
                               n_paths, dW=dW)
        else:
            X = milstein(X0, gbm.drift, gbm.diffusion, gbm.diffusion_dx,
                         T, n_steps, n_paths, dW=dW)
        hs.append(T / n_steps)
        errs.append(strong_error(X[:, -1], exact))
    return hs, errs


def test_euler_strong_order_is_half():
    hs, errs = _strong_errors("euler")
    p = estimate_order(hs, errs)
    assert 0.35 < p < 0.75, f"estimated strong order {p:.3f}, expected ~0.5"


def test_milstein_strong_order_is_one():
    hs, errs = _strong_errors("milstein")
    p = estimate_order(hs, errs)
    assert 0.8 < p < 1.2, f"estimated strong order {p:.3f}, expected ~1.0"


def test_errors_decrease_with_h():
    for name in ("euler", "milstein"):
        _, errs = _strong_errors(name, n_paths=20_000)
        assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:])), name


def test_euler_weak_order_is_one():
    """Weak error |E X_T^h - E X_T| with E X_T known in closed form.

    Two tricks make the h-bias measurable above the Monte Carlo noise:
    - a larger drift (the Euler bias on the mean is ~ mu^2 h T/2 * E X_T,
      invisible when mu is tiny),
    - antithetic pairing (Z and -Z), which kills most of the MC noise.
    """
    gbm = GBM(0.5, SIGMA)  # larger drift => measurable weak bias
    rng = np.random.default_rng(2024)
    n_paths = 200_000
    hs, errs = [], []
    for n_steps in [2, 4, 8, 16]:
        dt = T / n_steps
        Z = rng.standard_normal((n_paths, n_steps))
        dW = np.vstack([Z, -Z]) * np.sqrt(dt)
        X = euler_maruyama(X0, gbm.drift, gbm.diffusion, T, n_steps,
                           2 * n_paths, dW=dW)
        hs.append(dt)
       