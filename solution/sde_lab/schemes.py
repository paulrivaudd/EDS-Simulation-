"""Generic discretization schemes for scalar SDEs.

We consider SDEs of the form

    dX_t = a(X_t, t) dt + b(X_t, t) dW_t,    X_0 = x0.

Both schemes advance all paths at once (vectorized over paths, loop over
time steps only) and can consume pre-drawn Brownian increments `dW`, which
is essential for strong-convergence studies (the coarse and fine schemes
must be driven by the SAME Brownian motion).
"""

import numpy as np


def _brownian_increments(rng, n_paths, n_steps, dt):
    """Draw dW ~ N(0, dt) of shape (n_paths, n_steps)."""
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def euler_maruyama(x0, drift, diffusion, T, n_steps, n_paths, rng=None, dW=None):
    """Simulate paths with the Euler-Maruyama scheme.

        X_{n+1} = X_n + a(X_n, t_n) dt + b(X_n, t_n) dW_n

    Parameters
    ----------
    x0 : float             initial value
    drift : callable       a(x, t), vectorized in x
    diffusion : callable   b(x, t), vectorized in x
    T : float              horizon
    n_steps : int          number of time steps (dt = T / n_steps)
    n_paths : int          number of paths
    rng : np.random.Generator, optional (ignored if dW is given)
    dW : ndarray (n_paths, n_steps), optional pre-drawn Brownian increments

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1) — full paths, X[:, 0] == x0.
    """
    dt = T / n_steps
    if dW is None:
        if rng is None:
            rng = np.random.default_rng()
        dW = _brownian_increments(rng, n_paths, n_steps, dt)

    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for n in range(n_steps):
        t = n * dt
        x = X[:, n]
        X[:, n + 1] = x + drift(x, t) * dt + diffusion(x, t) * dW[:, n]
    return X


def milstein(x0, drift, diffusion, diffusion_dx, T, n_steps, n_paths,
             rng=None, dW=None):
    """Simulate paths with the Milstein scheme.

        X_{n+1} = X_n + a dt + b dW_n + 0.5 * b * b' * (dW_n^2 - dt)

    where b' = ∂b/∂x. The extra term raises the strong order from 0.5 to 1.

    Parameters are as in `euler_maruyama`, plus:
    diffusion_dx : callable   b'(x, t) = ∂b/∂x, vectorized in x

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1).
    """
    dt = T / n_steps
    if dW is None:
        if rng is None:
            rng = np.random.default_rng()
        dW = _brownian_increments(rng, n_paths, n_steps, dt)

    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for n in range(n_steps):
        t = n * dt
        x = X[:, n]
        b = diffusion(x, t)
        X[:, n + 1] = (
            x
            + drift(x, t) * dt
            + b * dW[:, n]
            + 0.5 * b * diffusion_dx(x, t) * (dW[:, n] ** 2 - dt)
        )
    return X
