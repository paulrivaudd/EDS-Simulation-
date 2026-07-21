"""Generic discretization schemes for scalar SDEs.  (Steps 1 & 2)

We consider SDEs of the form

    dX_t = a(X_t, t) dt + b(X_t, t) dW_t,    X_0 = x0.

Both schemes must advance all paths at once (vectorized over paths, a
Python loop over time steps only) and must be able to consume pre-drawn
Brownian increments `dW` — this is essential for the strong-convergence
study of Step 3, where the coarse and fine schemes have to be driven by
the SAME Brownian motion.
"""

import numpy as np


def _brownian_increments(rng, n_paths, n_steps, dt):
    """Draw dW ~ N(0, dt) of shape (n_paths, n_steps).  (given)"""
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def euler_maruyama(x0, drift, diffusion, T, n_steps, n_paths, rng=None, dW=None):
    """
    Parameters
    ----------
    x0 : float             initial value
    drift : callable       a(x, t), vectorized in x
    diffusion : callable   b(x, t), vectorized in x
    T : float              horizon
    n_steps : int          number of time steps
    n_paths : int          number of paths
    rng : np.random.Generator, optional (ignored if dW is given)
    dW : ndarray (n_paths, n_steps), optional pre-drawn Brownian increments

    """

    dt = T /n_steps
    if dW is None:
        if rng is None:
            rng = np.random.default_rng()
        dW = _brownian_increments(rng, n_paths, n_steps, dt)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for n in range(n_steps):
        t = n *dt 
        X[:, n + 1] = X[:, n] + drift(X[:, n], t) * dt + diffusion(X[:, n], t) * dW[:, n  ]

    return X 


    


def milstein(x0, drift, diffusion, diffusion_dx, T, n_steps, n_paths,
             rng=None, dW=None):
    """Simulate paths with the Milstein scheme.

    Math reminder — the Milstein correction adds one term to Euler:

        X_{n+1} = X_n + a dt + b dW_n + 0.5 * b * b' * (dW_n^2 - dt)

    where b' = ∂b/∂x. This raises the strong order from 0.5 to 1.

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
        Xn = X[:, n]
        dWn = dW[:, n]
        b = diffusion(Xn, t)
        b_dx = diffusion_dx(Xn, t)
        X[:, n + 1] = (Xn + drift(Xn, t) * dt + b * dWn + 0.5 * b * b_dx * (dWn**2 - dt))

    return X

