"""Strong and weak convergence measurement.  (Step 3)

Definitions (h = T / n_steps):

    strong error(h) = E[ |X_T^h - X_T| ]          (pathwise closeness)
    weak error(h)   = |E[X_T^h] - E[X_T]|         (closeness in law/moments)

Theory (for SDEs with smooth coefficients):
    Euler-Maruyama : strong order 1/2, weak order 1
    Milstein       : strong order 1,   weak order 1

The empirical order is the slope of log(error) vs log(h): if
error(h) ~ C h^p then log error = log C + p log h, so a least-squares
line through the (log h, log error) points estimates p.
"""

import numpy as np


def strong_error(scheme_terminal, exact_terminal):
    """E[|X_T^h - X_T|] from two arrays of terminal values on the SAME paths.

    Both arrays must be driven by the same Brownian increments, otherwise
    the pathwise difference is meaningless.

    TODO: one line with np.mean and np.abs; return a float.
    """
    raise NotImplementedError("Step 3: implement strong_error")


def weak_error(scheme_terminal, exact_expectation):
    """|E[X_T^h] - E[X_T]| where E[X_T] is known in closed form.

    TODO: one line; return a float.
    """
    raise NotImplementedError("Step 3: implement weak_error")


def estimate_order(step_sizes, errors):
    """Least-squares slope of log(error) against log(h).

    Parameters
    ----------
    step_sizes : sequence of h values
    errors : sequence of corresponding errors (same length)

    Returns
    -------
    float — the estimated convergence order p.

    TODO: take logs, fit a degree-1 polynomial (np.polyfit), return the slope.
    """
    raise NotImplementedError("Step 3: implement estimate_order")
