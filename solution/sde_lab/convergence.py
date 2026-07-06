"""Strong and weak convergence measurement for discretization schemes.

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
    """
    return float(np.mean(np.abs(scheme_terminal - exact_terminal)))


def weak_error(scheme_terminal, exact_expectation):
    """|E[X_T^h] - E[X_T]| where E[X_T] is known in closed form."""
    return float(np.abs(np.mean(scheme_terminal) - exact_expectation))


def estimate_order(step_sizes, errors):
    """Least-squares slope of log(error) against log(h).

    Parameters
    ----------
    step_sizes : sequence of h values
    errors : sequence of corresponding errors (same length)

    Returns
    -------
    float — the estimated convergence order p.
    """
    log_h = np.log(np.asarray(step_sizes, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return float(slope)
