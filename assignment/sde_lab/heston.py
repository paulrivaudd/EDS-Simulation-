"""Heston stochastic-volatility model.  (Step 5)

    dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Simulation recipe: full-truncation Euler on the variance (v+ = max(v, 0)
inside both coefficients) combined with an EXACT log-Euler step on the
asset:

    v_{n+1}    = v_n + kappa (theta - v_n^+) dt + xi sqrt(v_n^+) dW_n^v
    ln S_{n+1} = ln S_n + (r - v_n^+ / 2) dt + sqrt(v_n^+) dW_n^S

Correlated increments: with Zv, Zp two INDEPENDENT standard normals,

    dW^v = Zv sqrt(dt)
    dW^S = (rho Zv + sqrt(1 - rho^2) Zp) sqrt(dt)

Simulating ln S (rather than S directly) guarantees positivity of the
asset and removes the discretization bias of the drift term r S dt.
"""

import numpy as np


class Heston:
    def __init__(self, r, kappa, theta, xi, rho):
        """
        r     : risk-free rate
        kappa : variance mean-reversion speed
        theta : long-run variance
        xi    : vol-of-vol
        rho   : correlation between asset and variance Brownian motions
        """
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate(self, s0, v0, T, n_steps, n_paths, rng=None):
        """Simulate (S, v) paths with full-truncation Euler.

        Returns
        -------
        S : ndarray (n_paths, n_steps + 1)
        v : ndarray (n_paths, n_steps + 1)   (raw values, may dip below 0)

        TODO:
        - default rng if None; dt = T / n_steps
        - allocate logS and v, fill first columns with log(s0) and v0
        - per step: floor v, draw Zv and Zp, build correlated increments,
          update v (full truncation) and logS (log-Euler) as in the module
          docstring
        - return (np.exp(logS), v)
        """
        raise NotImplementedError("Step 5: implement Heston.simulate")

    def call_price(self, s0, v0, K, T, n_steps, n_paths, rng=None):
        """Monte Carlo price of a European call:  e^{-rT} E[(S_T - K)^+].

        Returns
        -------
        (price, stderr) — the MC estimate and its standard error
        (std of the discounted payoff, ddof=1, divided by sqrt(n_paths)).

        TODO: simulate, discount the payoff, return mean and standard error.
        """
        raise NotImplementedError("Step 5: implement Heston.call_price")


def bs_call(s0, K, T, r, sigma):
    """Black-Scholes call price (reference for the xi -> 0 sanity check).  (given)"""
    from math import log, sqrt, erf, exp

    def N(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    d1 = (log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return s0 * N(d1) - K * exp(-r * T) * N(d2)
