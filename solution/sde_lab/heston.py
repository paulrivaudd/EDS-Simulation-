"""Heston stochastic-volatility model.

    dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Simulation: full-truncation Euler on the variance (v+ = max(v, 0) inside
both coefficients) combined with an EXACT log-Euler step on the asset:

    v_{n+1}    = v_n + kappa (theta - v_n^+) dt + xi sqrt(v_n^+) dW_n^v
    ln S_{n+1} = ln S_n + (r - v_n^+ / 2) dt + sqrt(v_n^+) dW_n^S

Correlated increments: dW^S = rho dW^v + sqrt(1 - rho^2) dW^perp.

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
        """
        if rng is None:
            rng = np.random.default_rng()
        dt = T / n_steps
        sqdt = np.sqrt(dt)

        logS = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        logS[:, 0] = np.log(s0)
        v[:, 0] = v0

        for n in range(n_steps):
            vp = np.maximum(v[:, n], 0.0)
            Zv = rng.standard_normal(n_paths)
            Zp = rng.standard_normal(n_paths)
            dWv = Zv * sqdt
            dWs = (self.rho * Zv + np.sqrt(1 - self.rho**2) * Zp) * sqdt

            v[:, n + 1] = (v[:, n]
                           + self.kappa * (self.theta - vp) * dt
                           + self.xi * np.sqrt(vp) * dWv)
            logS[:, n + 1] = (logS[:, n]
                              + (self.r - 0.5 * vp) * dt
                              + np.sqrt(vp) * dWs)

        return np.exp(logS), v

    def call_price(self, s0, v0, K, T, n_steps, n_paths, rng=None):
        """Monte Carlo price of a European call:  e^{-rT} E[(S_T - K)^+].

        Returns
        -------
        (price, stderr) — the MC estimate and its standard error.
        """
        S, _ = self.simulate(s0, v0, T, n_steps, n_paths, rng=rng)
        payoff = np.exp(-self.r * T) * np.maximum(S[:, -1] - K, 0.0)
        price = float(np.mean(payoff))
        stderr = float(np.std(payoff, ddof=1) / np.sqrt(n_paths))
        return price, stderr


def bs_call(s0, K, T, r, sigma):
    """Black-Scholes call price (reference for the xi -> 0 sanity check)."""
    from math import log, sqrt, erf, exp

    def N(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    d1 = (log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return s0 * N(d1) - K * exp(-r * T) * N(d2)
