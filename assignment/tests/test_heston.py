"""Step 5 — Heston model: full-truncation Euler simulation and call pricing."""

import numpy as np
import pytest

from sde_lab import Heston
from sde_lab.heston import bs_call

S0, V0, K, T = 100.0, 0.04, 100.0, 1.0
R = 0.03


def test_simulation_shapes_and_positivity():
    h = Heston(r=R, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    rng = np.random.default_rng(1)
    S, v = h.simulate(S0, V0, T, n_steps=50, n_paths=1_000, rng=rng)
    assert S.shape == (1_000, 51) and v.shape == (1_000, 51)
    assert (S > 0).all()          # log-Euler guarantees positive prices
    assert np.isfinite(v).all()   # full truncation guarantees no NaN


def test_martingale_property():
    """Under the risk-neutral measure, e^{-rT} E[S_T] = S0."""
    h = Heston(r=R, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    rng = np.random.default_rng(5)
    S, _ = h.simulate(S0, V0, T, n_steps=100, n_paths=400_000, rng=rng)
    assert np.exp(-R * T) * np.mean(S[:, -1]) == pytest.approx(S0, rel=3e-3)


def test_reduces_to_black_scholes_when_vol_of_vol_is_zero():
    """With xi = 0 and v0 = theta the variance stays constant at v0, so the
    Heston price must equal Black-Scholes with sigma = sqrt(v0)."""
    h = Heston(r=R, kappa=2.0, theta=V0, xi=0.0, rho=0.0)
    rng = np.random.default_rng(9)
    price, stderr = h.call_price(S0, V0, K, T, n_steps=100,
                                 n_paths=400_000, rng=rng)
    ref = bs_call(S0, K, T, R, np.sqrt(V0))
    assert price == pytest.approx(ref, abs=3.5 * stderr)


def test_put_call_parity():
    """C - P = S0 - K e^{-rT}, computed on the same simulated paths."""
    h = Heston(r=R, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    rng = np.random.default_rng(13)
    S, _ = h.simulate(S0, V0, T, n_steps=100, n_paths=400_000, rng=rng)
    disc = np.exp(-R * T)
    call = disc * np.maximum(S[:, -1] - K, 0.0)
    put = disc * np.maximum(K - S[:, -1], 0.0)
    lhs = np.mean(call) - np.mean(put)
    rhs = S0 - K * disc
    stderr = np.std(call - put, ddof=1) / np.sqrt(S.shape[0])
    assert lhs == pytest.approx(rhs, abs=4 * stderr)


def test_negative_rho_creates_left_skew():
    """rho < 0 fattens the left tail: OTM puts are worth more than under
    rho > 0 with otherwise identical parameters."""
    rng1 = np.random.default_rng(17)
    rng2 = np.random.default_rng(17)
    h_neg = Heston(r=R, kappa=2.0, theta=0.04, xi=0.6, rho=-0.8)
    h_pos = Heston(r=R, kappa=2.0, theta=0.04, xi=0.6, rho=+0.8)
    S_neg, _ = h_neg.simulate(S0, V0, T, 100, 200_000, rng=rng1)
    S_pos, _ = h_pos.simulate(S0, V0, T, 100, 200_000, rng=rng2)
    K_otm_put = 80.0
    put_neg = np.mean(np.maximum(K_otm_put - S_neg[:, -1], 0.0))
    put_pos = np.mean(np.maximum(K_otm_put - S_pos[:, -1], 0.0))
    assert put_neg > put_pos
