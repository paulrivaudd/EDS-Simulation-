"""Step 4 — short-rate models: Vasicek and CIR, ZCB pricing vs closed forms."""

import numpy as np
import pytest

from sde_lab import Vasicek, CIR, euler_maruyama
from sde_lab.models import zcb_mc_price

T = 2.0


class TestVasicek:
    model = Vasicek(kappa=1.5, theta=0.04, sigma=0.02)
    r0 = 0.06

    def _paths(self, n_steps=200, n_paths=100_000, seed=11):
        rng = np.random.default_rng(seed)
        return euler_maruyama(self.r0, self.model.drift, self.model.diffusion,
                              T, n_steps, n_paths, rng=rng)

    def test_mean_matches_closed_form(self):
        r = self._paths()
        assert np.mean(r[:, -1]) == pytest.approx(
            self.model.exact_mean(self.r0, T), abs=3e-4)

    def test_variance_matches_closed_form(self):
        r = self._paths()
        assert np.var(r[:, -1]) == pytest.approx(
            self.model.exact_var(T), rel=0.05)

    def test_zcb_mc_matches_closed_form(self):
        r = self._paths()
        p_mc = zcb_mc_price(r, T)
        p_cf = self.model.zcb_price(self.r0, T)
        assert p_mc == pytest.approx(p_cf, rel=2e-3)


class TestCIR:
    model = CIR(kappa=1.2, theta=0.05, sigma=0.25)  # Feller: 0.12 >= 0.0625
    r0 = 0.03

    def _paths(self, n_steps=400, n_paths=100_000, seed=22):
        rng = np.random.default_rng(seed)
        return euler_maruyama(self.r0, self.model.drift, self.model.diffusion,
                              T, n_steps, n_paths, rng=rng)

    def test_feller_condition_holds(self):
        assert self.model.feller()

    def test_full_truncation_handles_negative_excursions(self):
        """The scheme must run without NaN even though Euler paths can dip
        below zero — that is the whole point of full truncation."""
        r = self._paths(n_steps=50, n_paths=50_000)
        assert np.isfinite(r).all()

    def test_mean_matches_closed_form(self):
        r = self._paths()
        assert np.mean(r[:, -1]) == pytest.approx(
            self.model.exact_mean(self.r0, T), rel=0.02)

    def test_zcb_mc_matches_closed_form(self):
        r = self._paths()
        p_mc = zcb_mc_price(r, T)
        p_cf = self.model.zcb_price(self.r0, T)
        assert p_mc == pytest.approx(p_cf, rel=3e-3)
