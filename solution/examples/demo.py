"""End-to-end demo: convergence table + rates + Heston smile sketch."""

import numpy as np

from sde_lab import (GBM, CIR, Heston, Vasicek, estimate_order,
                     euler_maruyama, milstein, strong_error)
from sde_lab.heston import bs_call
from sde_lab.models import zcb_mc_price


def convergence_table():
    gbm = GBM(0.05, 0.4)
    x0, T, n_paths = 1.0, 1.0, 50_000
    rng = np.random.default_rng(0)
    n_fine = 128
    dW_fine = rng.standard_normal((n_paths, n_fine)) * np.sqrt(T / n_fine)
    exact = gbm.exact_terminal(x0, T, dW_fine.sum(axis=1))

    print("\n=== Strong convergence on GBM ===")
    print(f"{'n_steps':>8} {'h':>10} {'Euler err':>12} {'Milstein err':>13}")
    hs, ee, em = [], [], []
    for n_steps in [8, 16, 32, 64, 128]:
        dW = dW_fine.reshape(n_paths, n_steps, n_fine // n_steps).sum(axis=2)
        Xe = euler_maruyama(x0, gbm.drift, gbm.diffusion, T, n_steps, n_paths, dW=dW)
        Xm = milstein(x0, gbm.drift, gbm.diffusion, gbm.diffusion_dx,
                      T, n_steps, n_paths, dW=dW)
        h = T / n_steps
        e1, e2 = strong_error(Xe[:, -1], exact), strong_error(Xm[:, -1], exact)
        hs.append(h); ee.append(e1); em.append(e2)
        print(f"{n_steps:>8} {h:>10.4f} {e1:>12.3e} {e2:>13.3e}")
    print(f"Estimated orders: Euler {estimate_order(hs, ee):.2f} (theory 0.5), "
          f"Milstein {estimate_order(hs, em):.2f} (theory 1.0)")


def rates():
    print("\n=== Zero-coupon bonds, T = 2y ===")
    rng = np.random.default_rng(1)
    vas, r0 = Vasicek(1.5, 0.04, 0.02), 0.06
    r = euler_maruyama(r0, vas.drift, vas.diffusion, 2.0, 200, 100_000, rng=rng)
    print(f"Vasicek  MC {zcb_mc_price(r, 2.0):.6f} | closed form {vas.zcb_price(r0, 2.0):.6f}")
    cir, r0 = CIR(1.2, 0.05, 0.25), 0.03
    r = euler_maruyama(r0, cir.drift, cir.diffusion, 2.0, 400, 100_000, rng=rng)
    print(f"CIR      MC {zcb_mc_price(r, 2.0):.6f} | closed form {cir.zcb_price(r0, 2.0):.6f}")


def heston_smile():
    print("\n=== Heston implied-skew sketch (T = 1y) ===")
    h = Heston(r=0.03, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    rng = np.random.default_rng(2)
    S, _ = h.simulate(100.0, 0.04, 1.0, 100, 200_000, rng=rng)
    disc = np.exp(-0.03)
    for K in [80, 90, 100, 110, 120]:
        price = disc * np.mean(np.maximum(S[:, -1] - K, 0.0))
        bs = bs_call(100.0, K, 1.0, 0.03, 0.2)
        print(f"K={K:>4}: Heston call {price:>7.3f} | BS(20%) {bs:>7.3f}")


if __name__ == "__main__":
    convergence_table()
    rates()
    heston_smile()
