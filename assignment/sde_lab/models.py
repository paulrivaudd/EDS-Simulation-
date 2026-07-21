"""SDE models: drift/diffusion coefficients and closed-form references.

GBM is GIVEN (it is the Project 1 model — use it as a template).
Vasicek, CIR and zcb_mc_price are Step 4.

Each model exposes drift(x, t), diffusion(x, t) (and diffusion_dx where
needed) so it can be plugged directly into the generic schemes, plus
closed-form quantities to validate the numerical schemes against.
"""

import numpy as np


class GBM:
    """Geometric Brownian motion:  dX = mu X dt + sigma X dW.   (GIVEN)

    Exact solution (Itô on ln X):
        X_T = x0 * exp[(mu - sigma^2/2) T + sigma W_T]
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def drift(self, x, t):
        return self.mu * x

    def diffusion(self, x, t):
        return self.sigma * x

    def diffusion_dx(self, x, t):
        return self.sigma * np.ones_like(x)

    def exact_terminal(self, x0, T, W_T):
        """Exact X_T given the terminal Brownian value W_T (array)."""
        return x0 * np.exp((self.mu - 0.5 * self.sigma**2) * T
                           + self.sigma * W_T)

    def exact_mean(self, x0, T):
        """E[X_T] = x0 * exp(mu T)."""
        return x0 * np.exp(self.mu * T)


class Vasicek:
    """Vasicek short-rate model:  dr = kappa (theta - r) dt + sigma dW.

    Gaussian Ornstein-Uhlenbeck process. Closed forms to implement:

        E[r_t]   = theta + (r0 - theta) e^{-kappa t}
        Var[r_t] = sigma^2 / (2 kappa) * (1 - e^{-2 kappa t})

    Zero-coupon bond (affine model)  P(0, T) = A(T) e^{-B(T) r0}  with

        B(T) = (1 - e^{-kappa T}) / kappa
        A(T) = exp[ (theta - sigma^2/(2 kappa^2)) (B(T) - T)
                    - sigma^2 B(T)^2 / (4 kappa) ]
    """

    def __init__(self, kappa, theta, sigma):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def drift(self, x, t):
        return self.kappa * (self.theta - x)

    def diffusion(self, x, t):
        return self.sigma * np.ones_like(x)
    def diffusion_dx(self, x, t):
        return np.zeros_like(x)
    
    def exact_mean(self, r0, t):
        return self.theta + (r0 - self.theta) * np.exp(-self.kappa * t)

    def exact_var(self, t):
        # TODO: formula in the class docstring
        raise NotImplementedError("Step 4: Vasicek.exact_var")

    def zcb_price(self, r0, T):
        """Closed-form zero-coupon bond price P(0, T)."""
        # TODO: A(T), B(T) formulas in the class docstring
        raise NotImplementedError("Step 4: Vasicek.zcb_price")


class CIR:
    """Cox-Ingersoll-Ross model:  dr = kappa (theta - r) dt + sigma sqrt(r) dW.

    The square-root diffusion keeps rates non-negative in continuous time
    (strictly positive under the Feller condition 2 kappa theta >= sigma^2),
    but a naive Euler scheme can push r below 0, where sqrt(r) is undefined.
    The standard fix is the FULL TRUNCATION scheme (Lord, Koekkoek &
    van Dijk, 2010): use r+ = max(r, 0) inside both coefficients,

        r_{n+1} = r_n + kappa (theta - r_n^+) dt + sigma sqrt(r_n^+) dW_n.

    Write drift and diffusion directly in full-truncation form (floor x at
    0 inside them) so the class plugs straight into euler_maruyama.

    Closed forms:
        E[r_t] = theta + (r0 - theta) e^{-kappa t}

    Zero-coupon bond  P(0, T) = A(T) e^{-B(T) r0}  with
        gamma = sqrt(kappa^2 + 2 sigma^2)
        B(T)  = 2 (e^{gamma T} - 1) /
                [ (gamma + kappa)(e^{gamma T} - 1) + 2 gamma ]
        A(T)  = [ 2 gamma e^{(kappa + gamma) T / 2} /
                  ( (gamma + kappa)(e^{gamma T} - 1) + 2 gamma ) ]^{2 kappa theta / sigma^2}
    """

    def __init__(self, kappa, theta, sigma):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def drift(self, x, t):
        # TODO: full truncation — use np.maximum(x, 0.0) inside
        raise NotImplementedError("Step 4: CIR.drift")

    def diffusion(self, x, t):
        # TODO: full truncation — sqrt of the FLOORED value
        raise NotImplementedError("Step 4: CIR.diffusion")

    def exact_mean(self, r0, t):
        # TODO: same formula as Vasicek (the drift is identical)
        raise NotImplementedError("Step 4: CIR.exact_mean")

    def feller(self):
        """True if the Feller condition 2 kappa theta >= sigma^2 holds.  (given)"""
        return 2 * self.kappa * self.theta >= self.sigma**2

    def zcb_price(self, r0, T):
        """Closed-form zero-coupon bond price P(0, T)."""
        # TODO: gamma, B(T), A(T) formulas in the class docstring
        raise NotImplementedError("Step 4: CIR.zcb_price")


def zcb_mc_price(r_paths, T):
    """Monte Carlo zero-coupon bond price from simulated short-rate paths.

        P(0, T) = E[ exp( - integral_0^T r_t dt ) ]

    Parameters
    ----------
    r_paths : ndarray (n_paths, n_steps + 1) — simulated short-rate paths.
    T : float — horizon.

    Returns
    -------
    float — Monte Carlo estimate of P(0, T).

    TODO:
    - floor the paths at 0 (full-truncation paths can dip below)
    - integrate each path with the trapezoidal rule
      (np.trapezoid on recent numpy, np.trapz on older — use
       getattr(np, "trapezoid", getattr(np, "trapz", None)))
    - average exp(-integral) over paths, return a float
    """
    raise NotImplementedError("Step 4: implement zcb_mc_price")
