"""SDE models: drift/diffusion coefficients and closed-form references.

Each model exposes
    drift(x, t), diffusion(x, t), diffusion_dx(x, t)
so it can be plugged directly into the generic schemes, plus whatever
closed-form quantities exist (exact terminal law, moments, bond prices)
to validate the numerical schemes against.
"""

import numpy as np


class GBM:
    """Geometric Brownian motion:  dX = mu X dt + sigma X dW.

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

    Gaussian Ornstein-Uhlenbeck process. Closed forms:
        E[r_t]   = theta + (r0 - theta) e^{-kappa t}
        Var[r_t] = sigma^2 / (2 kappa) * (1 - e^{-2 kappa t})
        Zero-coupon bond P(0, T) = A(T) e^{-B(T) r0}   (affine model)
    """

    def __init__(self, kappa, theta, sigma):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def drift(self, x, t):
        return self.kappa * (self.theta - x)

    def diffusion(self, x, t):
        return self.sigma * np.ones_like(np.asarray(x, dtype=float))

    def diffusion_dx(self, x, t):
        return np.zeros_like(np.asarray(x, dtype=float))

    def exact_mean(self, r0, t):
        return self.theta + (r0 - self.theta) * np.exp(-self.kappa * t)

    def exact_var(self, t):
        return self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * t))

    def zcb_price(self, r0, T):
        """Closed-form zero-coupon bond price P(0, T)."""
        k, th, s = self.kappa, self.theta, self.sigma
        B = (1 - np.exp(-k * T)) / k
        A = np.exp((th - s**2 / (2 * k**2)) * (B - T) - s**2 * B**2 / (4 * k))
        return A * np.exp(-B * r0)


class CIR:
    """Cox-Ingersoll-Ross model:  dr = kappa (theta - r) dt + sigma sqrt(r) dW.

    The square-root diffusion keeps rates non-negative in continuous time
    (strictly positive under the Feller condition 2 kappa theta >= sigma^2),
    but a naive Euler scheme can push r below 0, where sqrt(r) is undefined.
    The standard fix is the FULL TRUNCATION scheme (Lord, Koekkoek &
    van Dijk, 2010): use r+ = max(r, 0) inside both coefficients,

        r_{n+1} = r_n + kappa (theta - r_n^+) dt + sigma sqrt(r_n^+) dW_n.

    Closed forms:
        E[r_t] = theta + (r0 - theta) e^{-kappa t}
        Zero-coupon bond P(0, T) = A(T) e^{-B(T) r0}   (affine model)
    """

    def __init__(self, kappa, theta, sigma):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    # Coefficients already written in "full truncation" form: they accept
    # any real x and internally floor it at 0, so they can be fed straight
    # into the generic euler_maruyama scheme.
    def drift(self, x, t):
        return self.kappa * (self.theta - np.maximum(x, 0.0))

    def diffusion(self, x, t):
        return self.sigma * np.sqrt(np.maximum(x, 0.0))

    def exact_mean(self, r0, t):
        return self.theta + (r0 - self.theta) * np.exp(-self.kappa * t)

    def feller(self):
        """True if the Feller condition 2 kappa theta >= sigma^2 holds."""
        return 2 * self.kappa * self.theta >= self.sigma**2

    def zcb_price(self, r0, T):
        """Closed-form zero-coupon bond price P(0, T)."""
        k, th, s = self.kappa, self.theta, self.sigma
        gamma = np.sqrt(k**2 + 2 * s**2)
        e = np.exp(gamma * T) - 1
        denom = (gamma + k) * e + 2 * gamma
        B = 2 * e / denom
        A = (2 * gamma * np.exp((k + gamma) * T / 2) / denom) ** (2 * k * th / s**2)
        return A * np.exp(-B * r0)


def zcb_mc_price(r_paths, T):
    """Monte Carlo zero-coupon bond price from simulated short-rate paths.

        P(0, T) = E[ exp( - integral_0^T r_t dt ) ]

    The integral is approximated with the trapezoidal rule along each path.

    Parameters
    ----------
    r_paths : ndarray (n_paths, n_steps + 1) — simulated short-rate paths.
              Negative values (possible under full-truncation Euler) are
              floored at 0 before integrating.
    T : float — horizon.

    Returns
    -------
    float — Monte Carlo estimate of P(0, T).
    """
    r = np.maximum(r_paths, 0.0)
    n_steps = r.shape[1] - 1
    dt = T / n_steps
    trapezoid = getattr(np, "trapezoid", np.trapz)  # numpy 2.x renamed trapz
    integral = trapezoid(r, dx=dt, axis=1)
    return float(np.mean(np.exp(-integral)))
