"""sde_lab — numerical simulation of stochastic differential equations.

Modules:
    schemes      Generic Euler-Maruyama and Milstein discretization schemes.
    models       SDE models: GBM, Vasicek, CIR (drift/diffusion + closed forms).
    convergence  Strong/weak error measurement and order estimation.
    heston       Heston stochastic-volatility model and call pricer.
"""

from .schemes import euler_maruyama, milstein
from .models import GBM, Vasicek, CIR
from .convergence import strong_error, weak_error, estimate_order
from .heston import Heston

__all__ = [
    "euler_maruyama",
    "milstein",
    "GBM",
    "Vasicek",
    "CIR",
    "strong_error",
    "weak_error",
    "estimate_order",
    "Heston",
]
