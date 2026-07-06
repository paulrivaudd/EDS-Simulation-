"""sde_lab — numerical simulation of stochastic differential equations.

Modules (fill them in, in this order):
    schemes      Step 1 & 2: Euler-Maruyama and Milstein schemes.
    convergence  Step 3: strong/weak error and order estimation.
    models       Step 4: Vasicek and CIR short-rate models (GBM is given).
    heston       Step 5: Heston stochastic-volatility model.
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
