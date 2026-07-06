# SDE Lab — Solution

Complete reference implementation. **Only look at this if you are stuck** — the
assignment lives in `../assignment/`.

## Contents

| Module | What it does |
|---|---|
| `sde_lab/schemes.py` | Generic Euler-Maruyama and Milstein schemes for `dX = a(X,t) dt + b(X,t) dW` |
| `sde_lab/models.py` | GBM, Vasicek, CIR (coefficients + closed forms), MC zero-coupon bond pricer |
| `sde_lab/convergence.py` | Strong/weak error, empirical order estimation (log-log regression) |
| `sde_lab/heston.py` | Heston model, full-truncation Euler + log-Euler asset step, MC call pricer |

## Install & run

```bash
pip install -e ".[dev]"
pytest -v
python examples/demo.py
```

## Key results reproduced by the tests

- Euler-Maruyama: strong order ≈ 0.5, weak order ≈ 1.0
- Milstein: strong order ≈ 1.0 (one extra term, twice the order)
- Vasicek/CIR Monte Carlo ZCB prices match the affine closed forms
- Heston with `xi = 0, v0 = theta` collapses to Black-Scholes
- `rho < 0` produces the equity left skew (OTM puts more expensive)
