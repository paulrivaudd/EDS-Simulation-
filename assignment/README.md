# Project 2 — Simulating Stochastic Differential Equations

Follow-up to the Monte Carlo pricer. In Project 1 you priced options with the
EXACT solution of GBM. Most SDEs have no exact solution — this project is
about the discretization schemes you use instead, how fast they converge, and
what they unlock (interest-rate models, stochastic volatility).

You work in this folder. The full solution lives in `../solution/` — only look
at it if you are stuck.

## Setup

```bash
pip install -e ".[dev]"
pytest            # everything fails with NotImplementedError — that's the starting point
```

Validate each step with its test file before moving on.

## Step 1 — Euler-Maruyama (`sde_lab/schemes.py`)

The workhorse scheme. For `dX = a(X,t) dt + b(X,t) dW`, freeze the
coefficients on each interval of size `h = T/n`:

```
X_{n+1} = X_n + a(X_n, t_n) h + b(X_n, t_n) dW_n,    dW_n ~ N(0, h)
```

Implement `euler_maruyama`. Vectorize over paths (single loop over time
steps), and honor the optional `dW` argument — Step 3 depends on it.

```bash
pytest tests/test_schemes.py -k "shapes or euler" -v
```

## Step 2 — Milstein (`sde_lab/schemes.py`)

One Itô-Taylor term further:

```
X_{n+1} = X_n + a h + b dW_n + 0.5 b b' (dW_n² - h),    b' = ∂b/∂x
```

That single correction doubles the strong order (0.5 → 1). Implement
`milstein`, then:

```bash
pytest tests/test_schemes.py -v
```

## Step 3 — Convergence study (`sde_lab/convergence.py`)

Quantify what "order" means:

- strong error `E|X_T^h − X_T|` — pathwise closeness (needs the SAME
  Brownian increments for scheme and exact solution: sum consecutive fine
  increments to get the coarse ones),
- weak error `|E X_T^h − E X_T|` — closeness of expectations,
- empirical order = slope of log(error) vs log(h) (least squares).

Implement `strong_error`, `weak_error`, `estimate_order`, then:

```bash
pytest tests/test_convergence.py -v
```

Expected: Euler strong ≈ 0.5, Milstein strong ≈ 1.0, Euler weak ≈ 1.0.

## Step 4 — Short-rate models (`sde_lab/models.py`)

Two classic models with no need for a new scheme — your generic
`euler_maruyama` handles both:

- **Vasicek** `dr = κ(θ − r) dt + σ dW` — Gaussian, closed-form mean,
  variance and zero-coupon bond price (formulas in the docstrings).
- **CIR** `dr = κ(θ − r) dt + σ√r dW` — the √r kills naive Euler (negative
  rates → NaN). Implement the **full-truncation** fix: floor r at 0 inside
  both coefficients.

Also implement `zcb_mc_price` (Monte Carlo bond price
`E[exp(−∫r dt)]`, trapezoidal rule) and check it against the affine
closed forms:

```bash
pytest tests/test_rates.py -v
```

## Step 5 — Heston (`sde_lab/heston.py`)

The real thing: stochastic volatility, two correlated Brownian motions,
no exact simulation.

```
dS = r S dt + √v S dW^S
dv = κ(θ − v) dt + ξ √v dW^v,      corr(dW^S, dW^v) = ρ dt
```

Implement `Heston.simulate` (full-truncation Euler on v + log-Euler on S,
recipe in the module docstring) and `Heston.call_price`. The tests check
real financial structure: martingale property, put-call parity, collapse
to Black-Scholes when ξ = 0, and the left skew created by ρ < 0.

```bash
pytest tests/test_heston.py -v
```

## Done?

```bash
pytest             # 21 passed
python examples/demo.py
```

The demo prints your convergence table, bond prices vs closed forms, and a
Heston-vs-Black-Scholes strike ladder — the numerical signature of the
volatility smile.
