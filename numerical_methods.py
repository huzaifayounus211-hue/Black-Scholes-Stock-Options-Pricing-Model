"""
numerical_methods.py
====================

Three independent ways to price the same option WITHOUT using the closed-form
formula. If all three land on the Black-Scholes answer, that's strong evidence
the formula (and our code for it) is right.

    binomial_tree_price()     Cox-Ross-Rubinstein tree  (discrete random walk)
    monte_carlo_price()       simulate thousands of possible futures, average the payoff
    crank_nicolson_price()    solve the Black-Scholes PDE directly on a grid

All three take the same inputs as EuropeanOption: S, K, T, r, sigma, option_type, q.
"""

import math

import numpy as np
from scipy.linalg import solve_banded


def payoff(S_T, K: float, option_type: str):
    """What the option is worth at expiry. Works on single numbers or numpy arrays."""
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


# ----------------------------------------------------------------------
# 1. Binomial tree (Cox, Ross & Rubinstein, 1979)
# ----------------------------------------------------------------------
def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: int = 500,
    american: bool = False,
) -> float:
    """Price an option on a recombining binomial tree.

    Each step the stock goes up by a factor u or down by d = 1/u. We use the
    risk-neutral probability p (the one that makes the stock earn exactly the
    risk-free rate on average), work out the payoff at the end of the tree,
    then step backwards, discounting as we go.

    As steps -> infinity the tree converges to Black-Scholes.
    Set american=True to allow early exercise (Black-Scholes can't do this).
    """
    option_type = option_type.lower()
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)  # risk-neutral up-probability
    disc = math.exp(-r * dt)

    # Stock prices at expiry: S·u^j·d^(steps-j) for j = 0..steps up-moves
    j = np.arange(steps + 1)
    S_T = S * u**j * d ** (steps - j)
    values = payoff(S_T, K, option_type)

    # Walk backwards through the tree
    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            S_now = S * u**j * d ** (step - j)
            values = np.maximum(values, payoff(S_now, K, option_type))
    return float(values[0])


# ----------------------------------------------------------------------
# 2. Monte Carlo simulation
# ----------------------------------------------------------------------
def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 200_000,
    antithetic: bool = True,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Price by simulating the stock under geometric Brownian motion.

    Under the risk-neutral measure the stock follows
        dS = (r - q)·S·dt + σ·S·dW
    which has the exact solution
        S_T = S·exp[(r - q - σ²/2)·T + σ·√T·Z],   Z ~ N(0, 1)
    so for a European option we can jump straight to expiry in one step.

    Antithetic variates: for every random draw Z we also use -Z. The two
    paths are mirror images, their errors partly cancel, and the estimate
    gets more accurate for free.

    Returns (price, standard_error). The true price should lie within about
    ±2 standard errors of the estimate 95% of the time.
    """
    option_type = option_type.lower()
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        pay_up = payoff(S * np.exp(drift + vol * Z), K, option_type)
        pay_down = payoff(S * np.exp(drift - vol * Z), K, option_type)
        samples = disc * 0.5 * (pay_up + pay_down)  # one sample per antithetic PAIR
    else:
        Z = rng.standard_normal(n_paths)
        samples = disc * payoff(S * np.exp(drift + vol * Z), K, option_type)

    price = samples.mean()
    std_error = samples.std(ddof=1) / math.sqrt(len(samples))
    return float(price), float(std_error)


# ----------------------------------------------------------------------
# 3. Finite-difference PDE solver (Crank-Nicolson)
# ----------------------------------------------------------------------
def crank_nicolson_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_space: int = 800,
    n_time: int = 800,
    s_max_multiple: float | None = None,
) -> float:
    """Solve the Black-Scholes PDE on a grid of (stock price, time).

    The PDE, written in τ = time remaining until expiry:
        ∂V/∂τ = ½σ²S²·∂²V/∂S² + (r - q)·S·∂V/∂S - r·V

    We start from the known payoff at τ = 0 (expiry) and march backwards to
    τ = T (today). Crank-Nicolson averages the explicit and implicit schemes,
    which makes it both stable and second-order accurate. The first few steps
    use the fully implicit scheme ("Rannacher smoothing") to damp the wiggles
    the kink in the payoff would otherwise cause.

    The grid is built so today's stock price S sits exactly on a grid node,
    so no interpolation is needed to read off the answer. By default the top
    of the grid is set about 5 standard deviations above today's price (and
    at least twice the strike), so no grid points are wasted on prices the
    stock can't realistically reach.
    """
    option_type = option_type.lower()
    if s_max_multiple is None:
        s_max_multiple = max(2.0 * max(1.0, K / S), math.exp(5 * sigma * math.sqrt(T)))
    nodes_below_spot = int(round(n_space / s_max_multiple))
    dS = S / nodes_below_spot
    M = n_space                              # grid runs from 0 to M·dS
    S_grid = np.arange(M + 1) * dS
    dtau = T / n_time
    if K > 0.8 * S_grid[-1]:
        raise ValueError("strike too far above spot for this grid; raise s_max_multiple")

    # Coefficients of the discretised right-hand side, for interior nodes i = 1..M-1
    i = np.arange(1, M)
    alpha = 0.5 * sigma**2 * i**2 - 0.5 * (r - q) * i   # multiplies V[i-1]
    beta = -(sigma**2) * i**2 - r                        # multiplies V[i]
    gamma = 0.5 * sigma**2 * i**2 + 0.5 * (r - q) * i   # multiplies V[i+1]

    def boundary(tau: float) -> tuple[float, float]:
        """Option value at S = 0 and at S = S_max, with tau years left."""
        if option_type == "call":
            return 0.0, S_grid[-1] * math.exp(-q * tau) - K * math.exp(-r * tau)
        return K * math.exp(-r * tau), 0.0

    V = payoff(S_grid, K, option_type).astype(float)
    n_smoothing = 4  # fully implicit steps at the start

    for n in range(n_time):
        theta = 1.0 if n < n_smoothing else 0.5  # 1 = implicit, 0.5 = Crank-Nicolson
        tau_new = (n + 1) * dtau
        lo_new, hi_new = boundary(tau_new)

        # Left-hand matrix (I - θ·dτ·L), stored in banded form for solve_banded
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = -theta * dtau * gamma[:-1]      # upper diagonal
        ab[1, :] = 1 - theta * dtau * beta          # main diagonal
        ab[2, :-1] = -theta * dtau * alpha[1:]      # lower diagonal

        # Right-hand side (I + (1-θ)·dτ·L)·V_old, plus boundary contributions
        w = (1 - theta) * dtau
        rhs = V[1:-1] + w * (alpha * V[:-2] + beta * V[1:-1] + gamma * V[2:])
        rhs[0] += theta * dtau * alpha[0] * lo_new
        rhs[-1] += theta * dtau * gamma[-1] * hi_new

        V[1:-1] = solve_banded((1, 1), ab, rhs)
        V[0], V[-1] = lo_new, hi_new

    return float(V[nodes_below_spot])
