"""
black_scholes.py
================

The closed-form Black-Scholes-Merton model for European options.

    Call:  C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    Put:   P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

    d1 = [ ln(S/K) + (r - q + σ²/2)·T ] / (σ·√T)
    d2 = d1 - σ·√T

S = spot price, K = strike, T = years to expiry, r = risk-free rate
(continuously compounded), σ = volatility (annualised), q = dividend yield
(continuous; leave it at 0 for the textbook version of the formula).
N(·) is the standard normal cumulative distribution function.

This file contains:
    EuropeanOption        the pricer: price + all five Greeks
    implied_volatility()  the reverse problem: market price -> σ
    put_call_parity_gap() a sanity check that calls and puts agree
"""

from dataclasses import dataclass
import math

from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class EuropeanOption:
    """A European call or put, priced with Black-Scholes.

    Example
    -------
    >>> opt = EuropeanOption(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    >>> round(opt.price(), 2)
    4.76
    """

    S: float                  # spot price of the stock
    K: float                  # strike price
    T: float                  # time to expiry, in years
    r: float                  # risk-free interest rate (0.05 = 5%)
    sigma: float              # volatility (0.20 = 20% a year)
    option_type: str = "call" # "call" or "put"
    q: float = 0.0            # continuous dividend yield (0.01 = 1%)

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive (use the payoff for an expired option)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    # ------------------------------------------------------------------
    # The two building blocks, d1 and d2
    # ------------------------------------------------------------------
    @property
    def d1(self) -> float:
        return (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * math.sqrt(self.T)
        )

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * math.sqrt(self.T)

    # Discount factors used everywhere below
    @property
    def _disc_r(self) -> float:  # value today of $1 paid at expiry
        return math.exp(-self.r * self.T)

    @property
    def _disc_q(self) -> float:  # adjusts the stock for dividends paid before expiry
        return math.exp(-self.q * self.T)

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------
    def price(self) -> float:
        """Fair value of the option today."""
        S, K = self.S, self.K
        d1, d2 = self.d1, self.d2
        if self.option_type == "call":
            return float(S * self._disc_q * norm.cdf(d1) - K * self._disc_r * norm.cdf(d2))
        return float(K * self._disc_r * norm.cdf(-d2) - S * self._disc_q * norm.cdf(-d1))

    # ------------------------------------------------------------------
    # The Greeks: how the price responds when one input moves
    # ------------------------------------------------------------------
    def delta(self) -> float:
        """∂V/∂S — change in option price per $1 move in the stock (also the hedge ratio)."""
        if self.option_type == "call":
            return self._disc_q * norm.cdf(self.d1)
        return self._disc_q * (norm.cdf(self.d1) - 1.0)

    def gamma(self) -> float:
        """∂²V/∂S² — how fast delta changes. Same for calls and puts."""
        return self._disc_q * norm.pdf(self.d1) / (self.S * self.sigma * math.sqrt(self.T))

    def vega(self) -> float:
        """∂V/∂σ — change in price per 1.00 (i.e. 100 percentage points) of volatility.
        Divide by 100 to get the change per 1 vol point. Same for calls and puts."""
        return self.S * self._disc_q * norm.pdf(self.d1) * math.sqrt(self.T)

    def theta(self) -> float:
        """∂V/∂t — change in price per YEAR of time passing (usually negative).
        Divide by 365 to get the daily time decay."""
        S, K, r, q, sig, T = self.S, self.K, self.r, self.q, self.sigma, self.T
        d1, d2 = self.d1, self.d2
        decay = -S * self._disc_q * norm.pdf(d1) * sig / (2 * math.sqrt(T))
        if self.option_type == "call":
            return decay - r * K * self._disc_r * norm.cdf(d2) + q * S * self._disc_q * norm.cdf(d1)
        return decay + r * K * self._disc_r * norm.cdf(-d2) - q * S * self._disc_q * norm.cdf(-d1)

    def rho(self) -> float:
        """∂V/∂r — change in price per 1.00 (100 percentage points) of interest rate.
        Divide by 100 to get the change per 1% move in rates."""
        if self.option_type == "call":
            return self.K * self.T * self._disc_r * norm.cdf(self.d2)
        return -self.K * self.T * self._disc_r * norm.cdf(-self.d2)

    def greeks(self) -> dict:
        """All five Greeks, in 'trader units' (per $1, per vol point, per day, per 1% rate)."""
        return {
            "delta": float(self.delta()),
            "gamma": float(self.gamma()),
            "vega_per_vol_point": float(self.vega() / 100),
            "theta_per_day": float(self.theta() / 365),
            "rho_per_1pct": float(self.rho() / 100),
        }

    def __str__(self) -> str:
        return (
            f"{self.option_type.title()} K={self.K:g}, S={self.S:g}, T={self.T:.4f}y, "
            f"r={self.r:.2%}, σ={self.sigma:.2%}, q={self.q:.2%} -> price {self.price():.4f}"
        )


# ----------------------------------------------------------------------
# Implied volatility: run the formula backwards
# ----------------------------------------------------------------------
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Find the σ that makes the Black-Scholes price equal the market price.

    Uses Newton-Raphson first (fast: σ_new = σ - (price(σ) - market) / vega),
    and falls back to Brent's method (slower, but can't diverge) if Newton
    misbehaves, which happens for deep in/out-of-the-money options where
    vega is almost zero.

    Raises ValueError if the price is outside the range any σ could produce
    (that would be an arbitrage, or more often a stale/bad quote).
    """
    option_type = option_type.lower()
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)

    # No-arbitrage bounds: the price at σ→0 and at σ→∞
    if option_type == "call":
        lower, upper = max(0.0, S * disc_q - K * disc_r), S * disc_q
    else:
        lower, upper = max(0.0, K * disc_r - S * disc_q), K * disc_r
    if not (lower < market_price < upper):
        raise ValueError(
            f"price {market_price:.4f} is outside the no-arbitrage range ({lower:.4f}, {upper:.4f})"
        )

    def bs_price(sig: float) -> float:
        return EuropeanOption(S, K, T, r, sig, option_type, q).price()

    # --- Newton-Raphson ---------------------------------------------
    # Starting guess (Brenner & Subrahmanyam, 1988): good near the money
    sigma = math.sqrt(2 * math.pi / T) * market_price / S
    sigma = min(max(sigma, 0.05), 2.0)
    for _ in range(max_iter):
        opt = EuropeanOption(S, K, T, r, sigma, option_type, q)
        diff = opt.price() - market_price
        if abs(diff) < tol:
            return sigma
        v = opt.vega()
        if v < 1e-8:
            break  # vega too flat to trust a Newton step
        sigma -= diff / v
        if not (1e-4 < sigma < 5.0):
            break  # jumped somewhere silly, hand over to Brent

    # --- Brent's method fallback -------------------------------------
    return brentq(lambda sig: bs_price(sig) - market_price, 1e-4, 5.0, xtol=tol, maxiter=500)


def put_call_parity_gap(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """C - P - (S·e^(-qT) - K·e^(-rT)). Should be zero to machine precision."""
    call = EuropeanOption(S, K, T, r, sigma, "call", q).price()
    put = EuropeanOption(S, K, T, r, sigma, "put", q).price()
    return call - put - (S * math.exp(-q * T) - K * math.exp(-r * T))


if __name__ == "__main__":
    # Quick demo: the classic worked example from Hull, Options, Futures and Other
    # Derivatives (Black-Scholes-Merton chapter). Hull gets c = 4.76 and p = 0.81.
    for kind in ("call", "put"):
        opt = EuropeanOption(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type=kind)
        print(opt)
        for name, value in opt.greeks().items():
            print(f"    {name:<20} {value: .4f}")
