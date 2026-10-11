"""
market_data.py
==============

Loading a saved market snapshot and turning it into model inputs.

A snapshot is a folder in data/ containing:
    options.csv     the option chain (bid, ask, volume, open interest, ...)
    prices.csv      daily closing prices, for historical volatility
    snapshot.json   spot price, interest rate, dividends, sources

Make a fresh one for any ticker with:  python fetch_data.py TICKER
"""

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Snapshot:
    ticker: str
    as_of: date
    spot: float
    r: float                 # continuously compounded risk-free rate
    dividends: list          # [{"ex_date": date, "amount": float}, ...]
    options: pd.DataFrame
    prices: pd.DataFrame
    info: dict


def treasury_yield_to_continuous(y: float, days: int = 91) -> float:
    """Convert a 3-month T-bill yield (simple interest, quoted per year) to the
    continuously compounded rate Black-Scholes expects.

    $1 invested for `days` grows to 1 + y·days/365, and e^(r·t) must match it.
    The difference is tiny (4.24% -> 4.22%) but it's the correct input.
    """
    t = days / 365
    return math.log(1 + y * t) / t


def load_snapshot(folder: str | Path) -> Snapshot:
    folder = Path(folder)
    info = json.loads((folder / "snapshot.json").read_text())
    options = pd.read_csv(folder / "options.csv", dtype={"expiry": str})
    prices = pd.read_csv(folder / "prices.csv", parse_dates=["date"])

    as_of = datetime.strptime(info["snapshot_date"], "%Y-%m-%d").date()
    options["expiry_date"] = pd.to_datetime(options["expiry"], format="%y%m%d").dt.date
    options["mid"] = (options["bid"] + options["ask"]) / 2
    options["option_type"] = options["type"].map({"C": "call", "P": "put"})

    dividends = [
        {"ex_date": datetime.strptime(d["ex_date"], "%Y-%m-%d").date(), "amount": d["amount"]}
        for d in info.get("dividends", [])
    ]
    return Snapshot(
        ticker=info["ticker"],
        as_of=as_of,
        spot=info["spot"],
        r=treasury_yield_to_continuous(info["treasury_3m_yield"]),
        dividends=dividends,
        options=options,
        prices=prices,
        info=info,
    )


def years_between(start: date, end: date) -> float:
    """Calendar days / 365, the usual convention for option time-to-expiry."""
    return (end - start).days / 365


def pv_dividends(snap: Snapshot, expiry: date) -> float:
    """Present value of dividends that go ex BEFORE the option expires.

    On the ex-dividend date the share price drops by roughly the dividend,
    which an option holder doesn't receive. The standard fix ("escrowed
    dividend" model) is to price the option off S minus this amount.
    """
    total = 0.0
    for d in snap.dividends:
        if snap.as_of < d["ex_date"] <= expiry:
            total += d["amount"] * math.exp(-snap.r * years_between(snap.as_of, d["ex_date"]))
    return total


def historical_volatility(prices: pd.DataFrame, window: int = 252) -> float:
    """Annualised standard deviation of daily log returns over the last `window` days.

    252 trading days = 1 year. Daily volatility is scaled up by √252 because
    variance adds up over independent days, so standard deviation grows with √time.
    """
    closes = prices["close"].to_numpy()
    log_returns = np.diff(np.log(closes))[-window:]
    return float(log_returns.std(ddof=1) * math.sqrt(252))
