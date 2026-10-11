"""
fetch_data.py
=============

Download a fresh market snapshot for ANY ticker, in the same format as the
saved AAPL one, so you can re-run the accuracy test on it.

    pip install yfinance
    python fetch_data.py TSLA
    python run_accuracy_test.py data/TSLA_<date>

Data comes from Yahoo Finance via the yfinance library: free, but delayed
and occasionally patchy. Run it while the US market is open (14:30-21:00 UK
time) - outside those hours Yahoo often reports bid/ask as zero, and the
accuracy test filters those options out.

Two expiries are saved: the listed dates closest to 4 weeks and 12 weeks out.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("This script needs yfinance:  pip install yfinance")


def pick_expiries(expiries, today, target_days=(28, 84)):
    """Choose the listed expiry dates closest to each target number of days away."""
    chosen = []
    for target in target_days:
        best = min(expiries, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days - target))
        if best not in chosen:
            chosen.append(best)
    return chosen


def main(ticker: str):
    ticker = ticker.upper()
    tk = yf.Ticker(ticker)

    # --- Price history (1 year) and today's price -------------------------
    hist = tk.history(period="1y", auto_adjust=False)
    if hist.empty:
        sys.exit(f"No price data for {ticker} - check the symbol.")
    as_of = hist.index[-1].date()
    spot = float(hist["Close"].iloc[-1])

    # --- Option chains ----------------------------------------------------
    if not tk.options:
        sys.exit(f"{ticker} has no listed options on Yahoo Finance.")
    expiries = pick_expiries(tk.options, as_of)
    rows = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for kind, df in (("C", chain.calls), ("P", chain.puts)):
            for r in df.itertuples():
                rows.append({
                    "expiry": datetime.strptime(exp, "%Y-%m-%d").strftime("%y%m%d"),
                    "type": kind,
                    "strike": r.strike,
                    "bid": r.bid,
                    "ask": r.ask,
                    "last": r.lastPrice,
                    "volume": 0 if pd.isna(r.volume) else int(r.volume),
                    "open_interest": 0 if pd.isna(r.openInterest) else int(r.openInterest),
                    "source_iv": r.impliedVolatility,
                })

    # --- Risk-free rate: ^IRX is the 13-week US T-bill yield, quoted in percent
    irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
    rate = float(irx.iloc[-1]) / 100

    # --- Dividends: assume the latest one repeats every ~3 months ---------
    dividends = []
    divs = tk.dividends
    if len(divs) and (as_of - divs.index[-1].date()).days < 200:
        amount = float(divs.iloc[-1])
        last_expiry = max(datetime.strptime(e, "%Y-%m-%d").date() for e in expiries)
        ex_date = divs.index[-1].date() + timedelta(days=91)
        while ex_date <= last_expiry:
            if ex_date > as_of:
                dividends.append({"ex_date": ex_date.isoformat(), "amount": amount, "estimated": True})
            ex_date += timedelta(days=91)

    # --- Save ------------------------------------------------------------
    folder = Path(__file__).parent / "data" / f"{ticker}_{as_of}"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "options.csv", index=False)
    pd.DataFrame({"date": [d.date() for d in hist.index], "close": hist["Close"].round(4).values}).to_csv(
        folder / "prices.csv", index=False
    )
    info = {
        "ticker": ticker,
        "snapshot_date": as_of.isoformat(),
        "snapshot_time": f"downloaded {datetime.now():%Y-%m-%d %H:%M} local time",
        "spot": round(spot, 4),
        "treasury_3m_yield": round(rate, 5),
        "dividends": dividends,
        "sources": {"everything": "Yahoo Finance via yfinance; ^IRX for the risk-free rate"},
        "columns": {"source_iv": "Yahoo's own implied volatility (often rough), kept only as a cross-check"},
    }
    (folder / "snapshot.json").write_text(json.dumps(info, indent=2))

    print(f"Saved {len(rows)} options for {ticker} (spot {spot:.2f}, expiries {', '.join(expiries)}) to {folder}")
    print(f"Next:  python run_accuracy_test.py {folder.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python fetch_data.py TICKER")
    main(sys.argv[1])
