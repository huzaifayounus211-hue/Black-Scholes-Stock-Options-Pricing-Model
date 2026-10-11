"""
run_accuracy_test.py
====================

Two questions, answered in order:

  PART 1  Is the CODE right?
          Textbook values, plus three independent numerical methods that
          should all land on the closed-form price.

  PART 2  Is the MODEL right?
          Price real options from a saved market snapshot and compare with
          what they actually traded at.

Usage:
    python run_accuracy_test.py                          # the saved AAPL snapshot
    python run_accuracy_test.py data/TSLA_2026-10-02     # any snapshot from fetch_data.py

Charts and a per-option results table are written to outputs/.
"""

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from black_scholes import EuropeanOption, implied_volatility  # noqa: E402
from market_data import historical_volatility, load_snapshot, pv_dividends, years_between  # noqa: E402
from numerical_methods import binomial_tree_price, crank_nicolson_price, monte_carlo_price  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEFAULT_SNAPSHOT = ROOT / "data" / "AAPL_2026-09-25"

# Liquidity filter for the market test: only options that genuinely trade
MIN_BID = 0.05
MIN_OPEN_INTEREST = 100

# ----------------------------------------------------------------------
# Chart style
# ----------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, WASH = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#ecebe6"


def set_chart_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
    })


def title(fig, main, sub):
    """Chart title and subtitle, a fixed distance from the top whatever the figure height."""
    h = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.18 / h, main, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.46 / h, sub, fontsize=10, color=INK_2, va="top")
    return 1 - 0.78 / h  # top of the plotting area, for tight_layout


def dot(ax, x, y, color, label=None, hollow=False, size=6):
    ax.plot(x, y, "o", ms=size, color=color, mfc="none" if hollow else color,
            mec=color if hollow else SURFACE, mew=1.4 if hollow else 1.2, label=label, zorder=3)


# ======================================================================
# PART 1: IS THE CODE RIGHT?
# ======================================================================
def part1_code_validation(S, K, T, r, sigma):
    print("=" * 72)
    print("PART 1 - IS THE CODE RIGHT?")
    print("=" * 72)

    # Textbook check
    c = EuropeanOption(42, 40, 0.5, 0.10, 0.20, "call").price()
    p = EuropeanOption(42, 40, 0.5, 0.10, 0.20, "put").price()
    print("\nHull's textbook example (S=42, K=40, T=0.5, r=10%, sigma=20%):")
    print(f"  call {c:.4f}  (Hull: 4.76)    put {p:.4f}  (Hull: 0.81)")

    # Same option, four independent routes
    print(f"\nOne option, four methods  (S={S}, K={K}, T={T:.4f}y, r={r:.2%}, sigma={sigma:.1%}):")
    rows = []
    for kind in ("call", "put"):
        t0 = time.perf_counter()
        bs = EuropeanOption(S, K, T, r, sigma, kind).price()
        t_bs = time.perf_counter() - t0
        t0 = time.perf_counter()
        tree = binomial_tree_price(S, K, T, r, sigma, kind, steps=2000)
        t_tree = time.perf_counter() - t0
        t0 = time.perf_counter()
        mc, se = monte_carlo_price(S, K, T, r, sigma, kind, n_paths=1_000_000)
        t_mc = time.perf_counter() - t0
        t0 = time.perf_counter()
        pde = crank_nicolson_price(S, K, T, r, sigma, kind)
        t_pde = time.perf_counter() - t0
        for method, price, secs, extra in [
            ("Closed form (Black-Scholes)", bs, t_bs, ""),
            ("Binomial tree, 2,000 steps", tree, t_tree, ""),
            ("Monte Carlo, 1,000,000 paths", mc, t_mc, f"  (±{2 * se:.4f} at 95%)"),
            ("Crank-Nicolson PDE, 800x800", pde, t_pde, ""),
        ]:
            rows.append(dict(option=kind, method=method, price=price, error=price - bs,
                             pct_error=100 * (price - bs) / bs, seconds=secs))
            print(f"  {kind:<4}  {method:<30} {price:9.4f}   error {price - bs:+.4f} "
                  f"({100 * (price - bs) / bs:+.3f}%)   {secs * 1000:7.1f} ms{extra}")
    pd.DataFrame(rows).to_csv(OUT / "method_comparison.csv", index=False)

    chart_convergence(S, K, T, r, sigma)
    chart_greeks(S, K, T, r, sigma)


def chart_convergence(S, K, T, r, sigma):
    bs = EuropeanOption(S, K, T, r, sigma, "call").price()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    top = title(fig, "Three independent methods converge on the Black-Scholes price",
                f"At-the-money call: S = {S:g}, K = {K:g}, {T * 365:.0f} days, σ = {sigma:.0%}. "
                f"Closed-form price = {bs:.4f} (dark line)")

    # Binomial tree
    ax = axes[0]
    steps = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    tree = [binomial_tree_price(S, K, T, r, sigma, "call", steps=n) for n in steps]
    ax.axhline(bs, color=INK_2, lw=1)
    ax.plot(steps, tree, color=BLUE)
    dot(ax, steps, tree, BLUE)
    ax.set_xscale("log")
    ax.set_title("Binomial tree", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Steps in the tree")
    ax.set_ylabel("Call price ($)")

    # Monte Carlo
    ax = axes[1]
    paths = np.unique(np.logspace(3, 6.3, 23).astype(int))
    est = np.array([monte_carlo_price(S, K, T, r, sigma, "call", n_paths=n, seed=1) for n in paths])
    ax.fill_between(paths, est[:, 0] - 2 * est[:, 1], est[:, 0] + 2 * est[:, 1],
                    color=ORANGE, alpha=0.12, lw=0, label="95% confidence band")
    ax.plot(paths, est[:, 0], color=ORANGE, label="Monte Carlo estimate")
    ax.axhline(bs, color=INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_title("Monte Carlo (antithetic)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Simulated paths")
    ax.legend(loc="upper right", fontsize=9)

    # Crank-Nicolson
    ax = axes[2]
    grid = np.array([25, 50, 100, 200, 400, 800, 1600])
    err = [abs(crank_nicolson_price(S, K, T, r, sigma, "call", n_space=n, n_time=n) - bs) for n in grid]
    ax.plot(grid, err, color=AQUA)
    dot(ax, grid, err, AQUA)
    ax.axhline(0.005, color=INK_2, lw=1)
    ax.text(grid[0], 0.005, " half a cent", color=INK_2, va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Crank-Nicolson PDE solver", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Grid points (price and time)")
    ax.set_ylabel("Absolute error ($, log scale)")

    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "01_convergence.png", dpi=150)
    plt.close(fig)


def chart_greeks(S, K, T, r, sigma):
    spots = np.linspace(0.8 * S, 1.2 * S, 161)
    specs = [
        ("Price ($)", lambda o: o.price()),
        ("Delta  (per $1 move in stock)", lambda o: o.delta()),
        ("Gamma  (change in delta per $1)", lambda o: o.gamma()),
        ("Vega  ($ per 1 vol point)", lambda o: o.vega() / 100),
        ("Theta  ($ per day)", lambda o: o.theta() / 365),
        ("Rho  ($ per 1% rate move)", lambda o: o.rho() / 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    top = title(fig, "The Greeks: how the option's price reacts to each input",
                f"K = {K:g} (grey line), {T * 365:.0f} days to expiry, σ = {sigma:.0%}, r = {r:.2%}, "
                "against the stock price. Gamma and vega are identical for calls and puts.")
    for ax, (name, fn) in zip(axes.flat, specs):
        for kind, color in (("call", BLUE), ("put", ORANGE)):
            vals = [fn(EuropeanOption(s, K, T, r, sigma, kind)) for s in spots]
            ax.plot(spots, vals, color=color, label=kind.title(), lw=2.6 if kind == "call" else 1.8)
        ax.axvline(K, color=AXIS, lw=1)
        ax.set_title(name, loc="left", fontsize=10.5, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Stock price ($)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10, bbox_to_anchor=(0.99, 1 - 0.14 / 7.4))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "02_greeks.png", dpi=150)
    plt.close(fig)


# ======================================================================
# PART 2: IS THE MODEL RIGHT?  (real market data)
# ======================================================================
def part2_market_test(snap):
    print("\n" + "=" * 72)
    print(f"PART 2 - IS THE MODEL RIGHT?  {snap.ticker} options at the close on {snap.as_of}")
    print("=" * 72)
    hv = {w: historical_volatility(snap.prices, w) for w in (21, 63, 252)}
    sigma_hist = hv[252]
    print(f"\nSpot {snap.spot:.2f}   risk-free rate {snap.r:.3%} (continuous)")
    print(f"Historical volatility: 1 month {hv[21]:.1%}   3 months {hv[63]:.1%}   "
          f"1 year {hv[252]:.1%}  <- used for Model A")

    results = []
    for expiry, chain in snap.options.groupby("expiry_date"):
        T = years_between(snap.as_of, expiry)
        div = pv_dividends(snap, expiry)
        S = snap.spot - div          # escrowed-dividend adjustment
        fwd = S * math.exp(snap.r * T)

        # Out-of-the-money options only: calls above the forward, puts below it.
        # These are the ones that actually trade; in-the-money ones are mostly
        # intrinsic value and have wide, stale quotes.
        c = chain.copy()
        otm = ((c.option_type == "call") & (c.strike >= fwd)) | ((c.option_type == "put") & (c.strike < fwd))
        c = c[otm & (c.bid >= MIN_BID) & (c.open_interest >= MIN_OPEN_INTEREST)].sort_values("strike")

        # Implied volatility of every option from its mid price
        c["implied_vol"] = [
            implied_volatility(row.mid, S, row.strike, T, snap.r, row.option_type)
            for row in c.itertuples()
        ]
        # At-the-money vol: read the smile at the forward price
        sigma_atm = float(np.interp(fwd, c.strike, c.implied_vol))

        c["model_A_hist_vol"] = [EuropeanOption(S, k, T, snap.r, sigma_hist, t).price()
                                 for k, t in zip(c.strike, c.option_type)]
        c["model_B_flat_atm_vol"] = [EuropeanOption(S, k, T, snap.r, sigma_atm, t).price()
                                     for k, t in zip(c.strike, c.option_type)]
        # How much would American-style early exercise add? (listed stock options are American)
        c["early_exercise_premium"] = [
            binomial_tree_price(S, k, T, snap.r, iv, t, steps=400, american=True)
            - binomial_tree_price(S, k, T, snap.r, iv, t, steps=400)
            for k, t, iv in zip(c.strike, c.option_type, c.implied_vol)
        ]
        c["T_years"], c["spot_used"], c["sigma_hist"], c["sigma_atm"] = T, S, sigma_hist, sigma_atm
        results.append(c)

        print(f"\nExpiry {expiry}  ({T * 365:.0f} days, {len(c)} liquid out-of-the-money options, "
              f"dividends before expiry ${div:.2f})")
        print(f"  At-the-money implied vol {sigma_atm:.1%}.  Model A uses {sigma_hist:.1%}, "
              f"Model B uses {sigma_atm:.1%} for every strike.")
        print(f"  {'':34}{'mean abs error':>16}{'median % error':>16}{'inside bid-ask':>16}")
        for col, name in (("model_A_hist_vol", "Model A: 1-year historical vol"),
                          ("model_B_flat_atm_vol", "Model B: flat at-the-money vol")):
            err = c[col] - c.mid
            inside = ((c[col] >= c.bid) & (c[col] <= c.ask)).mean()
            print(f"  {name:<34}{'$' + format(err.abs().mean(), '.3f'):>16}"
                  f"{format((100 * err.abs() / c.mid).median(), '.1f') + '%':>16}{inside:>16.0%}")
        if "source_iv" in c and (c.source_iv > 0).any():
            ok = c.source_iv > 0
            diff = 100 * (c.implied_vol[ok] - c.source_iv[ok])
            print(f"  Implied vol cross-check vs data provider: mean abs difference "
                  f"{diff.abs().mean():.2f} vol points (max {diff.abs().max():.2f})")
        print(f"  Largest early-exercise premium in this set: ${c.early_exercise_premium.max():.3f}")

    res = pd.concat(results)
    keep = ["expiry_date", "option_type", "strike", "bid", "ask", "mid", "volume", "open_interest",
            "implied_vol", "source_iv", "model_A_hist_vol", "model_B_flat_atm_vol",
            "early_exercise_premium", "T_years", "spot_used", "sigma_hist", "sigma_atm"]
    res[[k for k in keep if k in res]].round(5).to_csv(OUT / f"{snap.ticker.lower()}_results.csv", index=False)
    chart_pricing_errors(snap, res)
    chart_smile(snap, res, hv)
    return res


def chart_pricing_errors(snap, res):
    expiries = sorted(res.expiry_date.unique())
    fig, axes = plt.subplots(1, len(expiries), figsize=(6.5 * len(expiries), 4.8), squeeze=False)
    sig_h = res.sigma_hist.iloc[0]
    top = title(fig, f"How far Black-Scholes is from real {snap.ticker} option prices",
                f"Model price minus market mid-price at the close on {snap.as_of:%d %b %Y}. Above zero = "
                "model too expensive. Inside the grey band = within the bid-ask spread.")
    for ax, exp in zip(axes[0], expiries):
        c = res[res.expiry_date == exp].sort_values("strike")
        ax.fill_between(c.strike, c.bid - c.mid, c.ask - c.mid, color=WASH, lw=0, label="Bid-ask spread")
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(c.strike, c.model_A_hist_vol - c.mid, color=ORANGE,
                label=f"Model A: 1-year historical vol ({sig_h:.1%})")
        dot(ax, c.strike, c.model_A_hist_vol - c.mid, ORANGE, size=5)
        ax.plot(c.strike, c.model_B_flat_atm_vol - c.mid, color=BLUE,
                label=f"Model B: flat at-the-money vol ({c.sigma_atm.iloc[0]:.1%})")
        dot(ax, c.strike, c.model_B_flat_atm_vol - c.mid, BLUE, size=5)
        ax.axvline(snap.spot, color=AXIS, lw=1)
        ax.text(snap.spot, ax.get_ylim()[1], f" spot ${snap.spot:.2f}", color=INK_2, fontsize=9, va="top")
        ax.set_title(f"Expiry {exp:%d %b %Y} ({c.T_years.iloc[0] * 365:.0f} days)  ·  puts left of spot, "
                     "calls right", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Strike ($)")
        ax.set_ylabel("Model − market ($)")
        ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "03_model_vs_market.png", dpi=150)
    plt.close(fig)


def chart_smile(snap, res, hv):
    expiries = sorted(res.expiry_date.unique())
    fig, axes = plt.subplots(1, len(expiries), figsize=(6.5 * len(expiries), 4.8), squeeze=False,
                             sharey=True)
    top = title(fig, "The volatility smile: the market doesn't use one σ for every strike",
                "Black-Scholes assumes volatility is constant. If it were, each panel would be a flat line.")
    for ax, exp in zip(axes[0], expiries):
        c = res[res.expiry_date == exp].sort_values("strike")
        ax.axhline(100 * hv[252], color=MUTED, lw=1.2)
        ax.text(c.strike.max(), 100 * hv[252], f"1-year historical vol {hv[252]:.1%}", color=INK_2,
                fontsize=9, ha="right", va="bottom")
        ax.plot(c.strike, 100 * c.implied_vol, color=BLUE)
        dot(ax, c.strike, 100 * c.implied_vol, BLUE, label="Implied vol, this project's solver")
        if "source_iv" in c:
            ok = c.source_iv > 0
            dot(ax, c.strike[ok], 100 * c.source_iv[ok], ORANGE, hollow=True, size=8,
                label="Implied vol published by Cboe (cross-check)")
        ax.axvline(snap.spot, color=AXIS, lw=1)
        ax.set_title(f"Expiry {exp:%d %b %Y} ({c.T_years.iloc[0] * 365:.0f} days)", loc="left",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("Strike ($)")
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0, 0].set_ylabel("Implied volatility (%)")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "04_volatility_smile.png", dpi=150)
    plt.close(fig)


# ======================================================================
def main():
    OUT.mkdir(exist_ok=True)
    set_chart_style()
    snap = load_snapshot(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAPSHOT)

    # Part 1 uses a realistic option from the snapshot: the nearest expiry, strike nearest spot
    first_expiry = min(snap.options.expiry_date)
    strikes = snap.options.strike.unique()
    K = float(strikes[np.argmin(abs(strikes - snap.spot))])
    T = years_between(snap.as_of, first_expiry)
    part1_code_validation(snap.spot, K, T, snap.r, sigma=0.21)

    part2_market_test(snap)
    print(f"\nCharts and results saved to {OUT}/")


if __name__ == "__main__":
    main()
