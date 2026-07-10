"""
week7_get_data.py
-------------------
Downloads real daily data for TWO markets, as required by the final
project: SPY (a ticker already used in Week 6 -- allowed, since the brief
permits reusing at most one) and ^NSEI (Nifty 50, India -- a genuinely new
market, directly suggested by the course itself).

Run:
    python week7_get_data.py

Outputs (per market):
    data/<ticker>_prices.csv
    data/<ticker>_returns.npy
    plots/<ticker>_price_and_returns.png
"""

import os
import numpy as np
from week6_get_data import (
    download_prices, compute_log_returns, describe_returns,
    plot_price_and_returns,
)

MARKETS = ["SPY", "^NSEI"]


def safe_name(ticker):
    return ticker.replace("^", "")


def get_market_data(ticker, start="2015-01-01", end="2024-12-31"):
    name = safe_name(ticker)
    close = download_prices(
        ticker, start=start, end=end,
        filename=f"data/{name}_prices.csv",
    )
    returns = compute_log_returns(close)
    print(f"\n--- {ticker} ---")
    stats = describe_returns(returns)
    plot_price_and_returns(
        close, returns,
        filename=f"plots/{name}_price_and_returns.png",
    )
    np.save(f"data/{name}_returns.npy", returns)
    return stats


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    all_stats = {}
    for ticker in MARKETS:
        all_stats[ticker] = get_market_data(ticker)

    print("\n" + "=" * 70)
    print("SUMMARY -- both markets")
    print("=" * 70)
    print(f"{'Ticker':<10} {'Days':>6} {'Mean':>10} {'Std':>10} {'Lag-1 AC':>10}")
    for ticker, s in all_stats.items():
        print(f"{ticker:<10} {s['n_days']:>6} {s['mean']:>+10.5f} "
              f"{s['std']:>10.5f} {s['lag1_autocorr']:>+10.4f}")
