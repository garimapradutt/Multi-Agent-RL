"""
week6_get_data.py
------------------
Downloads real daily price data, computes log returns, and inspects their
statistical properties -- including the lag-1 autocorrelation we learned to
measure in Week 5. This is the moment we find out whether real markets look
more like rho=0.5 (Week 5's momentum market) or rho=0.0 (a random walk).

Run:
    python week6_get_data.py

Outputs:
    data/prices.csv
    data/returns.npy
    plots/price_and_returns.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_prices(ticker="SPY", start="2015-01-01",
                     end="2024-12-31", filename="data/prices.csv"):
    os.makedirs("data", exist_ok=True)
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    # auto_adjust=True folds dividends and stock splits back into the price
    # series, so e.g. a 2-for-1 split doesn't show up as a fake -50% crash
    # in the return series -- without it, corporate actions would look
    # exactly like real (but bogus) price moves.
    close = data["Close"].squeeze().dropna()  # a pandas Series
    close.to_csv(filename)
    print(f"Saved {len(close)} rows to {filename}")
    return close


def load_prices(filename="data/prices.csv"):
    """Fallback loader: works with any CSV whose first column is a date
    and second column is a closing price."""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df.iloc[:, 0].dropna()


def compute_log_returns(close):
    """
    np.diff(np.log(prices)) computes r_t = ln(P_t) - ln(P_t-1) = ln(P_t/P_t-1)
    -- the log return at each step. Log returns (rather than raw price
    differences or simple percentage changes) are used because they ADD
    across time: the total log return over N days is just the sum of the
    daily log returns, which makes cumulative P&L calculations clean and
    consistent with how the reward is computed elsewhere in this project
    (position * return * 100, summed over an episode).
    """
    prices = close.to_numpy(dtype=float)
    return np.diff(np.log(prices))


def describe_returns(returns):
    lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    print(f"Number of days:        {len(returns)}")
    print(f"Mean daily return:     {returns.mean():+.5f}")
    print(f"Std of daily returns:  {returns.std():.5f}")
    print(f"Lag-1 autocorrelation: {lag1:+.4f}")
    return {
        "n_days": len(returns),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "lag1_autocorr": float(lag1),
    }


def plot_price_and_returns(close, returns,
                            filename="plots/price_and_returns.png"):
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    ax1.plot(close.to_numpy(dtype=float), color="steelblue")
    ax1.set_ylabel("Close price")
    ax2.plot(returns, color="darkorange", linewidth=0.5)
    ax2.set_ylabel("Daily log return")
    ax2.set_xlabel("Trading day")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    close = download_prices("SPY")
    returns = compute_log_returns(close)
    stats = describe_returns(returns)
    plot_price_and_returns(close, returns)
    np.save("data/returns.npy", returns)

    print(f"\nComparison to synthetic environments (daily_vol=0.01):")
    print(f"  Real SPY std={stats['std']:.5f} vs synthetic daily_vol=0.01000")
    print(f"  Real SPY lag-1 autocorr={stats['lag1_autocorr']:+.4f} "
          f"vs Week 5's rho=0.5 momentum market")
