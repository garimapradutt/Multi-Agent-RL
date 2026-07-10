"""
week7_metrics.py
-----------------
Two risk-aware evaluation metrics, on top of final cumulative P&L:
Sharpe ratio (reward per unit of risk) and maximum drawdown (worst
peak-to-trough losing stretch).
"""

import numpy as np


def sharpe_ratio(step_rewards, periods_per_year=252):
    """Annualised Sharpe ratio of a sequence of per-step rewards.
    252 = trading days in a year (daily data)."""
    r = np.asarray(step_rewards, dtype=float)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(step_rewards):
    """Largest peak-to-trough decline of the cumulative P&L curve,
    in the same units as the reward."""
    equity = np.cumsum(np.asarray(step_rewards, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())
