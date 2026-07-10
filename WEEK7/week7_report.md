# The Honest Backtest: An RL Trading System on Two Real Markets

## 1. Introduction

This project asks a single question: **does adding richer state features and finer-grained position sizing to a PPO trading agent produce a more profitable, or safer, trading strategy than a simpler baseline, on real daily market data?** The honest answer, established below with a clean chronological train/test split, multiple seeds, and two independent markets, is **no** — the upgraded agent underperformed both a simpler baseline PPO agent and buy-and-hold on both markets tested, while also taking on more risk (larger drawdowns, higher variance across seeds). This is reported as the actual, evidence-backed finding rather than adjusted or re-run until a more flattering result appeared.

## 2. Data

Two markets, ~10 years of daily data each (2015-01-01 to 2024-12-31), downloaded via `yfinance` with `auto_adjust=True` (dividends/splits folded into price):

| Ticker | Market | Days | Mean daily return | Std daily return | Lag-1 autocorrelation |
|---|---|---|---|---|---|
| SPY | US equity index (S&P 500 ETF) | 2,514 | +0.00049 | 0.01114 | −0.1163 |
| ^NSEI | Indian equity index (Nifty 50) | 2,457 | +0.00042 | 0.01054 | −0.0284 |

SPY was reused from Week 6 (permitted once); ^NSEI was chosen as the genuinely new market, directly following the course's own suggested example for Indian markets. Both markets show small positive drift (consistent with long-run index growth) and near-zero, slightly negative lag-1 autocorrelation — i.e., almost no linearly exploitable short-term momentum or mean-reversion in either market, and notably *less* autocorrelation in NSEI than SPY. Nothing unusual was noticed in the raw price series beyond the expected features (SPY shows the March 2020 COVID crash clearly; both series are otherwise smooth, upward-trending index paths).

## 3. Environment design

Starting from Week 6's `HistoricalTradingEnv` (3 actions: long/hold/short; 6-dim state: 5 raw returns + position), I implemented two upgrades in `ImprovedTradingEnv`:

**A. Richer state features.** Added two features beyond the raw 5-return window: **rolling volatility** (std of the last 20 returns) and **distance from a 20-day moving average** of price (`(price - MA) / MA`). *Rationale:* the raw window carries almost no linear signal (autocorrelation ≈ −0.03 to −0.12), so I hypothesized that longer-horizon summary statistics — volatility and trend distance — might let the agent manage risk (e.g. size down when volatility is high) even without being able to predict direction. *Expected behaviour:* smaller positions or more caution during volatile stretches. Every feature is computed strictly from data up to and including the current step (verified with an explicit lookahead-bias test in `week7_improved_env.py` that confirms changing future returns never changes a past observation).

**B. Position sizing.** Replaced the 3-action space with 5 actions: full long, half long, flat, half short, full short (positions +1, +0.5, 0, −0.5, −1). The existing transaction-cost term (proportional to `|position change|`) required no modification. *Expected behaviour:* the agent could use half-sized positions when its signal is weak or mixed, rather than being forced into an all-or-nothing bet.

I chose these two specifically because they're structurally independent of each other (one only changes the state, the other only changes the action space) and both directly target the same weakness identified in Week 6: a near-zero raw signal that a richer, more flexible policy *might* be able to exploit better than a blunt 3-action agent — a clean, testable hypothesis rather than a grab-bag of changes.

## 4. Experimental setup

- **Split:** chronological 80/20 per market (SPY: 2,011 train / 503 test days; NSEI: 1,965 train / 492 test days). All design decisions were made on training data only; test sets were touched only for the final numbers below.
- **Seeds:** 3 seeds (7, 13, 99) per trained configuration; results reported as mean ± std across seeds.
- **Training budget:** 100,000 timesteps per run, PPO (`learning_rate=3e-4, n_steps=512, batch_size=64, gamma=0.99`), identical for both the improved and baseline agent so any difference in outcome isolates the environment upgrades, not the training budget.
- **A technical note carried over from Weeks 5-6:** real daily returns are tiny in scale (~0.01), which causes PPO's default-initialized network to struggle to learn a clean policy. All training here uses SB3's `VecNormalize` (observation standardization) for this reason, applied identically to both the improved and baseline agent.
- **Four strategies compared per market, all evaluated on the test set, 50 episodes each** (random repeated 3×; buy-and-hold is deterministic, one pass):
  1. Improved agent (this project's upgrades)
  2. Baseline agent (PPO on the plain Week 6 environment, same budget)
  3. Random agent
  4. Buy-and-hold

## 5. Results

**SPY (test set, mean ± std over seeds/repeats):**

| Strategy | Final P&L | Sharpe | Max Drawdown |
|---|---|---|---|
| Improved agent | +202.16 ± 82.53 | +1.31 ± 0.64 | 21.78 ± 18.90 |
| Baseline agent | **+458.26 ± 19.44** | **+1.88 ± 0.09** | **14.04 ± 2.89** |
| Random agent | −325.80 ± 60.18 | −1.32 ± 0.27 | 331.78 ± 56.63 |
| Buy-and-hold | +448.74 ± 0.00 | +1.83 ± 0.00 | 12.70 ± 0.00 |

**^NSEI (test set, mean ± std over seeds/repeats):**

| Strategy | Final P&L | Sharpe | Max Drawdown |
|---|---|---|---|
| Improved agent | +279.11 ± 40.80 | +1.25 ± 0.17 | 18.14 ± 3.91 |
| Baseline agent | **+373.22 ± 50.23** | **+1.52 ± 0.22** | 14.56 ± 1.34 |
| Random agent | −315.60 ± 36.19 | −1.34 ± 0.17 | 321.64 ± 33.08 |
| Buy-and-hold | +377.18 ± 0.00 | +1.62 ± 0.00 | **11.60 ± 0.00** |

Cumulative P&L plots (`plots/SPY_four_strategy_pnl.png`, `plots/NSEI_four_strategy_pnl.png`) show the same story visually: on SPY, the baseline agent and buy-and-hold curves are nearly superimposed, climbing steadily to ~+9-10 cumulative P&L by the end of the episode, while the improved agent's curve rises far more slowly, ending under +2.5. On NSEI the gap is smaller — the improved agent tracks the baseline/buy-and-hold curves reasonably closely for the first ~60% of the episode before falling slightly behind — but the final ranking is the same. In both markets, the random agent's curve is a clear, steadily worsening line below zero, driven by constant churn costs.

**Train-vs-test check (improved agent):** SPY: train +185.56 ± 59.55 vs. test +202.16 ± 82.53 (test slightly *higher*); NSEI: train +293.19 ± 47.41 vs. test +279.11 ± 40.80 (test slightly lower). Neither shows the classic large-gap overfitting signature (train ≫ test); the differences are within one standard deviation of the seed-to-seed spread in both cases.

## 6. Discussion

**Did the upgrades help over the baseline agent? No — on both markets, the improved agent underperformed the baseline agent on every metric:** lower final P&L, lower Sharpe ratio, and *larger* max drawdown, with substantially higher variance across seeds (e.g. SPY final-P&L std of 82.53 for the improved agent vs. 19.44 for the baseline). **Did anything beat buy-and-hold, and do I believe it?** No strategy clearly and consistently beat buy-and-hold on both markets; the baseline agent came within noise of it on both (slightly above on SPY, slightly below on NSEI), which is consistent with Week 6's finding that PPO tends to converge toward a buy-and-hold-like policy when the underlying signal is this weak. I don't believe either "beat" buy-and-hold in any exploitable sense — the differences are small relative to the seed-to-seed spread.

**What surprised me** was not that the upgrades failed to find profit (expected, given the near-zero autocorrelation measured in Week 6) but *that they actively hurt performance and increased risk* relative to the simpler baseline, given an identical training budget. My best explanation: the improved agent faces a strictly harder learning problem — an 8-dimensional state (vs. 6) and a 5-action space (vs. 3) — with the *same* 100,000-timestep budget as the baseline. With a near-zero real signal to find, the extra capacity mostly translates into extra ways to converge inconsistently (visible directly in the much larger seed-to-seed standard deviations for the improved agent on every metric), rather than into a genuinely better policy. The added features and finer position granularity are a real hypothesis worth testing, but this experiment suggests they need either a larger training budget or a more carefully shaped reward to pay for their added complexity — they didn't pay for themselves here.

**My single most defensible conclusion:** **On both SPY and Nifty 50, richer features and finer position sizing did not produce a more profitable or safer PPO trading agent than a simpler 3-action baseline under an identical training budget — they underperformed it on every metric measured, while adding meaningfully more risk and run-to-run variance.**

## 7. Limitations and next steps

This backtest still ignores several things that matter for real trading: slippage and partial fills (the environment assumes instant execution at the observed price), regime change beyond what a single chronological split can capture (10 years still contains only one COVID-scale shock and one clear bull run each), and it covers only two liquid, large-cap equity indices — nothing about single stocks, other asset classes, or intraday dynamics. With two more weeks, the first thing I would try is giving the improved agent a matched or larger training budget than the baseline (rather than an equal one) to test whether its extra capacity converges more reliably given more time, before concluding the added features themselves are the problem rather than the budget.
