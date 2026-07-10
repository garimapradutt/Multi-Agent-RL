# Week 6 — Reflection Notes

## A note on VecNormalize (carried over from Week 5)

Real daily returns are similarly tiny in scale to Week 5's synthetic returns (std ≈ 0.011, see below), so I ran into the same PPO-collapses-to-a-degenerate-policy problem I documented in `week5_notes.md`, and used the same fix: wrapping training in SB3's `VecNormalize` and evaluating through a `NormalizedPredictor` wrapper that stays compatible with `week4_evaluate.py`'s `average_cumulative_pnl`.

---

## On the data (Task 1)

Downloaded SPY daily closes, 2015-01-01 to 2024-12-31, via `yfinance` (2,514 trading days, 2,515 raw price rows). Measured statistics (`week6_get_data.py`):

| Statistic | Value |
|---|---|
| Number of days | 2,514 |
| Mean daily return | +0.00049 |
| Std of daily returns | 0.01114 |
| Lag-1 autocorrelation | −0.1163 |

### Q1 — Lag-1 autocorrelation of real returns vs. Week 5 markets

The real measured autocorrelation is **−0.1163** — small in magnitude and, notably, slightly *negative* rather than the near-zero-but-could-go-either-way I expected. It's an order of magnitude smaller than Week 5's rho=0.5 momentum market, and even smaller in magnitude than the rho=0.25 "weak signal" case from the sweep. This implies there is very little exploitable linear structure available to an agent using only a short window of raw past returns — any edge here, if it exists at all, is far weaker than anything I trained against in Week 5, and Week 5's sweep already showed that even a "medium" edge (rho=0.25) barely survived a modest transaction cost.

### Q2 — Mean and std vs. daily_vol=0.01; what does a small positive mean represent?

The real std (0.01114) is very close to the synthetic `daily_vol=0.01` — a good sign that the synthetic environments were calibrated to a realistic volatility scale. The mean is **not exactly zero**: it's a small positive number (+0.00049 per day). Compounded over ~252 trading days a year, that's roughly +12-13% annualized — this is the market's long-run upward drift (the reason buy-and-hold investors in a broad index have historically made money over time), and it's exactly the kind of drift Week 5's synthetic markets deliberately did NOT have (they were constructed with zero mean by design, so any profit could only come from exploiting rho, not drift).

---

## On the environment (Task 2)

### Q3 — Why the split must be chronological, not shuffled

A shuffled split would scatter days from across the whole 10-year history randomly into both the training and test sets, meaning the training set would contain days from *inside and even after* the nominal test period. This is **lookahead bias**: the agent (and the VecNormalize statistics fit on "training" data) would implicitly learn about the volatility level, trend, and specific events of what's supposed to be the unseen future. Concrete example: if the March 2020 COVID crash landed in the training set while some of the following recovery days landed in the "test" set (because of random shuffling), the agent could effectively learn "after a huge crash, prices recover sharply" from training data that literally contains days *chronologically after* some of its test days — a backtest that looks great purely because it secretly already knows how the test period's story ends.

### Q4 — What the random-window trick does and doesn't fix

Random contiguous windows give PPO variety during training (~1,900 different overlapping 100-day windows from ~2,000 training days, rather than replaying one fixed episode) and let evaluation average over many different stretches of history rather than one lucky/unlucky one. What it **cannot** fix: there is still only one real history. The windows overlap heavily, so 50 evaluation episodes are not 50 independent samples of "what the market could have done" — they're 50 different views of the *same* single realized decade. No amount of resampling manufactures new information beyond the one sequence of real prices we actually have.

---

## On train vs. test (Task 3)

Chronological 80/20 split: **2,011 training days**, **503 test days**.

### Q5 — The four P&L numbers, and is this surprising?

| Strategy | Final mean cumulative P&L (test set) |
|---|---|
| PPO trained | +8.32 |
| Random agent | −7.32 |
| Buy-and-hold | +9.26 |

(PPO on TRAIN data: +2.67 — see Q6.)

PPO clearly beat the random baseline (+8.32 vs. −7.32) and came very close to, but slightly below, buy-and-hold (+8.32 vs. +9.26). Given Weeks 4-5, this is *not* surprising: Week 6's Data section shows the test period sits within a market with real positive drift (unlike Week 4-5's zero-drift synthetic markets), and Week 5 already showed that a very weak/near-zero signal (much like our measured −0.1163 autocorrelation) isn't enough edge to reliably beat a strong directional baseline once transaction costs are involved. The agent essentially learned to behave close to buy-and-hold — capturing most of the market's drift while occasionally paying small costs — rather than discovering a large exploitable pattern that doesn't really exist in this data.

### Q6 — Train/test gap

The gap here is unusual: PPO scored **higher on test (+8.32) than on train (+2.67)** — the opposite direction from the classic overfitting signature. This isn't overfitting; it more likely reflects a **regime difference** between the two periods: the training slice (2015 through mid-2023) contains several genuinely difficult, choppy, and crash-heavy stretches (2018 Q4, the March 2020 COVID crash, 2022's bear market), while the last 20% of the data (test, roughly late 2023 to end of 2024) was a comparatively calm, strongly trending bull run. A policy that leans toward "stay long, don't churn" — which is what PPO seems to have converged toward here — will naturally score better in a smoother, more strongly-trending test window than in a training window full of sharp reversals. A near-zero or negative gap here would tell me the agent didn't memorize anything training-period-specific; a large *positive* train-minus-test gap would be the overfitting red flag to watch for, and that's not what happened.

### Q7 — Spread across 3 runs

| Seed | PPO on TEST |
|---|---|
| 7 | +8.32 |
| 13 | +9.30 |
| 99 | +9.29 |

The spread was small (+8.32 to +9.30, a range of under 1.0) — all three seeds converged to a very similar, buy-and-hold-like policy and a very similar test outcome. This is reassuring: it suggests this particular result (PPO ≈ buy-and-hold, clearly > random) is not a fluke of one lucky training run. That said, if I'd only seen one run, I'd have drawn essentially the same conclusion here regardless of which seed I happened to pick — but that's a property of this data/setup, not a general guarantee, which is exactly why the course insists on checking every time rather than assuming it.

### Q8 — Why beating buy-and-hold is a much higher bar than beating random

Beating the random agent only requires *not actively destroying value through unnecessary trading costs* — random churns constantly and pays for it, so almost any deliberate policy beats it. Beating buy-and-hold requires *finding a real, exploitable edge on top of the market's own drift* — buy-and-hold already captures the market's positive expected return for free, with zero transaction costs (it never trades after the initial position). To beat it, an agent would need to time entries/exits well enough that the P&L gained from correctly avoiding down-moves (or adding leverage on up-moves) exceeds both the transaction costs of doing so AND the drift it temporarily sits out of. Given the tiny (−0.1163) autocorrelation measured in Task 1, there's very little timeable signal available to earn that back — which is exactly why PPO landed just under, not over, buy-and-hold.

---

## Big picture

### Q9 — Where might genuine structure hide that this state representation can't see?

1. **Cross-asset / macro signals** — e.g. bond yields, volatility indices (VIX), or sector rotation patterns; our state only ever sees this one asset's own past returns, so any structure that lives in the *relationship* between assets is invisible to it.
2. **Longer-horizon or non-linear patterns** — our history window is only 5 days of raw returns; real, weak effects (e.g. medium-term momentum over months, or volatility clustering) operate on time scales or through non-linear features (like rolling volatility) that a 5-return linear window simply doesn't represent.
3. **Fundamental / alternative data** — earnings, macro news, order-book/microstructure data, sentiment. Real quant funds often derive edge from information sources entirely outside the price series itself, which this project's state representation (by design, so far) has never included.

### Q10 — 3 questions for my mentor

1. The train/test P&L gap went the "wrong way" (test higher than train) here because of what looks like a regime difference between the two periods rather than overfitting — is there a standard way to test for this (e.g., comparing volatility or drift statistics of the train vs. test slices directly) rather than just eyeballing the price chart?
2. Given how close PPO landed to buy-and-hold, is there a principled way to tell "the agent learned to approximate buy-and-hold because that's genuinely close to optimal here" apart from "the agent gave up trying to trade the noise, which happens to look similar"?
3. For Week 7-8, when I add richer state features (e.g. rolling volatility), what's the standard first check to make sure I haven't introduced lookahead bias, beyond the "change day t+1 and see if the feature at day t changes" test mentioned in the final project brief?
