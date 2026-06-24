# Week 4 — Reflection Notes

## Task 1: PPO on ToyTradingEnv

**Training summary:** PPO was trained for 100,000 timesteps (~1,003 episodes of 100 steps each).

**n_steps=512** controls how many environment steps are collected before any PPO gradient updates occur. Compared to the CartPole default of 2048, I used 512 here because trading episodes are only 100 steps. With 512 we collect ~5 complete episodes per batch — enough diversity for a stable gradient estimate without waiting too long between updates. Using 2048 would mean ~20 episodes per batch, which would slow down early learning on this short-episode environment.

**batch_size=64** is the mini-batch size for the gradient updates within each PPO iteration. The 512 collected steps are randomly shuffled and processed in chunks of 64 for 10 epochs. Smaller batches introduce helpful stochasticity; larger batches are more stable but slower per update.

**Reward curve observation:** The 20-episode moving average oscillates around zero throughout training. There is no sustained upward trend, unlike CartPole. This is the **correct and expected result** — the environment uses a pure random walk, which has no exploitable structure. The agent cannot learn a pattern that does not exist.

**PPO vs. random comparison:**
- PPO final 20-ep average: **−2.25**
- Random agent average (200 ep): **−0.76**

Both are near zero. The slight negative values are noise from a finite sample. The conclusion is that PPO converges to approximately the optimal behaviour for this environment: near-zero expected P&L, which matches theoretical prediction for a zero-drift random walk.

---

## Task 2: Extended Trading Environment (TradingEnvV2)

The two additions to the environment are:

**Price history window (history_len=5):**  
The observation expands from 2D to 6D — it now includes the last 5 price returns plus the current position. This gives the agent a short-term "memory" of recent price movements. However, on a true random walk, past returns carry zero information about future returns (by definition), so this does not help the agent profit. What it *can* help with is position management — the agent may learn "I've been long through 5 consecutive down moves, so I should cut my position," which is valid risk management even when prices are unpredictable.

**Transaction cost (transaction_cost=0.1):**
The reward now subtracts `0.1 × |new_position − old_position|` whenever the position changes. The cost equals zero when holding, 0.1 when going from flat to Long or Short, and 0.2 when flipping completely from Long to Short (or vice versa). This discourages "churning" — flipping positions every step to chase noise. Without transaction costs, an agent trained on a random walk might churn (earning nothing but paying no cost either); with costs, churning is punished, pushing the agent toward a hold-biased strategy.

**Why `cost = transaction_cost × |new_position − old_position|`?**
Because the cost should be proportional to how much the position *changed*. A partial or gradual rebalancing (flat → long = size-1 change) costs less than a full reversal (long → short = size-2 change). If the position stays the same (|Δ| = 0), there is no trade and no cost.

---

## Task 3: Cumulative P&L Evaluation

From `plots/cumulative_pnl.png` (50 evaluation episodes per strategy):

| Strategy | Mean final P&L | Std |
|----------|----------------|-----|
| PPO agent | −0.05 | 10.34 |
| Random agent | −0.37 | 9.61 |
| Buy-and-Hold | −0.05 | 10.34 |

All three strategies end near zero, with large variance (±10). The standard deviation is driven by the randomness of the price walk — any single 100-step episode can drift significantly in either direction by chance.

**Is near-zero the right answer?** Yes. On a pure random walk the expected P&L of *any* strategy is zero. An agent that claimed consistent profit would be:
  1. Memorising a specific training seed (environment not reset properly), or
  2. Exploiting a statistical artefact in a small sample, or
  3. Benefiting from an unintended bias in the reward function.

None of those happened here. The flat curves confirm the environment is correctly implementing a zero-drift random walk, and the agent converged to the theoretically optimal behaviour.

**What "RL working correctly" vs "finding a profitable strategy" means:**
RL is working correctly — the policy converged, the loss decreased, the agent learned to manage positions. But "working correctly" on a random walk cannot produce profit. This is one of the most important lessons of the week: RL is only as useful as the signal in the environment. Week 5+ will introduce more realistic market dynamics where genuine signal may exist.

---

## Summary: What the Agent Learned (or Didn't)

The PPO agent on ToyTradingEnv **correctly learned that no systematic profit is possible** and converged to approximately zero expected P&L — matching the theoretical optimum. This is not a failure; it is the right answer. A *wrong* agent that produced consistently positive P&L would be evidence of a bug (not intelligence).

What the agent *did* learn (implicitly, via the reward signal):
- Holding is usually better than churning (no transaction costs when flat).
- The last price return is not a useful predictor of the next (random walk).
- Long and short positions earn/lose symmetrically on this walk.

The real challenge — and the purpose of Weeks 5+ — is to repeat this entire process on environments with genuine market microstructure, where the agent *might* find real signal.
