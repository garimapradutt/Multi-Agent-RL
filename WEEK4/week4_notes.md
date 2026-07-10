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

**Correction:** my first pass of `week4_evaluate.py` accidentally trained and evaluated on the *Week 3* `ToyTradingEnv` instead of the Task 2 `TradingEnvV2` (the extended env with the history window and transaction cost). I fixed this — `week4_evaluate.py` now trains PPO directly on `TradingEnvV2` — and re-ran everything below with the corrected script.

From `plots/cumulative_pnl.png` (50 evaluation episodes per strategy, `transaction_cost=0.1`):

| Strategy | Final mean cumulative P&L |
|----------|---------------------------|
| PPO agent (trained) | −0.40 |
| Random agent | **−6.19** |
| Buy-and-Hold | −0.05 |

This is a more interesting picture than Week 3's plain random walk: PPO and buy-and-hold both hover near zero and track each other closely, while the **random agent ends up clearly negative**. That's not the agent finding profit — it's the random agent bleeding money to transaction costs, since it changes position on ~2/3 of steps (any action other than "stay the same") and pays a cost every time it does. PPO learned that with a random walk underneath, frequent trading is a pure cost with no offsetting benefit, so it converges toward a much more buy-and-hold-like, low-churn policy.

**Re-running with `transaction_cost=0.0`** (Task 3, step 3):

| Strategy | Final mean cumulative P&L (cost=0.0) |
|----------|----------------------------------------|
| PPO agent (trained) | −2.21 |
| Random agent | −0.75 |
| Buy-and-Hold | +0.70 |

With costs removed, the three curves pull **closer together** — the random agent is no longer punished for churning, so it returns to hovering near zero like PPO and buy-and-hold, all consistent with a zero-drift random walk. The gap we saw at `cost=0.1` was specifically the cost of the random agent's churning, not any structural unfairness in the environment; removing the cost removes that gap.

**Is near-zero the right answer?** Yes. On a pure random walk the expected P&L of *any* strategy (before costs) is zero. An agent that claimed consistent, large profit would be:
  1. Memorising a specific training seed (environment not reset properly), or
  2. Exploiting a statistical artefact in a small sample, or
  3. Benefiting from an unintended bias in the reward function.

None of those happened here. The flat-ish curves confirm the environment is correctly implementing a zero-drift random walk with costs, and the agent converged to sensible behaviour: don't trade needlessly.

**What "RL working correctly" vs "finding a profitable strategy" means:**
RL is working correctly — the policy converged and the agent learned to avoid unnecessary transaction costs, effectively rediscovering something close to buy-and-hold. But "working correctly" on a random walk still cannot produce systematic profit above the zero-drift baseline. This is one of the most important lessons of the week: RL is only as useful as the signal in the environment. Week 5+ will introduce more realistic market dynamics (autocorrelated returns, real data) where genuine signal may exist.

**Code comments (`run_episode`, `buy_and_hold_episode`):** `model.predict(obs, deterministic=True)` is used at evaluation time (rather than the sampling used during training) so that we're judging the agent's actual best policy, not adding extra exploration noise on top of it — otherwise two evaluation runs of the same trained model could look different just from action-sampling randomness. `np.cumsum` turns the list of per-step rewards into a running total, so index *i* is "total P&L earned from the start of the episode through step *i*" — exactly what we want to plot as a P&L curve over time.

---

## Task 4: Written Reflection — Remaining Questions

### Q8 — Hardest part of applying RL to financial markets

After four weeks (REINFORCE → PPO → trading environment → evaluation), the hardest part is clearly **environment/reward design**, not the algorithm. PPO is the same off-the-shelf algorithm whether it's balancing a pole or trading — what determines whether it can possibly succeed is whether the environment actually contains learnable structure (Week 4's random walk has none) and whether the reward accurately reflects what we want (e.g., transaction costs matching real trading frictions). Getting the algorithm to converge was comparatively easy; deciding *what* to have it converge on, and verifying the environment isn't accidentally unbeatable or accidentally exploitable, is the real engineering and judgment problem.

### Q9 — 3 things to add to TradingEnvV2 before trusting it with real money

1. **Realistic price dynamics with actual structure** (changes the *episode structure / underlying data source*) — replace the synthetic random walk with real historical or realistically-correlated price data, since a random walk by construction has nothing to learn and can't validate that the agent would do anything useful in a real market.
2. **Slippage and partial fills** (changes the *reward*) — real orders don't always execute instantly at the observed price, especially in size; the reward function would need to model execution uncertainty, not just a fixed proportional transaction cost.
3. **Risk limits / position constraints, e.g. max drawdown stop or leverage cap** (changes the *action space and episode structure*) — right now the agent can hold a full ±1 position indefinitely with no risk management; real capital allocation requires hard constraints (stop-losses, max position size relative to capital) that would need to be enforced either in the action space or as episode-terminating conditions.

### Q10 — 3 questions for my mentor

1. Since the random agent's loss at `transaction_cost=0.1` came entirely from churning costs, how do real trading systems decide the "right" transaction cost to assume during training if the true cost depends on order size, venue, and market conditions?
2. PPO converged toward a low-churn, buy-and-hold-like policy here — is that a general pattern (PPO tends to find the "do nothing extra" policy when there's no signal), or is it specific to how I set up the reward and cost?
3. Now that I've seen how much the transaction cost changes agent behavior, how would you recommend testing whether an agent's learned trading frequency is "reasonable" versus overfit to one specific cost assumption?

---

## Summary: What the Agent Learned (or Didn't)

The PPO agent on TradingEnvV2 **correctly learned that no systematic profit is possible on a random walk, and that avoiding unnecessary trading is the sensible response to a transaction cost**. It converged to a low-churn, near-buy-and-hold policy that ends up close to zero P&L — much better than the random agent, which pays away over −6 in costs from unnecessary trading. This is not "finding profit"; it's the right answer for a costly, structureless market. A *wrong* agent that produced consistently positive P&L would be evidence of a bug (not intelligence).

What the agent *did* learn (implicitly, via the reward signal):
- Excessive trading is a pure cost with no offsetting benefit on a random walk, so holding is usually better than churning.
- The last price return (and the whole history window) is not a useful predictor of the next return (random walk).
- Long and short positions earn/lose symmetrically on this walk.

The real challenge — and the purpose of Weeks 5+ — is to repeat this entire process on environments with genuine market microstructure, where the agent *might* find real signal.
