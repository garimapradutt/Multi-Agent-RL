# Week 5 — Reflection Notes

## A note on VecNormalize (important — read before the rest)

While building `week5_train_and_evaluate.py`, PPO with the assignment's exact hyperparameters (`learning_rate=3e-4, n_steps=512, batch_size=64, gamma=0.99`) reliably converged to a **degenerate fixed policy** — always long, always short, or always flat — regardless of the actual sign of the last return. I verified this by probing the trained policy directly with synthetic inputs (e.g. feeding it observations where the history window was clearly positive vs. clearly negative) rather than trusting noisy episode-level P&L, and confirmed the policy's output didn't depend on the sign at all across several seeds and training lengths (I tried 100k, 200k, and multiple seeds before diagnosing this).

The root cause: `daily_vol=0.01` means the raw state features are tiny (~±0.01), while PPO's default network initialization is tuned for roughly unit-scale inputs. This makes it very hard for gradient descent to find a sharp decision boundary at zero. Wrapping the training environment in Stable-Baselines3's `VecNormalize` (which standardizes observations using a running mean/std — a standard, well-known tool for exactly this problem) fixed it immediately: with normalized observations, the same PPO hyperparameters reliably learn a genuine sign-following policy. I verified this both by direct probing and by measuring a "sign-match fraction" (does the new position match the sign of the return the agent saw when it chose that position?) — it jumped from ~50% (chance level) to ~78-79% once normalization was added. All numbers below use this fix (`train_ppo_normalized` in `week5_train_and_evaluate.py`), applied via a thin `NormalizedPredictor` wrapper so it still plugs into `week4_evaluate.py`'s unchanged `average_cumulative_pnl`/`plot_pnl`.

---

## On the signal check (Task 2)

### Q1 — Measured autocorrelations, and why they don't match rho exactly

From `week5_check_signal.py` (10,000 steps each):

| rho set | measured autocorrelation |
|---------|---------------------------|
| 0.00 | −0.000 |
| 0.50 | +0.501 |
| −0.50 | −0.501 |

These are extremely close to the true rho — within 0.001 — because 10,000 samples is a fairly large sample for estimating a lag-1 correlation. They don't match *exactly* because (a) it's still a finite sample, so there's residual sampling noise, and (b) `collect_returns` resets the environment periodically (every 100 steps), and each `reset()` clears `_last_return` to 0.0, which briefly breaks the AR(1) chain at each episode boundary — a small, expected effect the assignment itself calls out.

### Q2 — Random agent's mean reward: slightly negative and roughly constant across rho

Measured random-agent mean rewards: rho=0.00 → **−9.43**, rho=0.50 → **−7.49**, rho=−0.50 → **−8.14**. All three are negative and similar in magnitude. The negative drift comes from transaction costs: a uniformly random agent picks a new position uniformly among {Long, Hold, Short} every step, so on roughly 2/3 of steps it either changes position (paying a cost) or ends up somewhere different from where it started — either way, it churns constantly and pays the `transaction_cost × |Δposition|` penalty far more often than a deliberate agent would. Market structure alone doesn't help an agent that ignores the state: the random agent never looks at the return history at all, so whether `rho` is 0, +0.5, or −0.5 is irrelevant to it — its expected P&L *before* costs is 0 in all three worlds (it doesn't correlate its position with anything), and the *cost* term is identical in all three (its churn rate doesn't depend on rho either). That's exactly why all three numbers land in the same ballpark.

---

## On training (Task 3)

### Q3 — What changed vs. Week 4's cumulative_pnl.png

Week 4's plot (pure random walk, rho=0) showed PPO, random, and buy-and-hold all hovering near zero (with the random agent noticeably worse due to costs). This week's `cumulative_pnl_momentum.png` (rho=0.5) shows **PPO pulling clearly and steadily above both baselines**, ending at **+16.04** vs. random's **−9.23** and buy-and-hold's **−2.77**. The single property responsible is `rho`: with rho=0.5, `E[r_{t+1} | r_t] = 0.5·r_t ≠ 0`, so for the first time the expected next return actually depends on the current state, and a policy that conditions on that state can earn positive expected reward. Everything else (state, actions, reward, evaluation harness) is unchanged from Week 4.

### Q4 — The policy I observed vs. the hand-coded rule

I measured a **sign-match fraction of ~0.78-0.79**: about 4 times out of 5, the agent's new position matches the sign of the return it just observed — reasonably close to the theoretically optimal rule "match the sign of the last return," though clearly not a perfect 1.0 match. The gap from 1.0 is exactly where the transaction cost matters: with `transaction_cost=0.1`, flipping position on every tiny wiggle isn't worth it (the expected gain from a very small `|r_t|` is smaller than the cost of flipping), so a cost-aware agent should — and appears to — sometimes hold through small moves rather than chasing every sign change. This matches the qualitative prediction from the Week 5 resources almost exactly.

### Q5 — Why a fixed position can't exploit autocorrelation (buy-and-hold ends near zero)

Buy-and-hold always holds position = +1, completely ignoring the sign of the last return. Its expected reward is `1 × E[r_{t+1}] × 100 = 1 × 0 × 100 = 0`, because the *unconditional* mean of an AR(1) process is still zero regardless of rho (rho only affects how returns are correlated *with each other*, not their long-run average). Exploiting momentum requires actively *conditioning the position on the current state* — going long after an up-move and short after a down-move — which is exactly what a fixed policy structurally cannot do.

---

## On the sweep (Task 4)

Sweep table (from `week5_sweep.py`, 50,000 timesteps per run, `average_cumulative_pnl` over 50 episodes):

**Sweep 1 — signal strength (transaction_cost = 0.1 fixed):**

| rho | final mean cumulative P&L |
|-----|----------------------------|
| 0.00 | −0.04 |
| 0.25 | +0.23 |
| 0.50 | +17.87 |

**Sweep 2 — transaction cost (rho = 0.25 fixed):**

| transaction_cost | final mean cumulative P&L |
|-------------------|----------------------------|
| 0.00 | +7.96 |
| 0.10 | −0.27 |
| 0.30 | −0.64 |

### Q6 — How did final P&L change with rho?

It grew sharply and monotonically with rho: essentially flat at rho=0 (no signal to trade), small and only slightly positive at rho=0.25 (a weak, marginal signal), then a large jump at rho=0.5 (+17.87). This matches expectations — a stronger AR(1) coefficient means a more predictable next return, so the theoretical edge per correctly-timed trade (`≈ ρ·E[|r|]·100`) scales roughly linearly with rho, and the jump from a small edge to a comfortably cost-covering edge shows up as a jump in realized P&L, not a smooth line.

### Q7 — Where did the cost eat the whole edge at rho=0.25?

At rho=0.25, going from `cost=0.00` (+7.96) to `cost=0.10` (−0.27) wiped out essentially the entire edge — the small remaining signal at rho=0.25 (a weaker, noisier momentum effect than rho=0.5) is only barely worth trading even with zero cost, so a modest cost of 0.1 was already enough to erase it. In plain terms: **there's a small pattern in this market, but it's so faint that even a small trading fee makes it not worth chasing** — you'd spend more on fees repositioning than you'd earn from correctly betting on the pattern.

### Q8 — Optimal position rule for rho = −0.5 (mean reversion)

For rho<0, `E[r_{t+1}|r_t] = ρ·r_t` is *negative* when `r_t>0` and positive when `r_t<0` — the expected next move is in the **opposite** direction of the last one. So the optimal rule flips relative to the momentum case: **bet against the last move** — go short after an up-move, go long after a down-move.

**Optional extension result:** I trained a fresh agent directly on `TradingEnvV3(rho=-0.5, transaction_cost=0.1)` and it reached **+11.69** final P&L, confirming the mean-reversion signal is learnable in the same way the momentum signal was. I then took the *momentum-trained* agent (trained on rho=+0.5, never seeing rho=−0.5 during training) and evaluated it — without retraining — directly on the mean-reversion market. I expected it to lose money outright, since it should have learned exactly the wrong rule for this world. What I actually measured was **+8.81** over 100 fresh test episodes (sign-match fraction against the mean-reversion market's optimal rule: 0.71, i.e. it's *still* betting momentum-style even where mean-reversion is correct) — worse than the properly-trained mean-reversion agent's +11.69, but not the clear loss I predicted. My best explanation: `|rho|=0.5` is identical in both regimes, so a policy that reacts to the *magnitude* of the recent return window (not purely its sign relative to the single last observation) may still be picking up some usable structure even with the "wrong" sign convention, especially since the state includes a 5-return window rather than just the most recent point. This is a genuinely interesting, honest result worth flagging to my mentor rather than a clean confirmation of the textbook prediction — it suggests the actual learned policy is more nuanced than the simple hand-coded "match sign of last return" rule I was comparing it to, and I'd want to inspect the policy's response to the full 5-return window (not just the last element) to understand it properly.

---

## Big picture

### Q9 — Prediction for real data next week

I expect real daily stock/index returns to have a lag-1 autocorrelation much closer to **0.0** than to 0.5. Public, liquid markets are traded by enormous numbers of participants, and any autocorrelation as strong and stable as rho=0.5 would be an enormous, easily-exploitable edge that would already have been arbitraged away. This implies next week will be **much harder** than this week: the signal (if any) will likely be tiny, possibly not statistically distinguishable from zero over the sample we have, and any apparent PPO profit will need much more skepticism (multiple seeds, out-of-sample testing) before it's believable, rather than the clean, large, easily-reproduced edge I found here with a hand-chosen rho=0.5.

### Q10 — 3 questions for my mentor

1. I had to add VecNormalize to get PPO to learn the momentum signal at all with the assignment's stated hyperparameters — is this a well-known standard fix for small-scale continuous features, or is there a more "first-principles" alternative (e.g., scaling the reward or returns directly inside the environment) that's generally preferred?
2. My momentum-trained agent, when deployed on the mean-reversion market, didn't lose money the way I expected from the simple hand-coded-rule intuition — it still ended up positive, just smaller than the properly-trained agent. Is this a sign that PPO here learned something more sophisticated than "match sign of last return," or is this more likely still be noise from a single training run and single seed?
3. When a training run collapses to a degenerate policy (like the "always flat" or "always long" cases I saw before adding VecNormalize), what's the standard diagnostic checklist real practitioners use to distinguish "the environment has no learnable signal" from "the optimizer/features are the problem"?
