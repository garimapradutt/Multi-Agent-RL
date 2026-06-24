# Week 3 — Reflection Notes

## Task 1: Probing REINFORCE Failure Modes

Results from `week3_probe.py` (300 episodes, same seed):

| Run | LR | γ | Final 10-ep avg | Observation |
|-----|----|---|-----------------|-------------|
| Baseline (Week 2) | 3e-3 | 0.99 | **432.2** | Learns well, reaches near-optimal |
| Run A — lr=0.1 | 0.1 | 0.99 | **9.3** | Diverged immediately |
| Run B — lr=1e-5 | 1e-5 | 0.99 | **17.1** | Stuck at random-level performance |
| Run C — γ=0.50 | 3e-3 | 0.50 | **63.9** | Learns, but poorly — short-sighted |

**Run A (high lr):** The policy diverged catastrophically. A learning rate of 0.1 is so large that each gradient step destroys whatever the policy learned in the previous episode. The policy collapses to a degenerate strategy (e.g., always push left) from which it never recovers. This confirms that REINFORCE's noisy gradient estimates are dangerously amplified by a large step size.

**Run B (tiny lr):** The agent barely moves. After 300 episodes it still performs at random-agent level (~17). The gradient signal is real but the steps are so tiny that 300 episodes is nowhere near enough to learn anything meaningful. This is a sample-efficiency failure — the algorithm is correct but impractically slow.

**Run C (γ=0.5):** The agent does learn (63.9 >> random ~17), but much worse than the baseline. A discount of γ=0.5 means rewards two steps away are worth only 25% of immediate rewards (0.5² = 0.25). CartPole needs the agent to balance *long-term* — keeping the pole up for 100+ steps. With γ=0.5, the agent essentially ignores anything more than ~5 steps ahead, so it cannot learn to plan for sustained balance.

---

## Task 2: PPO on CartPole

PPO evaluation after 50,000 timesteps: **500.0 ± 0.0** — a perfect score across 10 evaluation episodes.

**n_steps=2048:** This is the number of environment steps PPO collects before running any gradient updates. PPO is an on-policy algorithm: it gathers experience, trains on it, then discards it and gathers fresh data. Larger n_steps gives a more diverse batch (lower variance estimates) but delays the first update. 2048 steps ≈ 4–50 CartPole episodes, giving a good mix of short and long trajectories per update.

**batch_size=64:** After collecting n_steps of experience, PPO shuffles those steps and updates the policy on random mini-batches of 64. This is more efficient than one giant gradient step (which would have high curvature) and allows SGD's stochasticity to help avoid local minima. The 2048 steps are reused for n_epochs=10 passes before being discarded.

**EpisodeRewardLogger:** A lightweight hook into SB3's training loop. At each step it accumulates the reward; when `dones=True` it records the episode total and resets its counter. This gives us a per-episode learning curve even though SB3 internally thinks in timesteps, not episodes.

---

## Task 3: REINFORCE vs PPO Comparison

From `plots/reinforce_vs_ppo.png`:

**REINFORCE:** Slow improvement starting around episode 60, peaks around 400–480 average in episodes 240–280, then oscillates between 150–490 for the rest of training. Never locks in.

**PPO:** Reaches near-perfect performance (500) very quickly (around episode 100 out of ~395 total episodes), and holds it stably. Virtually no oscillation once it converges.

The contrast is stark. PPO is not just faster — it's qualitatively more stable. The two algorithmic differences explain this:
1. **Batch updates:** PPO averages gradients over 2048 steps, not one episode; this alone cuts gradient variance by ~40×.
2. **Clipping:** PPO's clipped objective prevents large, destabilising policy changes — the updates that cause REINFORCE to collapse simply cannot happen under PPO.

---

## Task 4: Toy Trading Environment

The `ToyTradingEnv` implements a Gymnasium-compatible trading environment. A random agent over 10 episodes averages ~0 reward (as expected on a pure random walk with no drift). The environment interface satisfies `reset()` and `step()`, making it compatible with Stable-Baselines3 — which we will use in Week 4 to train PPO on it.

**Key design decisions:**

- **State:** `[last_price_return, current_position]` — minimal but sufficient to let the agent know where the price just moved and what it currently holds.
- **Action:** Discrete 3 (Long, Hold, Short) — simpler than continuous and sufficient for a first experiment.
- **Reward:** `position × price_return × 100` — immediate step-level P&L; no transaction costs yet (added in Week 4).
- **Episode length:** Fixed at 100 steps — uniform episodes make training more predictable.

On a pure random walk the expected reward of any strategy is zero. The random agent confirms this (average ≈ 0 over enough episodes). Any agent that consistently beats zero would require a structural bias in the environment — which this environment deliberately avoids.
