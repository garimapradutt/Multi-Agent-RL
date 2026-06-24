# Week 2 — Reflection Notes

## Training Summary

| Parameter   | Value   |
|-------------|---------|
| Algorithm   | REINFORCE (vanilla policy gradient) |
| Environment | CartPole-v1 |
| Hidden dim  | 128 |
| Learning rate | 3e-3 |
| Discount γ  | 0.99 |
| Episodes    | 500 |

---

## Question 1 — Describe the trend in episode returns

The moving average shows a clear upward trend across training:

- **Episodes 1–50:** Baseline-level performance (~10–23), similar to random.
- **Episodes 50–130:** Steady improvement; the agent starts balancing for 30–140 steps.
- **Episodes 130–300:** Strong learning phase — the moving average climbs from ~135 to ~465, and individual episodes frequently hit the maximum of 500.
- **Episodes 300–500:** The characteristic REINFORCE collapse kicks in. The moving average oscillates between ~150 and ~490. The agent periodically rediscovers good policies but cannot consistently hold them.

This oscillation plateau is not a bug — it is a known property of vanilla REINFORCE. Because gradient estimates come from single trajectories, they are noisy, and a large update can push the policy away from a good region as easily as it found it. The agent improves overall compared to episode 1, but it never *locks in* a stable solution.

---

## Question 2 — Trained policy vs. random policy (qualitative comparison)

Running a few episodes with the trained policy vs. random:

- **Random agent:** The pole falls almost immediately, typically within 10–20 steps. The cart makes no attempt to compensate for tilt; actions are purely coincidental.
- **Trained agent:** The cart visibly chases the pole. When the pole starts leaning one way, the cart accelerates under it — the same fundamental insight as the hand-coded PD rule from Week 1, but discovered automatically through gradient updates. The agent routinely balances for 150–500 steps.

The key qualitative difference is **purposeful movement**: the trained agent's actions are clearly correlated with the state, whereas the random agent's actions are independent of the observation entirely.

---

## Question 3 — Hyperparameter experiment

**Experiment:** Changed `learning_rate` from 3e-3 to 0.1 (10× higher).

**Result:** Learning became wildly unstable. The moving average spiked above 200 around episode 50, then crashed to ~9–12 for the rest of training — essentially worse than random. The high learning rate caused each gradient step to overshoot good parameter regions, and the policy collapsed to a degenerate "always push left" behaviour.

**What I learned:** REINFORCE is extremely sensitive to the learning rate because gradient estimates from single episodes have high variance. A step that is "too large" can destroy a good policy in one update. Lower learning rates (3e-3 or 1e-3) allow smaller, more conservative updates that accumulate reliably over hundreds of episodes. This brittleness to hyperparameters is one of the main practical limitations of vanilla REINFORCE — and the primary motivation for more robust algorithms like PPO, which we will use in Week 3.

---

## Question 4 — What does the REINFORCE loss − log π(at | st) Gt try to do?

The loss `L = -Σ_t log π(a_t | s_t) * G_t` does the following:

**In plain English:** For each action taken in an episode, scale its log-probability by how much total reward followed. If an action led to high return (G_t is large), the loss term is very negative, so minimising the total loss means *increasing* that action's probability. If an action led to low return (G_t is small or negative after baseline subtraction), the loss term is positive, so minimising it means *decreasing* that action's probability.

**Why the minus sign?** PyTorch optimisers perform gradient *descent* (minimisation). REINFORCE wants gradient *ascent* on the expected return J(θ). Since we cannot switch the optimiser's direction, we flip the sign of the objective: minimising `-J(θ)` is mathematically identical to maximising `J(θ)`. The minus sign is simply the bridge between "maximise reward" (RL goal) and "minimise loss" (what optimisers do).

**The normalisation step** (`returns - mean) / std`) subtracts a baseline to reduce gradient variance, so the policy gets a clearer signal about *relative* quality (was this episode better or worse than usual?) rather than absolute returns, which can vary widely in scale.
