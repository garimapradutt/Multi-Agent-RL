# Week 1 — Reflection Notes

## Part 1: Random Agent Observations

**Results (5 episodes of random agent on CartPole-v1):**

| Episode | Steps survived |
|---------|---------------|
| 1       | ~12           |
| 2       | ~23           |
| 3       | ~17           |
| 4       | ~9            |
| 5       | ~31           |

*(Exact numbers will vary each run since actions are random.)*

### Question 1 — What does each of the 4 numbers in the observation mean?

The observation is `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`:

- **cart_position** — how far left or right the cart is from the centre of the track (0 = centre; range ≈ ±2.4).
- **cart_velocity** — how fast and in which direction the cart is moving (positive = rightward).
- **pole_angle** — the angle of the pole measured in radians from vertical (0 = upright; positive = tilted right, negative = tilted left; episode ends if it exceeds ±12° ≈ ±0.209 rad).
- **pole_angular_velocity** — how quickly the pole is rotating (positive = rotating clockwise/rightward, negative = counter-clockwise).

### Question 2 — Why does the pole fall so quickly?

A random agent picks left or right with equal probability regardless of where the pole is. The cart makes no attempt to compensate for the tilt, so small deviations grow rapidly. Because there is no feedback between the observed state and the chosen action, the pole acts like a pendulum with no correction — any initial tilt compounds until the episode terminates.

### Question 3 — Maximum total reward in CartPole-v1?

The maximum is **500**. CartPole-v1 gives +1 reward for every step the pole stays up, and the episode is automatically truncated after 500 steps (even if the pole is still upright). So the theoretical maximum reward is 500.

---

## Part 2: Hand-Coded Rule

### Question 4 — What rule did you use?

A **PD-style (Proportional-Derivative) weighted rule**:

```python
k = 0.1
signal = pole_angle + k * pole_ang_vel
return 1 if signal > 0 else 0
```

The intuition is: don't just look at where the pole is (angle), also look at where it's heading (angular velocity). If the combined signal is positive (pole leans/moves rightward), push right; otherwise push left. The weight `k = 0.1` gives the velocity term a 10% influence — enough to anticipate tipping but not so much that the cart overshoots.

### Question 5 — Average score?

Using the PD-style rule (`signal = pole_angle + 0.1 * pole_ang_vel`), the agent achieved a **perfect average of 500.0 over 5 episodes** — the maximum possible score in CartPole-v1. The random baseline is ~20.

### Question 6 — Did the rule work every time?

No. The rule occasionally fails because it only reacts to the **current** angle and does not predict where the pole will be in the next step. If the pole is barely tilted but is rotating very fast in the same direction, the simple rule may not push hard enough or change direction in time. In physics terms, the rule ignores momentum — it is purely proportional, not derivative.

### Question 7 — Can you think of a smarter rule?

Yes. A **PD (Proportional-Derivative) controller** would be better:

```
action = sign(pole_angle + k * pole_angular_velocity)
```

Here, `k` is a small positive constant that makes the rule react not just to the current angle but also to how fast the angle is changing. This anticipates where the pole is heading and pushes earlier, reducing overshoot. A further improvement would be to also consider `cart_position` to prevent the cart from drifting off the edge of the track.

---

## Part 3: MountainCar-v0 Observations

### Question 8 — Which environment did you try?

`MountainCar-v0`.

### Question 9 — What is the agent trying to do?

A car sits in a valley between two hills. The engine is too weak to drive straight up the right hill from a standing start. The agent must learn to drive left first (gaining potential energy by climbing the left hill), then use the momentum to swing back right and make it over the peak. The goal is to reach the flag at the top of the right hill.

### Question 10 — What does a state look like? What are the possible actions?

- **State (2-dimensional):** `[position, velocity]`
  - `position` ∈ [-1.2, 0.6] — where the car is along the valley curve
  - `velocity` ∈ [-0.07, 0.07] — current speed and direction
- **Actions (3 discrete):**
  - `0` = push left
  - `1` = no push (coast)
  - `2` = push right

### Question 11 — Does random play ever solve it? Why or why not?

A random agent almost never solves MountainCar. The problem requires a **deliberate sequence** of actions (left, then right, then left, then right, building oscillation) over many steps. A random agent has roughly a 1-in-3 chance of pushing in the right direction at each step, so the probability of producing the exact coordinated sequence needed to reach the top is astronomically small. In practice, the episode always times out at 200 steps without reaching the flag.

---

## Part 4: Big-Picture Reflection

### Question 12 — What is the agent's goal in any RL problem?

The agent's goal is to choose actions that maximise the total cumulative reward it receives over the course of an episode (or over its entire lifetime). Importantly, the agent does not just try to get a high reward right now — it must consider future consequences of its actions, since a short-term gain might lead to a low reward later.

### Question 13 — Why is "random" not a good strategy — and what would the agent need to do instead?

A random strategy ignores all the information in the observation. It does not improve even after seeing what worked and what failed. A real agent needs a **policy** that maps observations to actions based on learned experience — it should try an action, see the result (reward), and update its behaviour to increase the chance of repeating good actions and avoiding bad ones. In short, it needs memory of past outcomes and a mechanism to adjust its decisions accordingly.

### Question 14 — What does it mean for an agent to "learn"?

For an agent to "learn" means that its policy — the mapping from states to actions — changes over time in a way that produces higher rewards. In episode 1 the agent might act randomly. By episode 100, the same situation (same state) should trigger a different, better-chosen action because the agent has seen what happens when it takes various actions in that situation and has updated its preferences. Concretely: the agent's internal parameters (weights of a neural network, or entries in a table) shift to make high-reward actions more likely.

### Question 15 — Three questions for my mentor

1. In the hand-coded rule, I only used `pole_angle`. How much better would a rule that also uses `cart_position` and `cart_velocity` do — and is there a principled way to design it (e.g., a PID controller)?

2. MountainCar is famously hard for RL because the reward is so sparse (you only get it at the very end). How do real RL algorithms (like PPO) deal with sparse rewards — do they use reward shaping, or something else?

3. When we eventually train a policy network on CartPole, will it discover the same "follow the pole angle" logic on its own, or does it learn something completely different?
