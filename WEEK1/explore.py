"""
Week 1 - Assignment 1: Explore CartPole
Author: (your name here)

This script:
  Part 1 - Runs a RANDOM agent on CartPole and observes how quickly the pole falls.
  Part 2 - Runs a HAND-CODED rule-based policy that does better than random.
  Part 3 - Runs a RANDOM agent on a second Gymnasium environment (MountainCar-v0).

No learning algorithms are used. The goal is pure observation and intuition.
"""

import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Watch a Random Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_random_agent(num_episodes: int = 5) -> None:
    """
    Run a completely random agent on CartPole-v1 and print how long it survives.
    render_mode is set to None so the script works on headless servers too.
    Change to "human" if you want to watch the window pop up.
    """
    env = gym.make("CartPole-v1", render_mode=None)
    obs, info = env.reset()

    print("=" * 55)
    print("PART 1 — Random Agent on CartPole-v1")
    print("=" * 55)
    print(f"Initial observation : {obs}")
    print("Observation layout  : [cart_pos, cart_vel, pole_angle, pole_ang_vel]")
    print(f"Action space        : {env.action_space}  (0 = push left, 1 = push right)")
    print()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()          # completely random
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {episode + 1}: lasted {steps:3d} steps | "
              f"total reward = {total_reward:.0f}")

    env.close()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Hand-Coded Rule Policy
# ─────────────────────────────────────────────────────────────────────────────

def my_policy(observation) -> int:
    """
    A hand-coded rule for CartPole-v1 that scores well above the random baseline.

    Observation = [cart_pos, cart_vel, pole_angle, pole_ang_vel]

    Strategy — Proportional + Derivative (PD) heuristic
    ----------------------------------------------------
    Instead of looking at the pole angle alone, we compute a weighted combination
    of the pole angle AND its angular velocity.  This anticipates where the pole
    is heading, not just where it is right now.

        signal = pole_angle + k * pole_ang_vel

    If signal > 0  →  push RIGHT (action 1)
    If signal ≤ 0  →  push LEFT  (action 0)

    The constant k = 0.1 was chosen by hand; it gives the velocity term
    about 10 % as much weight as the angle term.  Larger k makes the controller
    more "lookahead" (better at anticipating), but too large makes it overshoot.

    Why the minus sign is absent
    ----------------------------
    Positive pole_angle means the pole tilts right.  To bring it back, the cart
    must accelerate right (action 1) — which generates a leftward torque on the
    pole base.  So we push in the SAME direction as the tilt.
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    # PD-style weighted signal
    k = 0.1
    signal = pole_angle + k * pole_ang_vel

    return 1 if signal > 0 else 0


def run_rule_agent(num_episodes: int = 5) -> None:
    """Run the hand-coded rule policy for num_episodes and print the scores."""
    env = gym.make("CartPole-v1", render_mode=None)

    print("=" * 55)
    print("PART 2 — Hand-Coded Rule Policy on CartPole-v1")
    print("=" * 55)

    scores = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = my_policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        scores.append(total_reward)
        print(f"Episode {episode + 1}: total reward = {total_reward:.0f}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage over {num_episodes} episodes: {avg:.1f}")
    print("(Random baseline is ~20. Rule policy aims for ≥ 50.)\n")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Try a Different Environment (MountainCar-v0)
# ─────────────────────────────────────────────────────────────────────────────

def run_mountain_car(num_episodes: int = 3) -> None:
    """
    Run a random agent on MountainCar-v0 to observe a different RL problem.
    MountainCar gives -1 reward per step and truncates after 200 steps,
    so the total reward is always <= -1 per step.
    """
    env = gym.make("MountainCar-v0", render_mode=None)
    obs, _ = env.reset()

    print("=" * 55)
    print("PART 3 — Random Agent on MountainCar-v0")
    print("=" * 55)
    print(f"Observation space: {env.observation_space}  [position, velocity]")
    print(f"Action space     : {env.action_space}  (0=push left, 1=no push, 2=push right)")
    print()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        solved = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if terminated:
                solved = True

        status = "SOLVED!" if solved else "Did not reach the goal"
        print(f"Episode {episode + 1}: {steps:3d} steps | "
              f"total reward = {total_reward:.0f} | {status}")

    env.close()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_random_agent(num_episodes=5)
    run_rule_agent(num_episodes=5)
    run_mountain_car(num_episodes=3)
