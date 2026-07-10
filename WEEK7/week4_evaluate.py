"""
week4_evaluate.py
-----------------
Evaluates a PPO agent trained on TradingEnvV2 (the extended environment from
Task 2, with the price-history window and transaction costs) against two
baselines:

  1. A random agent
  2. A buy-and-hold agent (always stays Long)

This is the "properly evaluate the agent" step: instead of just watching the
training reward curve go up, we run the trained policy on fresh episodes and
look at its cumulative P&L against baselines.

Run:
    python week4_evaluate.py

Outputs:
    - plots/cumulative_pnl.png       (transaction_cost=0.1, the default)
    - plots/cumulative_pnl_zerocost.png  (transaction_cost=0.0, for comparison)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from week4_trading_env_v2 import TradingEnvV2


def run_episode(env, model=None):
    """
    Run one episode and return the list of per-step rewards.

    model=None means a random agent. When a model IS provided, we use
    deterministic=True to select actions.

    Why deterministic=True here but NOT during training?
    ------------------------------------------------------
    During training, PPO needs deterministic=False (the default) so it keeps
    SAMPLING from its action distribution -- that randomness is exploration,
    letting the agent discover better actions it hasn't tried yet. During
    evaluation we want to see what the agent's BEST current policy actually
    does, with no extra randomness muddying the result, so we always pick
    the highest-probability action instead of sampling. Evaluating with
    deterministic=False would make two evaluation runs of the same trained
    model look different purely from action-sampling noise, which would
    make it harder to tell "the policy changed" from "we got unlucky/lucky
    samples this run."
    """
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


def buy_and_hold_episode(env):
    """Always stay long (action=0). Returns the list of per-step rewards."""
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


def average_cumulative_pnl(env, n_episodes=50, model=None, strategy="random"):
    """
    Return the MEAN cumulative P&L curve (length = episode_length) over
    n_episodes fresh episodes.

    np.cumsum(step_rewards) turns a list of per-step rewards into a running
    total: index i is "total P&L earned from step 0 through step i". Doing
    this for many episodes and averaging element-wise gives a curve of
    "expected P&L so far, as a function of how far into the episode we are."
    """
    all_cumulative = []
    for _ in range(n_episodes):
        if strategy == "buy_and_hold":
            step_rewards = buy_and_hold_episode(env)
        else:
            step_rewards = run_episode(env, model=model)
        all_cumulative.append(np.cumsum(step_rewards))

    min_len = min(len(c) for c in all_cumulative)
    stacked = np.array([c[:min_len] for c in all_cumulative])
    return stacked.mean(axis=0)


def plot_pnl(pnl_trained, pnl_random, pnl_bnh, filename="plots/cumulative_pnl.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    steps = np.arange(len(pnl_trained))

    plt.figure(figsize=(9, 4))
    plt.plot(steps, pnl_trained, color="steelblue", label="PPO (trained)")
    plt.plot(steps, pnl_random, color="darkorange", label="Random agent")
    plt.plot(steps, pnl_bnh, color="green", label="Buy and hold")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Cumulative P&L (mean over episodes)")
    plt.title("Agent comparison on TradingEnvV2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved to {filename}")


def train_and_evaluate(transaction_cost=0.1, total_timesteps=100_000, n_eval_episodes=50):
    """Train PPO on TradingEnvV2 and evaluate it + two baselines. Returns the
    three final cumulative P&L numbers as a dict."""
    env = TradingEnvV2(transaction_cost=transaction_cost)
    print(f"Training PPO on TradingEnvV2 (transaction_cost={transaction_cost}) ...")
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps)
    print("Training complete.")

    env_eval = TradingEnvV2(transaction_cost=transaction_cost)
    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=n_eval_episodes, model=model, strategy="trained")
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=n_eval_episodes, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=n_eval_episodes, model=None, strategy="buy_and_hold")

    print(f"Final cumulative P&L (mean over {n_eval_episodes} episodes, "
          f"transaction_cost={transaction_cost}):")
    print(f"  PPO trained:  {pnl_trained[-1]:+8.2f}")
    print(f"  Random agent: {pnl_random[-1]:+8.2f}")
    print(f"  Buy and hold: {pnl_bnh[-1]:+8.2f}")

    return model, pnl_trained, pnl_random, pnl_bnh


if __name__ == "__main__":
    # ── Main run: default transaction_cost=0.1 ──────────────────────────────
    model, pnl_trained, pnl_random, pnl_bnh = train_and_evaluate(transaction_cost=0.1)
    plot_pnl(pnl_trained, pnl_random, pnl_bnh, filename="plots/cumulative_pnl.png")

    # ── Task 3, step 3: re-run with transaction_cost=0.0 for comparison ─────
    print("\n--- Re-running with transaction_cost=0.0 for comparison ---")
    _, pnl_trained0, pnl_random0, pnl_bnh0 = train_and_evaluate(transaction_cost=0.0)
    plot_pnl(pnl_trained0, pnl_random0, pnl_bnh0,
             filename="plots/cumulative_pnl_zerocost.png")

    print("\nDone.")
