"""
week4_evaluate.py
-----------------
Evaluates the trained PPO trading agent against:
  1. A random agent
  2. A buy-and-hold agent

Generates:
  plots/cumulative_pnl.png  — cumulative P&L curves for all three strategies

Run:
    python week4_evaluate.py

(Assumes ppo_trading_model.zip exists from week4_train_ppo_trading.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from week3_trading_env import ToyTradingEnv


def run_agent_episode(env, model=None, strategy="ppo", seed=0):
    """
    Run one episode and return a list of per-step rewards (P&L increments).

    strategy:
      'ppo'    — use the PPO model to select actions
      'random' — sample uniformly from the action space
      'bah'    — buy-and-hold: always stay Long (action=0)
    """
    obs, _ = env.reset(seed=seed)
    pnl_steps, done = [], False

    while not done:
        if strategy == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif strategy == "random":
            action = env.action_space.sample()
        else:  # buy-and-hold
            action = 0  # always Long

        obs, reward, terminated, truncated, _ = env.step(action)
        pnl_steps.append(reward)
        done = terminated or truncated

    return pnl_steps


def evaluate_all(model, num_episodes=50, seed_offset=1000):
    """
    Run num_episodes episodes for each strategy and collect cumulative P&L.
    Returns arrays of shape (num_episodes, episode_length).
    """
    env = ToyTradingEnv()
    episode_length = env.episode_length

    ppo_curves    = np.zeros((num_episodes, episode_length))
    random_curves = np.zeros((num_episodes, episode_length))
    bah_curves    = np.zeros((num_episodes, episode_length))

    for ep in range(num_episodes):
        seed = seed_offset + ep

        for arr, strat, mdl in [
            (ppo_curves,    "ppo",    model),
            (random_curves, "random", None),
            (bah_curves,    "bah",    None),
        ]:
            steps = run_agent_episode(env, model=mdl, strategy=strat, seed=seed)
            cumulative = np.cumsum(steps)
            arr[ep] = cumulative

    env.close()
    return ppo_curves, random_curves, bah_curves


def plot_cumulative_pnl(
    ppo_curves, random_curves, bah_curves,
    filename="plots/cumulative_pnl.png",
):
    """
    Plot mean ± 1-std cumulative P&L for each strategy over all eval episodes.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = np.arange(ppo_curves.shape[1])

    def plot_band(ax, curves, color, label):
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_band(ax, ppo_curves,    "steelblue",  "PPO agent")
    plot_band(ax, random_curves, "darkorange", "Random agent")
    plot_band(ax, bah_curves,    "seagreen",   "Buy-and-Hold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Cumulative P&L Comparison (mean ± 1 std, 50 episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved cumulative P&L plot to {filename}")


def print_summary(ppo_curves, random_curves, bah_curves):
    """Print final episode P&L statistics for all three strategies."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (final cumulative P&L per episode)")
    print("=" * 60)
    for label, curves in [
        ("PPO agent",    ppo_curves),
        ("Random agent", random_curves),
        ("Buy-and-Hold", bah_curves),
    ]:
        final_pnl = curves[:, -1]  # cumulative P&L at last step
        print(f"  {label:<16}  mean={final_pnl.mean():+8.2f}  "
              f"std={final_pnl.std():6.2f}  "
              f"min={final_pnl.min():+8.2f}  max={final_pnl.max():+8.2f}")
    print("=" * 60)
    print("\nNote: On a pure random walk, all three strategies should")
    print("have a mean P&L close to 0.  This is the CORRECT result —")
    print("no strategy can reliably profit from unpredictable prices.\n")


if __name__ == "__main__":
    # Load the trained model
    print("Loading PPO model ...")
    model = PPO.load("ppo_trading_model")

    # Evaluate
    print("Running 50 evaluation episodes per strategy ...")
    ppo_c, rand_c, bah_c = evaluate_all(model, num_episodes=50)

    # Plot & summarise
    plot_cumulative_pnl(ppo_c, rand_c, bah_c)
    print_summary(ppo_c, rand_c, bah_c)
