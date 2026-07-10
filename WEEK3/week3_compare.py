"""
week3_compare.py
----------------
Loads the saved episode returns from REINFORCE (Week 2) and PPO (this week),
then produces a side-by-side comparison plot.

Run:
    python week3_compare.py

(Assumes reinforce_returns.npy and ppo_returns.npy are in the same directory.)

Output:
    plots/reinforce_vs_ppo.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_comparison(
    reinforce_returns,
    ppo_returns,
    window=10,
    filename="plots/reinforce_vs_ppo.png",
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── REINFORCE ────────────────────────────────────────────────────────────
    r = np.array(reinforce_returns)
    ax.plot(r, alpha=0.15, color="steelblue")
    ax.plot(
        moving_average(r, window),
        color="steelblue",
        linewidth=2,
        label=f"REINFORCE ({window}-ep avg)",
    )

    # ── PPO ──────────────────────────────────────────────────────────────────
    p = np.array(ppo_returns)
    ax.plot(p, alpha=0.15, color="darkorange")
    ax.plot(
        moving_average(p, window),
        color="darkorange",
        linewidth=2,
        label=f"PPO ({window}-ep avg)",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE vs PPO on CartPole-v1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {filename}")


if __name__ == "__main__":
    reinforce_returns = np.load("reinforce_returns.npy").tolist()
    ppo_returns       = np.load("ppo_returns.npy").tolist()

    print(f"REINFORCE: {len(reinforce_returns)} episodes, "
          f"final 10-ep avg = {np.mean(reinforce_returns[-10:]):.1f}")
    print(f"PPO      : {len(ppo_returns)} episodes, "
          f"final 10-ep avg = {np.mean(ppo_returns[-10:]):.1f}")

    plot_comparison(reinforce_returns, ppo_returns,
                    window=10, filename="plots/reinforce_vs_ppo.png")
