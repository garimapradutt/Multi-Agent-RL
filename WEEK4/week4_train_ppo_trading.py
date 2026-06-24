"""
week4_train_ppo_trading.py
--------------------------
Trains PPO on the Week 3 ToyTradingEnv and compares it to a random baseline.

Run:
    python week4_train_ppo_trading.py

Outputs:
    - plots/ppo_trading_rewards.png   (training reward curve)
    - ppo_trading_returns.npy         (episode rewards, used by week4_evaluate.py)
    - ppo_trading_model/              (saved SB3 model)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from week3_trading_env import ToyTradingEnv


class EpisodeRewardLogger(BaseCallback):
    """Records the total reward of every completed episode during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def train_ppo_on_trading(total_timesteps=100_000, seed=42):
    """
    Train PPO on ToyTradingEnv and return the trained model + episode rewards.

    Key hyperparameter comments:
      n_steps=512
        Controls how many environment steps are collected before each PPO
        update. Smaller than the CartPole default (2048) because trading
        episodes are only 100 steps — with 512 we collect ~5 full episodes
        per batch, which gives a reasonable sample of trading outcomes before
        updating. Using 2048 would mean waiting for ~20 episodes between
        updates, slowing early learning.

      batch_size=64
        Mini-batch size for the SGD updates within each PPO iteration.
        The 512 collected steps are randomly shuffled and cut into chunks
        of 64 for gradient computation. Smaller batches introduce noise
        that can help escape local optima; larger batches are more stable
        but slower per update.
    """
    env      = ToyTradingEnv()
    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,          # see docstring above
        batch_size=64,        # see docstring above
        gamma=0.99,
        n_epochs=10,
        seed=seed,
    )

    print(f"Training PPO on ToyTradingEnv for {total_timesteps:,} timesteps ...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()

    print(f"Training complete. Episodes collected: {len(callback.episode_rewards)}")
    return model, callback.episode_rewards


def random_baseline(num_episodes=200, seed=0) -> float:
    """Run a random agent and return its average episode reward."""
    rng = np.random.default_rng(seed)
    env = ToyTradingEnv()
    rewards = []
    for _ in range(num_episodes):
        env.reset(seed=int(rng.integers(1e6)))
        ep_reward, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            _, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return float(np.mean(rewards))


def plot_rewards(episode_rewards, filename="plots/ppo_trading_rewards.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    rewards = np.array(episode_rewards)
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.2, color="steelblue", label="Episode reward")
    plt.plot(moving_average(rewards, 20), color="steelblue", linewidth=2,
             label="20-ep moving avg")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Zero")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on ToyTradingEnv")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved reward plot to {filename}")


if __name__ == "__main__":
    model, rewards = train_ppo_on_trading(total_timesteps=100_000)

    # Save rewards and model
    np.save("ppo_trading_returns.npy", np.array(rewards))
    model.save("ppo_trading_model")
    print("Model saved to ppo_trading_model.zip")

    # Plot
    plot_rewards(rewards)

    # Compare final performance vs random baseline
    ppo_final_avg    = np.mean(rewards[-20:])
    random_avg       = random_baseline(num_episodes=200)

    print(f"\nFinal 20-ep PPO average : {ppo_final_avg:.2f}")
    print(f"Random agent average    : {random_avg:.2f}")
    print(f"(On a random walk both should be near zero — that is expected!)")
