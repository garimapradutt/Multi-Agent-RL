"""
week3_ppo.py
------------
Trains a PPO agent on CartPole-v1 using Stable-Baselines3.

Run:
    python week3_ppo.py

Outputs:
    - Evaluation score printed to stdout
    - ppo_returns.npy  (episode rewards, needed by week3_compare.py)
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardLogger(BaseCallback):
    """
    Custom SB3 callback that records the total reward of each completed episode.

    How it works:
      SB3 calls _on_step() after every environment step. We accumulate the
      reward at each step; when 'dones' is True, the episode has ended and
      we append the accumulated total to episode_rewards, then reset.
    """

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # 'rewards' and 'dones' are 1-element lists for a single env
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True   # returning True tells SB3 to continue training


def train_ppo(env_name="CartPole-v1", total_timesteps=50_000):
    """
    Train a PPO agent for total_timesteps environment steps.

    Key PPO hyperparameters:
      n_steps=2048
        The number of steps PPO collects from the environment before doing
        ANY gradient updates. PPO is an on-policy algorithm: it gathers a
        batch of experience, learns from it, then discards it and gathers
        a new batch. Larger n_steps gives lower-variance gradient estimates
        but delays learning. 2048 is SB3's default — a good balance.

      batch_size=64
        After collecting n_steps of experience, PPO shuffles those steps and
        trains on mini-batches of size batch_size. This reuses the collected
        data multiple times (for n_epochs passes) and makes GPU usage
        efficient. batch_size must divide n_steps evenly.

      The EpisodeRewardLogger class:
        A lightweight hook into SB3's training loop. At each environment step
        it records the reward; when the episode ends ('dones'=True) it saves
        the cumulative episode reward so we can plot the learning curve.
    """
    env      = gym.make(env_name)
    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",          # policy architecture: Multi-Layer Perceptron
        env,
        verbose=0,
        learning_rate=3e-4,   # Adam learning rate for both actor and critic
        n_steps=2048,         # see comment above
        batch_size=64,        # see comment above
        gamma=0.99,           # discount factor
        n_epochs=10,          # how many passes through each collected batch
        seed=42,
    )

    print(f"Training PPO on {env_name} for {total_timesteps:,} timesteps ...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"PPO evaluation: {mean_reward:.1f} +/- {std_reward:.1f}")
    env.close()

    return callback.episode_rewards, model


if __name__ == "__main__":
    ppo_returns, model = train_ppo(total_timesteps=50_000)

    # Save for week3_compare.py
    np.save("ppo_returns.npy", np.array(ppo_returns))
    print(f"Saved {len(ppo_returns)} episode returns to ppo_returns.npy")
    print("Done ✅")
