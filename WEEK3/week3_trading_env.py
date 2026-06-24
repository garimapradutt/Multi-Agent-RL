"""
week3_trading_env.py
--------------------
A minimal custom Gymnasium trading environment for a single asset
following a random walk (log-normal returns).

Observation (2-D):
    [last_price_return, current_position]
    position in {-1.0 (short), 0.0 (flat), +1.0 (long)}

Actions (discrete 3):
    0 = Go Long   (buy / stay long)
    1 = Hold      (keep current position)
    2 = Go Short  (sell / stay short)

Reward:
    current_position * price_return * 100
    (positive if position matches price direction, negative otherwise)

Run as a script to execute a quick random-agent sanity check:
    python week3_trading_env.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ToyTradingEnv(gym.Env):
    """
    Simplest possible trading environment.

    Dynamics:
        - Asset price follows a random walk (log-normal returns).
        - Episode length is fixed at 'episode_length' steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int   = 100,
        initial_price:  float = 100.0,
        daily_vol:      float = 0.01,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.initial_price  = initial_price
        self.daily_vol      = daily_vol

        # Observation: [last_price_return, current_position]
        # Returns can be any real number; position is in [-1, +1]
        self.observation_space = spaces.Box(
            low  = np.array([-np.inf, -1.0], dtype=np.float32),
            high = np.array([ np.inf,  1.0], dtype=np.float32),
        )

        # Three discrete actions: Long / Hold / Short
        self.action_space = spaces.Discrete(3)  # 0=long, 1=hold, 2=short

        # State variables (initialised in reset)
        self.price      = None
        self.position   = None
        self.step_count = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the start of a new episode.

        Returns initial observation and empty info dict.
        The agent always starts flat (position = 0) at the initial price.
        """
        super().reset(seed=seed)          # seeds self.np_random

        self.price      = self.initial_price
        self.position   = 0.0             # start flat (no position)
        self.step_count = 0

        obs = np.array([0.0, self.position], dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        """
        Apply action, advance the market by one step, return transition.

        Price dynamics
        --------------
        We model daily returns as a Gaussian with mean 0 and std daily_vol.
        This is the discrete-time equivalent of Geometric Brownian Motion —
        the standard null hypothesis for an efficient market with no drift.

        Reward
        ------
        reward = position * price_return * 100

        If we are long (+1) and the price goes up (positive return), we earn.
        If we are long and the price goes down, we lose.
        Short (-1) earns on negative returns and loses on positive ones.
        Flat (0) earns nothing regardless of price movement.
        Multiplying by 100 scales the return (e.g. 0.01 return → 1.0 reward).
        """
        # ── Simulate one day of price movement ────────────────────────────
        price_return  = float(self.np_random.normal(0.0, self.daily_vol))
        self.price   *= (1.0 + price_return)

        # ── Map action integer to new position ─────────────────────────────
        action_to_position = {
            0:  1.0,           # Long
            1:  self.position, # Hold (keep whatever we have)
            2: -1.0,           # Short
        }
        new_position = action_to_position[int(action)]

        # ── Reward: old position × today's price move × 100 ───────────────
        reward = self.position * price_return * 100.0

        # Advance position and step counter
        self.position    = new_position
        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated  = False

        obs = np.array([price_return, self.position], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        """Simple text render (not used for training)."""
        print(f"  step={self.step_count:3d}  price={self.price:.2f}  pos={self.position:+.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — random agent
# ─────────────────────────────────────────────────────────────────────────────

def run_random_agent(num_episodes: int = 5) -> float:
    """Run a random agent and print per-episode rewards."""
    env = ToyTradingEnv()
    total_rewards = []

    print("=" * 55)
    print("ToyTradingEnv — Random Agent")
    print("=" * 55)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: total reward = {ep_reward:.2f}")

    avg = np.mean(total_rewards)
    print(f"\nAverage over {num_episodes} episodes: {avg:.2f}")
    print("(Expected ~0 on a random walk with no drift)\n")
    env.close()
    return avg


if __name__ == "__main__":
    random_avg = run_random_agent(num_episodes=10)
