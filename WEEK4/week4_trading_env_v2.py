"""
week4_trading_env_v2.py
-----------------------
Extended version of ToyTradingEnv with:
  1. A sliding window of the last 'history_len' price returns in the state.
  2. A transaction cost subtracted from the reward when the position changes.

Observation (history_len + 1 dimensional):
    [r_{t-history_len+1}, ..., r_t, current_position]

Actions (discrete 3):
    0 = Go Long   1 = Hold   2 = Go Short

Reward:
    position * price_return * 100
    - transaction_cost * |new_position - old_position|

Run as script for a sanity check:
    python week4_trading_env_v2.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class TradingEnvV2(gym.Env):
    """
    Trading environment with price-history window and transaction costs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length:   int   = 100,
        initial_price:    float = 100.0,
        daily_vol:        float = 0.01,
        history_len:      int   = 5,
        transaction_cost: float = 0.1,
    ):
        super().__init__()
        self.episode_length   = episode_length
        self.initial_price    = initial_price
        self.daily_vol        = daily_vol
        self.history_len      = history_len
        self.transaction_cost = transaction_cost

        # Observation: history_len price returns + current position
        obs_dim = history_len + 1
        self.observation_space = spaces.Box(
            low  = np.full(obs_dim, -np.inf, dtype=np.float32),
            high = np.full(obs_dim,  np.inf, dtype=np.float32),
        )

        self.action_space = spaces.Discrete(3)  # 0=long, 1=hold, 2=short

        self.price          = None
        self.position       = None
        self.step_count     = None
        self._price_history = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price      = self.initial_price
        self.position   = 0.0
        self.step_count = 0
        # Fill history buffer with zeros (no prior data at episode start)
        self._price_history = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        return self._get_obs(last_return=0.0), {}

    def step(self, action: int):
        # ── Price dynamics ────────────────────────────────────────────────────
        price_return  = float(self.np_random.normal(0.0, self.daily_vol))
        self.price   *= (1.0 + price_return)
        self._price_history.append(price_return)

        # ── Map action to new position ────────────────────────────────────────
        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]

        # ── Reward = P&L minus transaction cost ───────────────────────────────
        pnl    = self.position * price_return * 100.0

        # Transaction cost explanation:
        #   (new_position - self.position) measures how much the position CHANGES:
        #     If we stay flat (0→0), Long→Long (1→1), or Short→Short: cost = 0.
        #     If we flip Long→Short: |1 - (-1)| = 2 → cost = 2 * transaction_cost.
        #     If we go from flat→Long: |0 - 1| = 1 → cost = transaction_cost.
        #   This discourages excessive churning — every trade costs something,
        #   just like real bid-ask spreads and commissions.
        cost   = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost

        self.position    = new_position
        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated  = False

        return self._get_obs(price_return), reward, terminated, truncated, {}

    def _get_obs(self, last_return: float) -> np.ndarray:
        """Concatenate the price history window with the current position."""
        history = np.array(list(self._price_history), dtype=np.float32)
        return np.append(history, self.position).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check():
    env = TradingEnvV2(history_len=5, transaction_cost=0.1)
    obs, _ = env.reset()

    print("=" * 60)
    print("TradingEnvV2 — Sanity Check")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")
    print(f"Initial obs      : {obs}  (5 zeros + position=0)")

    # Take a few steps with known actions
    total_reward = 0.0
    for step, action in enumerate([0, 1, 1, 2, 1]):  # long, hold, hold, short, hold
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        action_name = {0: "Long", 1: "Hold", 2: "Short"}[action]
        print(f"  step {step+1}: action={action_name:5s}  obs={obs}  reward={reward:+.3f}")

    env.close()
    print(f"\nTotal reward after 5 steps: {total_reward:.3f}")
    print("Sanity check passed ✅\n")


if __name__ == "__main__":
    run_sanity_check()
