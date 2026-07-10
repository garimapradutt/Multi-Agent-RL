"""
week5_trading_env_v3.py
------------------------
Extends TradingEnvV2 (Week 4) with an AR(1) return process, giving the
market actual structure for the first time.

Price dynamics:
    r_t = rho * r_{t-1} + epsilon_t,   epsilon_t ~ N(0, daily_vol)

    rho > 0  -> momentum       (returns tend to repeat their sign)
    rho = 0  -> random walk    (identical to Week 4)
    rho < 0  -> mean reversion (returns tend to flip their sign)

Observation, actions, and reward are otherwise IDENTICAL to Week 4's
TradingEnvV2 -- this is a controlled experiment where only one thing
changes.

Run as a script for a sanity check:
    python week5_trading_env_v3.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class TradingEnvV3(gym.Env):
    """
    Trading environment with an AR(1) return process.

    Observation (history_len + 1 dimensional):
        [r_{t-history_len+1}, ..., r_t, current_position]

    Actions (discrete):
        0 = Go Long, 1 = Hold, 2 = Go Short

    Reward:
        position * price_return * 100
        - transaction_cost * |new_position - old_position|
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int = 100,
        initial_price: float = 100.0,
        daily_vol: float = 0.01,
        history_len: int = 5,
        transaction_cost: float = 0.1,
        rho: float = 0.5,
    ):
        super().__init__()
        assert -1.0 < rho < 1.0, "rho must be in (-1, 1) or returns explode"

        self.episode_length = episode_length
        self.initial_price = initial_price
        self.daily_vol = daily_vol
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        self.rho = rho

        obs_dim = history_len + 1
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(3)

        self.price = None
        self.position = None
        self.step_count = None
        self._price_history = None
        self._last_return = None

    # ---------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.initial_price
        self.position = 0.0
        self.step_count = 0
        self._last_return = 0.0
        self._price_history = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        return self._get_obs(), {}

    # ---------------------------------------------------------------- #
    def step(self, action):
        # AR(1) price dynamics -- the ONLY change from Week 4
        noise = float(self.np_random.normal(0.0, self.daily_vol))
        price_return = self.rho * self._last_return + noise
        # What this line does for each sign of rho:
        #   rho > 0: today's return carries over a fraction `rho` of
        #            yesterday's return, so an up move tends to be followed
        #            by another up move (and down by down) -- momentum.
        #   rho = 0: price_return = noise only, i.e. independent of the
        #            past -- exactly the Week 4 random walk.
        #   rho < 0: today's return is pulled in the OPPOSITE direction of
        #            yesterday's, so an up move tends to be followed by a
        #            down move -- mean reversion.
        self._last_return = price_return

        self.price *= (1.0 + price_return)
        self._price_history.append(price_return)

        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]

        pnl = self.position * price_return * 100.0
        cost = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost

        self.position = new_position
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ---------------------------------------------------------------- #
    def _get_obs(self):
        history = np.array(list(self._price_history), dtype=np.float32)
        return np.append(history, self.position).astype(np.float32)

    # ---------------------------------------------------------------- #
    def render(self):
        print(
            f"Step {self.step_count:3d} | "
            f"Price {self.price:8.2f} | "
            f"Position {self.position:+.0f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check():
    from gymnasium.utils.env_checker import check_env

    print("=" * 60)
    print("check_env sanity check (TradingEnvV3)")
    print("=" * 60)
    check_env(TradingEnvV3(), warn=True)
    print("check_env passed (the +/-inf bound warning above is expected).\n")


def compare_rho_zero_to_random_walk(n_episodes=3):
    """
    Confirm TradingEnvV3(rho=0.0) behaves like a plain random walk --
    a random agent's reward scale should look similar to Week 4's env.
    """
    print("=" * 60)
    print("TradingEnvV3(rho=0.0) -- should match Week 4's random walk")
    print("=" * 60)
    env = TradingEnvV3(rho=0.0)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: total reward = {total_reward:+.2f}")
    env.close()
    print()


if __name__ == "__main__":
    run_sanity_check()
    compare_rho_zero_to_random_walk()
