"""
week6_historical_env.py
-------------------------
Trading environment that replays real historical returns instead of a
formula. Observation, actions, and reward are identical to TradingEnvV3
(Week 5) -- only the source of the returns has changed.

Each episode picks a random contiguous window of `episode_length` returns
from the array passed in. The train/test split happens OUTSIDE this class:
construct one env with the training slice of returns and a separate env
with the test slice (see week6_train_test.py).

Run as a script for a sanity check:
    python week6_historical_env.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class HistoricalTradingEnv(gym.Env):
    """Trading environment that replays real historical returns."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,               # 1-D numpy array of returns
        episode_length=100,
        history_len=5,
        transaction_cost=0.1,
        reward_scale=100.0,
    ):
        super().__init__()
        returns = np.asarray(returns, dtype=np.float64)
        assert len(returns) > episode_length + 1, "not enough data"
        self.returns = returns
        self.episode_length = episode_length
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale

        obs_dim = history_len + 1
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(3)

        self.position = None
        self.step_count = None
        self._start = None
        self._price_history = None

    # ---------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a random contiguous window of history for this episode
        last_valid_start = len(self.returns) - self.episode_length - 1
        self._start = int(self.np_random.integers(0, last_valid_start))
        # Episodes start at a RANDOM position (rather than always index 0)
        # for two reasons: (1) variety -- with one fixed history, always
        # starting at day 0 would mean the agent trains on the exact same
        # single episode over and over; random starts give it ~2000
        # overlapping-but-different windows to learn from. (2) it also
        # means evaluation over many episodes samples many different
        # stretches of history rather than a single lucky/unlucky one.

        self.position = 0.0
        self.step_count = 0
        self._price_history = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        return self._get_obs(), {}

    # ---------------------------------------------------------------- #
    def step(self, action):
        # The next return comes from HISTORY, not from a formula
        price_return = float(self.returns[self._start + self.step_count])
        self._price_history.append(price_return)

        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]

        pnl = self.position * price_return * self.reward_scale
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


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check():
    import numpy as np
    from gymnasium.utils.env_checker import check_env

    returns = np.load("data/returns.npy")

    print("=" * 60)
    print("check_env sanity check (HistoricalTradingEnv)")
    print("=" * 60)
    check_env(HistoricalTradingEnv(returns), warn=True)
    print("check_env passed (the +/-inf bound warning above is expected).\n")

    print("=" * 60)
    print("3 random-agent episodes on real data")
    print("=" * 60)
    env = HistoricalTradingEnv(returns)
    for ep in range(3):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            assert np.isfinite(total), "reward became non-finite!"
            done = terminated or truncated
        print(f"Episode {ep + 1}: total reward = {total:+.2f}")
    env.close()


if __name__ == "__main__":
    run_sanity_check()
