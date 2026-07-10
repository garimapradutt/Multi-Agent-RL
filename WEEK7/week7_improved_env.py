"""
week7_improved_env.py
-----------------------
Final-project upgraded trading environment, built on top of Week 6's
HistoricalTradingEnv. Two upgrades from the assignment's menu:

  A. Richer state features -- two features added beyond the raw 5-return
     window: rolling volatility (std of the last `vol_window` returns) and
     the current price's distance from its own rolling moving average
     (`ma_window`-day). Rationale: on real data the raw last-5-returns
     window carries almost no signal (Week 6 measured autocorrelation of
     only -0.12), so the agent has little to work with. Volatility and
     trend-distance are classic technical features that summarise LONGER-
     horizon information the raw window can't see, without needing a much
     bigger raw window (which would mostly be noise). Expected behaviour:
     the agent might learn to reduce position size when rolling volatility
     is high (risk management), even if it can't predict direction.

  B. Position sizing -- 5 actions instead of 3: full long, half long, flat,
     half short, full short (positions +1, +0.5, 0, -0.5, -1). The existing
     transaction-cost term (proportional to |position change|) already
     handles fractional position changes with no modification needed.
     Expected behaviour: the agent might use half-sized positions when
     evidence is weak/mixed, rather than being forced to bet a full unit
     every time it wants any exposure at all.

Lookahead-bias check: every feature at step t is computed using only
returns/prices up to and INCLUDING the return that just occurred at step
t -- exactly the same timing convention as the raw history window in
Week 5/6 (append the new return, THEN read it back out in the observation
for the NEXT decision). Changing a future return does not change any
feature computed at an earlier step; see `run_lookahead_check()` below.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class ImprovedTradingEnv(gym.Env):
    """
    Observation (history_len + 3 dimensional):
        [r_{t-history_len+1}, ..., r_t, rolling_volatility, ma_distance,
         current_position]

    Actions (discrete, 5 -- upgrade B):
        0 = full long   (+1.0)
        1 = half long    (+0.5)
        2 = flat         (0.0)
        3 = half short   (-0.5)
        4 = full short   (-1.0)

    Reward (unchanged from Week 4-6):
        position * price_return * reward_scale
        - transaction_cost * |new_position - old_position|
    """

    metadata = {"render_modes": []}

    ACTION_TO_POSITION = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}

    def __init__(
        self,
        returns,
        episode_length=100,
        history_len=5,
        vol_window=20,
        ma_window=20,
        transaction_cost=0.1,
        reward_scale=100.0,
        initial_price=100.0,
    ):
        super().__init__()
        returns = np.asarray(returns, dtype=np.float64)
        assert len(returns) > episode_length + 1, "not enough data"
        self.returns = returns
        self.episode_length = episode_length
        self.history_len = history_len
        self.vol_window = vol_window
        self.ma_window = ma_window
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale
        self.initial_price = initial_price

        obs_dim = history_len + 3  # raw window + volatility + ma_distance + position
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(5)

        self.price = None
        self.position = None
        self.step_count = None
        self._start = None
        self._raw_history = None
        self._return_window = None
        self._price_window = None

    # ---------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        last_valid_start = len(self.returns) - self.episode_length - 1
        self._start = int(self.np_random.integers(0, last_valid_start))

        self.price = self.initial_price
        self.position = 0.0
        self.step_count = 0
        self._raw_history = deque([0.0] * self.history_len, maxlen=self.history_len)
        self._return_window = deque([0.0] * self.vol_window, maxlen=self.vol_window)
        self._price_window = deque([self.initial_price] * self.ma_window, maxlen=self.ma_window)

        return self._get_obs(), {}

    # ---------------------------------------------------------------- #
    def step(self, action):
        price_return = float(self.returns[self._start + self.step_count])
        self.price *= (1.0 + price_return)

        # Update all rolling buffers with the return/price that just
        # happened -- this is the only point where "new" information
        # enters the features, and it happens BEFORE we read them back
        # out in _get_obs() for the NEXT decision (see module docstring).
        self._raw_history.append(price_return)
        self._return_window.append(price_return)
        self._price_window.append(self.price)

        new_position = self.ACTION_TO_POSITION[int(action)]

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
        raw = np.array(list(self._raw_history), dtype=np.float32)
        rolling_vol = float(np.std(self._return_window))
        ma = float(np.mean(self._price_window))
        ma_distance = (self.price - ma) / ma if ma != 0 else 0.0
        return np.concatenate([
            raw,
            np.array([rolling_vol, ma_distance, self.position], dtype=np.float32),
        ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_lookahead_check():
    """
    Verify no lookahead bias: compute the observation at step t for two
    return arrays that are identical UP TO AND INCLUDING index t, but
    differ afterward. The two observations at step t must be identical.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, size=200)
    modified = base.copy()
    modified[150:] = rng.normal(0, 0.01, size=50)  # change everything after day 149

    env_a = ImprovedTradingEnv(base, episode_length=100)
    env_b = ImprovedTradingEnv(modified, episode_length=100)
    # force both to start at the same window
    env_a.reset(seed=1)
    env_b.reset(seed=1)
    env_a._start = env_b._start = 10  # deterministic shared start for the test

    obs_a = env_a._get_obs()
    obs_b = env_b._get_obs()
    for _ in range(120):  # walk past the divergence point (day 149) safely
        if env_a._start + env_a.step_count >= 149:
            break
        obs_a, _, term_a, trunc_a, _ = env_a.step(1)  # Hold-ish (half long every step)
        obs_b, _, term_b, trunc_b, _ = env_b.step(1)
        assert np.allclose(obs_a, obs_b), "Lookahead bias detected!"

    print("Lookahead check passed: observations identical up to the point "
          "where the return arrays diverge.")


def run_sanity_check():
    from gymnasium.utils.env_checker import check_env

    returns = np.random.default_rng(0).normal(0, 0.01, size=1000)
    print("=" * 60)
    print("check_env sanity check (ImprovedTradingEnv)")
    print("=" * 60)
    check_env(ImprovedTradingEnv(returns), warn=True)
    print("check_env passed (the +/-inf bound warning above is expected).\n")

    print("=" * 60)
    print("3 random-agent episodes")
    print("=" * 60)
    env = ImprovedTradingEnv(returns)
    for ep in range(3):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            assert np.isfinite(total)
            done = terminated or truncated
        print(f"Episode {ep + 1}: total reward = {total:+.2f}")
    env.close()

    print(f"\nObservation space: {env.observation_space}  (dim={env.observation_space.shape})")
    print("Compare to Week 6's HistoricalTradingEnv: dim=6 (5 raw returns + position).")
    print(f"ImprovedTradingEnv adds 2 dims (rolling_vol, ma_distance) -> dim=8, "
          f"and the action space grows from Discrete(3) to Discrete(5).")


if __name__ == "__main__":
    run_sanity_check()
    print()
    run_lookahead_check()
