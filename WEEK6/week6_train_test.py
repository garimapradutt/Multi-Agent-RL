"""
week6_train_test.py
---------------------
The most important discipline of the whole project: train on EARLIER data,
evaluate on LATER data the agent has never seen.

Run:
    python week6_train_test.py

Outputs:
    plots/test_cumulative_pnl.png
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from week6_historical_env import HistoricalTradingEnv
from week4_evaluate import average_cumulative_pnl, plot_pnl


class NormalizedPredictor:
    """
    Same fix as Week 5: real daily returns are tiny in scale (~0.01), which
    makes it hard for PPO's default-initialized network to learn a clean
    decision boundary. VecNormalize standardizes observations during
    training; this wrapper re-applies that same (frozen) normalization at
    evaluation time so the model can still be dropped into
    week4_evaluate.py's average_cumulative_pnl, which calls
    model.predict(obs, deterministic=True) on RAW observations.
    """

    def __init__(self, model, vec_normalize):
        self.model = model
        self.vec_normalize = vec_normalize
        self.vec_normalize.training = False  # freeze stats at eval time

    def predict(self, obs, deterministic=True):
        norm_obs = self.vec_normalize.normalize_obs(obs[None, :])[0]
        return self.model.predict(norm_obs, deterministic=deterministic)


def chronological_split(returns, train_fraction=0.8):
    """
    First 80% of days -> training, last 20% -> testing.

    NEVER shuffle before splitting. Shuffling would scatter days from
    across the whole history into both the training and test sets, so the
    "test" set would contain days from inside and even AFTER the training
    period alongside it. The agent (and, just as importantly, the
    normalization statistics computed from training data) would then be
    fit using information about the volatility level, trend, and specific
    events of what's supposed to be "unseen future" data -- silently
    leaking future information into training. This is lookahead bias, and
    it would make the test-set performance numbers meaningless: a
    "backtest" built this way could look great purely because the model
    already implicitly knows what the test period looked like.
    """
    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]


def train_ppo_normalized(returns, total_timesteps, seed=7):
    def make_env():
        return HistoricalTradingEnv(returns)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", vec_env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return NormalizedPredictor(model, vec_env)


def run_experiment(train_returns, test_returns, seed, total_timesteps=100_000):
    """One full train+evaluate run. Returns the four final P&L numbers."""
    predictor = train_ppo_normalized(train_returns, total_timesteps, seed=seed)

    train_eval = HistoricalTradingEnv(train_returns)
    test_eval = HistoricalTradingEnv(test_returns)

    pnl_train = average_cumulative_pnl(
        train_eval, n_episodes=50, model=predictor, strategy="trained")
    pnl_test = average_cumulative_pnl(
        test_eval, n_episodes=50, model=predictor, strategy="trained")
    pnl_random = average_cumulative_pnl(
        test_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        test_eval, n_episodes=50, model=None, strategy="buy_and_hold")

    return pnl_train, pnl_test, pnl_random, pnl_bnh


if __name__ == "__main__":
    returns = np.load("data/returns.npy")
    train_returns, test_returns = chronological_split(returns)
    print(f"Train: {len(train_returns)} days | Test: {len(test_returns)} days")

    print("\n=== Run 1 (seed=7) ===")
    pnl_train, pnl_test, pnl_random, pnl_bnh = run_experiment(
        train_returns, test_returns, seed=7)

    print("Final mean cumulative P&L over 50 episodes:")
    print(f"  PPO on TRAIN data:  {pnl_train[-1]:+8.2f}")
    print(f"  PPO on TEST data:   {pnl_test[-1]:+8.2f}")
    print(f"  Random (test):      {pnl_random[-1]:+8.2f}")
    print(f"  Buy & hold (test):  {pnl_bnh[-1]:+8.2f}")

    plot_pnl(pnl_test, pnl_random, pnl_bnh,
             filename="plots/test_cumulative_pnl.png")

    # ── Task 3, step 4: run twice more with fresh seeds to see the spread ──
    print("\n=== Run 2 (seed=13) ===")
    _, pnl_test_2, _, _ = run_experiment(train_returns, test_returns, seed=13)
    print(f"  PPO on TEST data:   {pnl_test_2[-1]:+8.2f}")

    print("\n=== Run 3 (seed=99) ===")
    _, pnl_test_3, _, _ = run_experiment(train_returns, test_returns, seed=99)
    print(f"  PPO on TEST data:   {pnl_test_3[-1]:+8.2f}")

    print("\n" + "=" * 50)
    print("SUMMARY: PPO on TEST data across 3 seeds")
    print("=" * 50)
    print(f"  seed=7:  {pnl_test[-1]:+8.2f}")
    print(f"  seed=13: {pnl_test_2[-1]:+8.2f}")
    print(f"  seed=99: {pnl_test_3[-1]:+8.2f}")
