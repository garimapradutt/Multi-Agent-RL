"""
week5_train_and_evaluate.py
-----------------------------
Train PPO on the momentum market (rho=0.5) and evaluate it exactly the way
Week 4 evaluated the random-walk agent -- same harness, same baselines.
The only thing that changed since Week 4 is the environment's rho.

A wrinkle discovered while building this: the raw state features (price
returns) are tiny in scale (~+/-0.01, since daily_vol=0.01), while PPO's
default network initialization expects roughly unit-scale inputs. With raw
observations, PPO reliably converged to a degenerate fixed policy (e.g.
"always long" or "always flat") that never actually looks at the sign of
the last return -- verified by probing the trained policy with synthetic
inputs. Wrapping the environment in SB3's VecNormalize (which standardizes
observations to roughly zero-mean, unit-variance using a running estimate)
fixed this: the same reward/env/action-space, just rescaled inputs, and the
policy reliably learns to follow the sign of the last return.

Run:
    python week5_train_and_evaluate.py

Outputs:
    plots/cumulative_pnl_momentum.png
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from week5_trading_env_v3 import TradingEnvV3
from week4_evaluate import average_cumulative_pnl, plot_pnl


class NormalizedPredictor:
    """
    Adapts a PPO model trained inside a VecNormalize wrapper so it can be
    dropped into week4_evaluate.py's average_cumulative_pnl unchanged.
    That function calls `model.predict(obs, deterministic=True)` on a RAW
    (unnormalized) observation from a plain env -- this wrapper normalizes
    the observation first (using the frozen running stats from training),
    then forwards to the real model, so it satisfies the same interface.
    """

    def __init__(self, model, vec_normalize):
        self.model = model
        self.vec_normalize = vec_normalize
        self.vec_normalize.training = False  # freeze running stats at eval time

    def predict(self, obs, deterministic=True):
        norm_obs = self.vec_normalize.normalize_obs(obs[None, :])[0]
        return self.model.predict(norm_obs, deterministic=deterministic)


def train_ppo_normalized(rho, transaction_cost, total_timesteps, seed=7):
    """Train PPO with observation normalization; return a NormalizedPredictor."""
    def make_env():
        return TradingEnvV3(rho=rho, transaction_cost=transaction_cost)

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


def inspect_policy(env, predictor, n_steps=20):
    """
    Run one episode step by step and print the newest return alongside the
    position the agent chose, so we can compare its behaviour to the
    hand-coded rule "hold the position that matches the sign of the last
    return" (the theoretically optimal rule for rho > 0).
    """
    print("\nStep-by-step policy inspection (last_return -> position):")
    obs, _ = env.reset()
    done, step = False, 0
    while not done and step < n_steps:
        action, _ = predictor.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  last return {obs[-2]:+.4f} -> position {obs[-1]:+.0f}")
        done = terminated or truncated
        step += 1


if __name__ == "__main__":
    print("Training PPO on the momentum market (rho = 0.5) ...")
    predictor = train_ppo_normalized(rho=0.5, transaction_cost=0.1, total_timesteps=100_000)
    print("Training complete.")

    env_eval = TradingEnvV3(rho=0.5, transaction_cost=0.1)
    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=50, model=predictor, strategy="trained")
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="buy_and_hold")

    print("Final cumulative P&L (mean over 50 episodes):")
    print(f"  PPO trained:  {pnl_trained[-1]:+8.2f}")
    print(f"  Random agent: {pnl_random[-1]:+8.2f}")
    print(f"  Buy and hold: {pnl_bnh[-1]:+8.2f}")

    plot_pnl(pnl_trained, pnl_random, pnl_bnh,
             filename="plots/cumulative_pnl_momentum.png")

    inspect_policy(env_eval, predictor)

    predictor.model.save("ppo_momentum_model")
    print("\nModel saved to ppo_momentum_model.zip")
