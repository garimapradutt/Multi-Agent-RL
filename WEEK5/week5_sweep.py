"""
week5_sweep.py
---------------
Explores the edge-vs-cost trade-off: is a signal worth trading, given a
transaction cost? Sweeps rho (signal strength) at fixed cost, then sweeps
cost at fixed rho.

Uses the same VecNormalize fix as week5_train_and_evaluate.py -- without
it, PPO tends to collapse to a fixed/flat policy on these small-scale
features and the sweep would just show noise instead of the expected
trends.

Run:
    python week5_sweep.py
"""

import numpy as np
from week5_trading_env_v3 import TradingEnvV3
from week5_train_and_evaluate import train_ppo_normalized
from week4_evaluate import average_cumulative_pnl


def train_and_score(rho, transaction_cost, total_timesteps=50_000, seed=7):
    """Train PPO, return the final mean cumulative P&L over 50 episodes."""
    predictor = train_ppo_normalized(
        rho=rho, transaction_cost=transaction_cost,
        total_timesteps=total_timesteps, seed=seed,
    )
    env_eval = TradingEnvV3(rho=rho, transaction_cost=transaction_cost)
    pnl = average_cumulative_pnl(env_eval, n_episodes=50,
                                  model=predictor, strategy="trained")
    return pnl[-1]


if __name__ == "__main__":
    print("--- Sweep 1: signal strength (transaction_cost = 0.1) ---")
    sweep1 = {}
    for rho in [0.0, 0.25, 0.5]:
        score = train_and_score(rho, transaction_cost=0.1)
        sweep1[rho] = score
        print(f"rho = {rho:.2f} | final mean cumulative P&L = {score:+8.2f}")

    print("\n--- Sweep 2: transaction cost (rho = 0.25) ---")
    sweep2 = {}
    for cost in [0.0, 0.1, 0.3]:
        score = train_and_score(0.25, transaction_cost=cost)
        sweep2[cost] = score
        print(f"cost = {cost:.2f} | final mean cumulative P&L = {score:+8.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Sweep 1 (cost fixed at 0.1):")
    for rho, score in sweep1.items():
        print(f"  rho={rho:.2f}: {score:+8.2f}")
    print("Sweep 2 (rho fixed at 0.25):")
    for cost, score in sweep2.items():
        print(f"  cost={cost:.2f}: {score:+8.2f}")
