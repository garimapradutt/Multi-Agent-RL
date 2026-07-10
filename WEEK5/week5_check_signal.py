"""
week5_check_signal.py
-----------------------
Before training anything, verify that the AR(1) structure we THINK we added
to TradingEnvV3 is actually there, by measuring it directly from the data.
This is the same habit we'll need in Week 6 with real data: check the data
before blaming the algorithm.

Run:
    python week5_check_signal.py
"""
import numpy as np
from week5_trading_env_v3 import TradingEnvV3

def collect_returns(env, n_steps=10_000):
    """
    Step through the env with Hold actions and record each observed return.

    We use Hold (action=1) specifically because the environment's price
    process doesn't depend on the agent's action at all -- price_return is
    generated purely from rho and noise, regardless of what the agent does.
    So any fixed action sequence (all Hold, all Long, random) would produce
    the exact same sequence of returns; Hold is just the simplest choice
    since it also keeps the position fixed and out of the way.
    """
    returns = []
    obs, _ = env.reset(seed=0)
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(1)  # Hold
        returns.append(obs[-2])  # newest return in the history window
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(returns)

def lag1_autocorrelation(returns):
    """
    Correlation between each return and the one immediately after it.
    np.corrcoef(returns[:-1], returns[1:]) builds a 2x2 correlation matrix
    between the "returns shifted back by one" series and the "returns
    shifted forward by one" series; [0, 1] pulls out the off-diagonal
    entry, i.e. corr(r_t, r_{t+1}) across the whole sequence.
    """
    return np.corrcoef(returns[:-1], returns[1:])[0, 1]

def random_agent_mean_reward(env, n_episodes=20):
    totals = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


if __name__ == "__main__":
    print("=" * 70)
    print(f"{'rho':>6} | {'measured autocorr':>18} | {'random agent mean reward':>26}")
    print("=" * 70)
    for rho in [0.0, 0.5, -0.5]:
        env = TradingEnvV3(rho=rho, transaction_cost=0.0)
        r = collect_returns(env)
        ac = lag1_autocorrelation(r)
        mean_rand = random_agent_mean_reward(TradingEnvV3(rho=rho))
        print(f"{rho:+6.2f} | {ac:+18.3f} | {mean_rand:+26.2f}")
