"""
week7_experiment.py
---------------------
Main experiment runner for the final project. For each of the two markets
(SPY, ^NSEI):

  1. Chronological 80/20 train/test split (Week 6 discipline, unchanged).
  2. Train the IMPROVED agent (ImprovedTradingEnv: richer features +
     5-action position sizing) for 3 seeds.
  3. Train a BASELINE agent (plain Week 6 HistoricalTradingEnv, same
     timestep budget) for 3 seeds -- isolates what the upgrades contributed.
  4. Evaluate all 4 strategies (improved, baseline, random, buy-and-hold)
     on the TEST split: final cumulative P&L, Sharpe ratio, max drawdown.
  5. Evaluate the improved agent on the TRAIN split too (overfitting check).
  6. Save one model per configuration, plus a 4-curve comparison plot.

Uses the same VecNormalize fix documented in Weeks 5-6 (real/synthetic
returns are tiny in scale, which otherwise causes PPO to collapse to a
degenerate fixed policy).

Run:
    python week7_experiment.py

Outputs:
    results/results.json               (all raw numbers)
    plots/<ticker>_four_strategy_pnl.png
    models/<ticker>_improved_seed<k>.zip, <ticker>_baseline_seed<k>.zip
"""

import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from week6_historical_env import HistoricalTradingEnv
from week7_improved_env import ImprovedTradingEnv
from week4_evaluate import average_cumulative_pnl, run_episode, buy_and_hold_episode
from week7_metrics import sharpe_ratio, max_drawdown

SEEDS = [7, 13, 99]
TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 50
MARKETS = ["SPY", "NSEI"]


class NormalizedPredictor:
    """Same VecNormalize fix as Weeks 5-6 -- see those notes for why."""

    def __init__(self, model, vec_normalize):
        self.model = model
        self.vec_normalize = vec_normalize
        self.vec_normalize.training = False

    def predict(self, obs, deterministic=True):
        norm_obs = self.vec_normalize.normalize_obs(obs[None, :])[0]
        return self.model.predict(norm_obs, deterministic=deterministic)


def chronological_split(returns, train_fraction=0.8):
    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]


def train_ppo_normalized(env_class, returns, total_timesteps, seed, **env_kwargs):
    def make_env():
        return env_class(returns, **env_kwargs)

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


def concat_rewards(env, model, strategy, n_episodes=N_EVAL_EPISODES):
    """Concatenate per-step rewards across n_episodes into one long series,
    for computing Sharpe ratio / max drawdown as a single "backtest"."""
    all_rewards = []
    for _ in range(n_episodes):
        if strategy == "buy_and_hold":
            all_rewards.extend(buy_and_hold_episode(env))
        elif strategy == "random":
            all_rewards.extend(run_episode(env, model=None))
        else:
            all_rewards.extend(run_episode(env, model=model))
    return np.array(all_rewards)


def evaluate_strategy(env, model, strategy, n_episodes=N_EVAL_EPISODES):
    rewards = concat_rewards(env, model, strategy, n_episodes)
    final_pnl = float(np.cumsum(rewards)[-1])
    return {
        "final_pnl": final_pnl,
        "sharpe": sharpe_ratio(rewards),
        "max_drawdown": max_drawdown(rewards),
    }


def mean_std(dicts, key):
    vals = [d[key] for d in dicts]
    return float(np.mean(vals)), float(np.std(vals))


def run_market(ticker, returns, results, models_dir="models"):
    train_returns, test_returns = chronological_split(returns)
    print(f"\n{'='*70}\n{ticker}: train={len(train_returns)} days, "
          f"test={len(test_returns)} days\n{'='*70}")

    market_results = {"train_days": len(train_returns), "test_days": len(test_returns)}

    # ── Improved agent: 3 seeds ─────────────────────────────────────────
    improved_test_metrics, improved_train_metrics = [], []
    improved_predictors = []
    for seed in SEEDS:
        print(f"[{ticker}] Training IMPROVED agent, seed={seed} ...")
        predictor = train_ppo_normalized(
            ImprovedTradingEnv, train_returns, TOTAL_TIMESTEPS, seed=seed)
        improved_predictors.append(predictor)

        test_env = ImprovedTradingEnv(test_returns)
        train_env = ImprovedTradingEnv(train_returns)
        improved_test_metrics.append(evaluate_strategy(test_env, predictor, "trained"))
        improved_train_metrics.append(evaluate_strategy(train_env, predictor, "trained"))
        print(f"  test final PnL={improved_test_metrics[-1]['final_pnl']:+.2f}  "
              f"train final PnL={improved_train_metrics[-1]['final_pnl']:+.2f}")

    improved_predictors[0].model.save(f"{models_dir}/{ticker}_improved_seed{SEEDS[0]}")

    # ── Baseline agent: 3 seeds (plain Week 6 env, same budget) ────────
    baseline_test_metrics = []
    baseline_predictors = []
    for seed in SEEDS:
        print(f"[{ticker}] Training BASELINE agent, seed={seed} ...")
        predictor = train_ppo_normalized(
            HistoricalTradingEnv, train_returns, TOTAL_TIMESTEPS, seed=seed)
        baseline_predictors.append(predictor)

        test_env = HistoricalTradingEnv(test_returns)
        baseline_test_metrics.append(evaluate_strategy(test_env, predictor, "trained"))
        print(f"  test final PnL={baseline_test_metrics[-1]['final_pnl']:+.2f}")

    baseline_predictors[0].model.save(f"{models_dir}/{ticker}_baseline_seed{SEEDS[0]}")

    # ── Random and buy-and-hold baselines (plain env, 3 repeats for random) ──
    random_metrics = []
    for _ in range(3):
        env = HistoricalTradingEnv(test_returns)
        random_metrics.append(evaluate_strategy(env, None, "random"))
    bnh_env = HistoricalTradingEnv(test_returns)
    bnh_metrics = evaluate_strategy(bnh_env, None, "buy_and_hold")

    # ── Assemble table ───────────────────────────────────────────────────
    def summarize(metrics_list, label):
        pnl_m, pnl_s = mean_std(metrics_list, "final_pnl")
        sh_m, sh_s = mean_std(metrics_list, "sharpe")
        dd_m, dd_s = mean_std(metrics_list, "max_drawdown")
        print(f"  {label:<28} PnL={pnl_m:+7.2f}±{pnl_s:5.2f}  "
              f"Sharpe={sh_m:+6.2f}±{sh_s:5.2f}  MaxDD={dd_m:6.2f}±{dd_s:5.2f}")
        return {"final_pnl_mean": pnl_m, "final_pnl_std": pnl_s,
                "sharpe_mean": sh_m, "sharpe_std": sh_s,
                "max_drawdown_mean": dd_m, "max_drawdown_std": dd_s}

    print(f"\n[{ticker}] RESULTS TABLE (test set, mean ± std over seeds/repeats):")
    market_results["improved_test"] = summarize(improved_test_metrics, "Improved agent")
    market_results["baseline_test"] = summarize(baseline_test_metrics, "Baseline agent")
    market_results["random_test"] = summarize(random_metrics, "Random agent")
    market_results["buy_and_hold_test"] = summarize([bnh_metrics], "Buy-and-hold")
    market_results["improved_train"] = summarize(improved_train_metrics, "Improved agent (TRAIN, overfit check)")

    # ── 4-curve cumulative P&L plot (test set) ──────────────────────────
    plot_env_improved = ImprovedTradingEnv(test_returns)
    plot_env_baseline = HistoricalTradingEnv(test_returns)
    pnl_improved = average_cumulative_pnl(plot_env_improved, n_episodes=N_EVAL_EPISODES,
                                           model=improved_predictors[0], strategy="trained")
    pnl_baseline = average_cumulative_pnl(plot_env_baseline, n_episodes=N_EVAL_EPISODES,
                                           model=baseline_predictors[0], strategy="trained")
    pnl_random = average_cumulative_pnl(plot_env_baseline, n_episodes=N_EVAL_EPISODES,
                                         model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(plot_env_baseline, n_episodes=N_EVAL_EPISODES,
                                      model=None, strategy="buy_and_hold")
    plot_four_strategies(pnl_improved, pnl_baseline, pnl_random, pnl_bnh,
                          filename=f"plots/{ticker}_four_strategy_pnl.png",
                          title=f"Agent comparison on {ticker} (test set)")

    results[ticker] = market_results


def plot_four_strategies(pnl_improved, pnl_baseline, pnl_random, pnl_bnh, filename, title):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    steps = np.arange(len(pnl_improved))
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, pnl_improved, color="steelblue", linewidth=2, label="Improved agent")
    plt.plot(steps, pnl_baseline, color="purple", linewidth=2, label="Baseline agent")
    plt.plot(steps, pnl_random, color="darkorange", label="Random agent")
    plt.plot(steps, pnl_bnh, color="green", label="Buy and hold")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Cumulative P&L (mean over episodes)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved to {filename}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results = {}
    for ticker in MARKETS:
        returns = np.load(f"data/{ticker}_returns.npy")
        run_market(ticker, returns, results)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/results.json")
    print("\nALL DONE.")
