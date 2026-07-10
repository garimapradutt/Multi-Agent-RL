"""
week3_probe.py
--------------
Probes REINFORCE's failure modes by running four training experiments
with deliberately varied hyperparameters.

Run:
    python week3_probe.py

Prints the final 10-episode average for each hyperparameter configuration.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ── Reuse the Week 2 implementation (copy-pasted here to be self-contained) ──

class PolicyNetwork(nn.Module):
    """Two-layer MLP that outputs a softmax action distribution."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)


def run_episode(env, policy, gamma):
    """Collect one episode; return (total_reward, returns, log_probs)."""
    rewards, log_probs = [], []
    obs, _ = env.reset()
    done = False
    while not done:
        state_t = torch.tensor(obs, dtype=torch.float32)
        probs = policy(state_t)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # Discounted returns (backwards)
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns   = torch.tensor(returns, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    return sum(rewards), returns, log_probs


def train_reinforce(learning_rate, gamma, num_episodes=300,
                    hidden_dim=128, seed=42):
    """
    Run REINFORCE for num_episodes and return the list of episode returns.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1", render_mode=None)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy    = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    all_returns = []

    for episode in range(num_episodes):
        total_reward, returns, log_probs = run_episode(env, policy, gamma)

        # Normalise returns (baseline = mean of this episode)
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE loss: minimise -sum log_pi * G_hat
        loss = -(log_probs * returns_norm).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        all_returns.append(total_reward)

    env.close()
    return all_returns


# ─────────────────────────────────────────────────────────────────────────────
# Four experimental runs (Table from Assignment)
# ─────────────────────────────────────────────────────────────────────────────

experiments = [
    # label,               learning_rate, gamma,  expected_behaviour
    ("Baseline (Week 2)",  3e-3,          0.99,   "Normal learning"),
    ("Run A — lr=0.1",     0.1,           0.99,   "Too large — diverge?"),
    ("Run B — lr=1e-5",    1e-5,          0.99,   "Too small — stuck?"),
    ("Run C — gamma=0.50", 3e-3,          0.50,   "Short-sighted?"),
]

print("=" * 65)
print("WEEK 3 — Probing REINFORCE Failure Modes (300 episodes each)")
print("=" * 65)

results = {}
for label, lr, gamma, expected in experiments:
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"  lr={lr}  gamma={gamma}  expected: {expected}")
    print(f"  Training ...", end="", flush=True)

    ep_returns = train_reinforce(learning_rate=lr, gamma=gamma, num_episodes=300)
    final_avg  = np.mean(ep_returns[-10:])
    results[label] = (lr, gamma, final_avg)

    print(f" done.")
    print(f"  Final 10-ep average: {final_avg:.1f}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Run':<28} {'LR':>8} {'Gamma':>6} {'Avg last 10':>12}")
print("-" * 65)
for label, (lr, gamma, avg) in results.items():
    print(f"{label:<28} {lr:>8.1e} {gamma:>6.2f} {avg:>12.1f}")
print("=" * 65)
