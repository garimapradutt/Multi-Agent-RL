"""
week2_reinforce.py
------------------
Implements the REINFORCE policy-gradient algorithm on CartPole-v1.

REINFORCE in one sentence:
  Sample full episodes from the current stochastic policy, compute the
  discounted return at each timestep, then push the policy parameters
  in the direction that increases the log-probability of actions that
  led to high returns.

Usage:
    python week2_reinforce.py

Outputs:
    - Training progress printed to stdout every 10 episodes
    - plots/cartpole_rewards.png  (learning curve)
    - reinforce_returns.npy       (raw episode returns, needed by Week 3)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    A small fully-connected neural network that acts as the stochastic policy.

    Input  : state vector (4 floats for CartPole)
    Output : probability distribution over actions (softmax over 2 logits)

    Using a stochastic (probabilistic) policy is important for REINFORCE because:
      1. It allows exploration — the agent doesn't always pick the same action.
      2. The gradient of the log-probability (log pi) is well-defined and used
         directly in the REINFORCE update rule.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state  : tensor of shape (state_dim,) or (batch, state_dim)
        returns: action PROBABILITIES of the same leading shape

        Softmax converts raw logits to a valid probability distribution
        (all values in [0, 1], sum to 1), which we can sample from.
        """
        logits = self.net(state)
        probs  = torch.softmax(logits, dim=-1)
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# Episode Collection
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    gamma: float = 0.99,
) -> tuple:
    """
    Run ONE complete episode using the current policy.

    Returns
    -------
    total_reward : float  — undiscounted sum of rewards (for logging)
    returns      : torch.Tensor shape (T,)  — discounted cumulative returns G_t
    log_probs    : torch.Tensor shape (T,)  — log pi(a_t | s_t) WITH gradients

    How actions are sampled from the policy
    ----------------------------------------
    At every step:
      1. Convert the numpy observation to a float tensor.
      2. Forward pass through the policy to get action probabilities.
      3. Build a Categorical distribution and SAMPLE one action.
      4. Record log pi(a_t | s_t) — the log-probability of that action.

    The whole forward pass is done WITH gradient tracking so that autograd
    can back-propagate through log_probs when we call loss.backward().

    How returns (G_t) are computed
    --------------------------------
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{T-t}*r_T

    Computed backwards to avoid an O(T^2) loop:
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
    """
    rewards, log_probs = [], []

    obs, _ = env.reset()
    done   = False

    while not done:
        # Convert observation to tensor — gradient tracking ON for log_prob
        state_t = torch.tensor(obs, dtype=torch.float32)

        # Forward pass: get action probabilities
        probs = policy(state_t)                          # shape (action_dim,)

        # Build Categorical distribution and sample
        dist   = torch.distributions.Categorical(probs=probs)
        action = dist.sample()                           # integer tensor

        # Store log-probability WITH gradient (needed for backprop)
        log_probs.append(dist.log_prob(action))

        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # ── Compute discounted returns backwards ─────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns   = torch.tensor(returns, dtype=torch.float32)  # shape (T,)
    log_probs = torch.stack(log_probs)                       # shape (T,)

    return sum(rewards), returns, log_probs


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_reinforce(
    env_name:     str   = "CartPole-v1",
    hidden_dim:   int   = 128,
    learning_rate: float = 3e-3,
    gamma:        float = 0.99,
    num_episodes: int   = 500,
    seed:         int   = 42,
) -> tuple:
    """
    Train a policy using vanilla REINFORCE for num_episodes episodes.

    Returns
    -------
    all_episode_returns : list of floats
    policy              : trained PolicyNetwork
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env        = gym.make(env_name, render_mode=None)
    state_dim  = env.observation_space.shape[0]   # 4 for CartPole
    action_dim = env.action_space.n               # 2 for CartPole

    policy    = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    all_episode_returns = []

    print(f"Training REINFORCE on {env_name} for {num_episodes} episodes ...")
    print(f"  hidden_dim={hidden_dim}  lr={learning_rate}  gamma={gamma}\n")

    for episode in range(num_episodes):

        # ── Collect one episode ───────────────────────────────────────────────
        total_reward, returns, log_probs = run_episode(env, policy, gamma)

        # ── Normalise returns (variance reduction) ────────────────────────────
        # Subtracting the mean and scaling by std reduces gradient variance
        # without biasing the direction of the update.  This is a simple
        # baseline trick — the normalised returns tell each action "were you
        # better or worse than average THIS episode?"
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── REINFORCE loss ────────────────────────────────────────────────────
        #
        #   L(theta) = -sum_t  log pi(a_t | s_t) * G_t_hat
        #
        # Why the MINUS sign?
        #   Adam minimises L by default (gradient descent).
        #   REINFORCE wants to MAXIMISE the expected return J(theta).
        #   Maximising J == minimising -J.
        #   The policy gradient theorem gives:
        #       grad J ≈ E[ sum_t  grad log pi(a_t|s_t) * G_t ]
        #   So we hand Adam the negative of that: L = -sum log_pi * G_hat
        #   When Adam takes a step to lower L, it is doing gradient ASCENT on J.
        #
        # Intuition:
        #   High G_t  → large negative loss term → Adam increases log pi(a_t|s_t)
        #                → that action becomes more probable in this state
        #   Low G_t   → small or positive term   → that action becomes less probable
        loss = -(log_probs * returns_norm).sum()

        # ── Gradient update ───────────────────────────────────────────────────
        optimizer.zero_grad()   # clear accumulated gradients
        loss.backward()         # backprop through log_probs to policy weights

        # Gradient clipping prevents catastrophically large updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        optimizer.step()        # Adam parameter update

        # ── Logging ───────────────────────────────────────────────────────────
        all_episode_returns.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_last_10 = np.mean(all_episode_returns[-10:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Return: {total_reward:6.1f} | "
                f"Avg last 10: {avg_last_10:6.1f}"
            )

    env.close()
    return all_episode_returns, policy


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_returns(
    returns:  list,
    window:   int = 10,
    filename: str = "plots/cartpole_rewards.png",
) -> None:
    """Save a learning-curve plot of episode returns + moving average."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    arr = np.array(returns)
    x   = np.arange(len(arr))
    sma = np.convolve(arr, np.ones(window) / window, mode="valid") \
          if len(arr) >= window else arr

    plt.figure(figsize=(9, 4))
    plt.plot(x, arr, alpha=0.25, color="steelblue", label="Episode return")
    plt.plot(
        np.arange(len(sma)), sma,
        color="steelblue", linewidth=2,
        label=f"{window}-episode moving avg",
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nPlot saved to {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    returns, trained_policy = train_reinforce(
        env_name="CartPole-v1",
        hidden_dim=128,
        learning_rate=3e-3,
        gamma=0.99,
        num_episodes=500,
        seed=42,
    )

    plot_returns(returns, window=10, filename="plots/cartpole_rewards.png")

    np.save("reinforce_returns.npy", np.array(returns))
    print("Episode returns saved to reinforce_returns.npy")

    final_avg = np.mean(returns[-50:])
    print(f"Final 50-episode average: {final_avg:.1f}")
    print("Training complete ✅")
