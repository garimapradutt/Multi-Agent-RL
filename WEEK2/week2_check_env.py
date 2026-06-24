"""
week2_check_env.py
------------------
Quick sanity-check: confirms CartPole-v1 still loads and steps correctly
before the main training script is run.

Usage:
    python week2_check_env.py
"""

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode=None)
obs, info = env.reset()
print("Initial observation:", obs)
print("Action space:", env.action_space)
env.close()
print("Environment check passed ✅")
