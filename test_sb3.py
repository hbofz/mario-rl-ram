import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

# Using RecurrentPPO is tricky if it's not installed, let's use the local environment.
import sys
print("System OK")
