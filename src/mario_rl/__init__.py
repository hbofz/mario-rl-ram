"""Mario RAM RL package."""

from mario_rl.env import make_mario_env
from mario_rl.levels import LEVEL_CONFIGS, get_level_reward_kwargs

__all__ = [
    "__version__",
    "make_mario_env",
    "LEVEL_CONFIGS",
    "get_level_reward_kwargs",
]

__version__ = "0.1.0"
