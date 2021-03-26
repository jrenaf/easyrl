from dataclasses import dataclass

from easyrl.configs.basic_config import BasicConfig
from easyrl.configs.ppo_config import PPOConfig


@dataclass
class GPConfig(PPOConfig):
    use_dynamics_randomization: bool = False
    simulator_name: str = "RAISIM"

gp_cfg = GPConfig()
