from dataclasses import dataclass
import gym
from easyrl.envs.vec_normalize import VecNormalize
from easyrl.utils.gym_util import save_vec_normalized_env
from easyrl.utils.gym_util import load_vec_normalized_env, load_vec_normalized_env_expert

@dataclass
class BaseAgent:
    env: gym.Env

    def get_action(self, ob, sample=True, **kwargs):
        raise NotImplementedError

    def optimize(self, data, **kwargs):
        raise NotImplementedError

    def save_env(self, save_dir):
        if isinstance(self.env, VecNormalize):
            save_vec_normalized_env(self.env, save_dir)

    def load_env(self, save_dir):
        if isinstance(self.env, VecNormalize):
            load_vec_normalized_env(self.env, save_dir)
        elif hasattr(self.env, "_gym_env") and isinstance(self.env._gym_env, VecNormalize):
            load_vec_normalized_env(self.env._gym_env, save_dir)

    def load_env_expert(self, expert_save_dir):
        if isinstance(self.env, VecNormalize):# or (hasattr(self.env, "_gym_env") and isinstance(self.env._gym_env, VecNormalize)):
            load_vec_normalized_env_expert(self.env, expert_save_dir)
        elif hasattr(self.env, "_gym_env") and isinstance(self.env._gym_env, VecNormalize):
            load_vec_normalized_env(self.env._gym_env, save_dir)

