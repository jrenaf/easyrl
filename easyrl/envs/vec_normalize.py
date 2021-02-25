import numpy as np

from easyrl.envs.vec_env import VecEnvWrapper
from easyrl.utils.common import RunningMeanStd

from gym import spaces

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, training=True, ob=True, ret=True, clipob=10.,
                 cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        if isinstance(self.observation_space, spaces.Dict):
            self.ob_rms = RunningMeanStd(shape=self.observation_space['ob'].shape) if ob else None
            self.state_rms = RunningMeanStd(shape=self.observation_space['state'].shape) if ob else None
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None

        self.expert_state_rms = None
        self.expert_ob_rms = None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            for idx, inf in enumerate(infos):
                inf['raw_reward'] = rews[idx]
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        #print("compare in obfilt")
        #print(obs["expert_state"][0:3, 0])
        #print(obs["state"][0:3, 0])


        if self.ob_rms:
            if isinstance(self.observation_space, spaces.Dict):
                if self.training:
                    #print("update rms in vecnorm")
                    self.ob_rms.update(obs['ob'])
                    self.state_rms.update(obs['state'])
                obs_scale = np.clip((obs['ob'] - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                              -self.clipob, self.clipob)
                state_scale = np.clip((obs['state'] - self.state_rms.mean) / np.sqrt(self.state_rms.var + self.epsilon),
                              -self.clipob, self.clipob)
                ob_dict = {'ob': obs_scale, 'state': state_scale}
                if 'expert_ob' in obs: 
                    if self.expert_ob_rms is not None:
                        ob_dict['expert_ob'] = np.clip((obs['expert_ob'] - self.expert_ob_rms.mean) / np.sqrt(self.expert_ob_rms.var + self.epsilon),
                               -self.clipob, self.clipob)
                        #print("apply expert scaling!")
                    else:
                        ob_dict['expert_ob'] = obs['expert_ob']
                if 'expert_state' in obs: 
                    if self.expert_state_rms is not None:
                        ob_dict['expert_state'] = np.clip((obs['expert_state'] - self.expert_state_rms.mean) / np.sqrt(self.expert_state_rms.var + self.epsilon),
                               -self.clipob, self.clipob)
                        #print("apply expert scaling!")
                    else:
                        ob_dict['expert_state'] = obs['expert_state']
                return ob_dict
            else:
                if self.training:
                    self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                              -self.clipob, self.clipob)
                return obs
        else:
            return obs

    def reset(self, cfgs=None):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset(cfgs)
        return self._obfilt(obs)

    def get_states(self):
        data = dict(
            ob_rms=self.ob_rms,
            state_rms=self.state_rms,
            ret_rms=self.ret_rms,
            clipob=self.clipob,
            cliprew=self.cliprew,
            gamma=self.gamma,
            epsilon=self.epsilon
        )
        return data

    def set_states(self, data):
        assert isinstance(data, dict)
        keys = ['ob_rms', 'state_rms', 'ret_rms', 'clipob',
                'cliprew', 'gamma', 'epsilon']
        for key in keys:
            if key in data:
                setattr(self, key, data[key])
            else:
                print(f'Warning: {key} does not exist in data.')

        import copy
        #setattr(self, 'expert_state_rms', copy.deepcopy(data['state_rms']))
        #etattr(self, 'expert_ob_rms', copy.deepcopy(data['ob_rms']))

    def set_states_bc(self, data):
        assert isinstance(data, dict)
        keys = ['ret_rms', 'clipob',
                'cliprew', 'gamma', 'epsilon']
        for key in keys:
            if key in data:
                setattr(self, key, data[key])
            else:
                print(f'Warning: {key} does not exist in data.')

        import copy
        setattr(self, 'expert_state_rms', copy.deepcopy(data['state_rms']))
        setattr(self, 'expert_ob_rms', copy.deepcopy(data['ob_rms']))
