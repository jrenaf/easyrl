from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.utils.common import linear_decay_percent
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import load_ckpt_data
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np


class PPOAgentHybrid(PPOAgent):
    def __init__(self, env, actor, critic, same_body=False, dim_cont=4):
        super().__init__(env, actor, critic, same_body)
        self.dim_cont = dim_cont
        
    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        self.eval_mode()
        t_ob = {key: torch_float(ob[key], device=cfg.alg.device) for key in ob}
        act_dist_cont, act_dist_disc, val = self.get_act_val(t_ob)
        action_cont = action_from_dist(act_dist_cont,
                                  sample=sample)
        action_discrete = action_from_dist(act_dist_disc,
                                  sample=sample)
        #print('456', action_discrete.shape, act_dist_disc)
        #print('123', action_cont.shape, act_dist_cont)
        log_prob_disc = action_log_prob(action_discrete, act_dist_disc)
        log_prob_cont = action_log_prob(action_cont, act_dist_cont)
        entropy_disc = action_entropy(act_dist_disc, log_prob_disc)
        entropy_cont = action_entropy(act_dist_cont, log_prob_cont)
        #print("cont:", torch_to_np(log_prob_cont).reshape(-1, 1))
        log_prob = log_prob_cont + torch.sum(log_prob_disc, axis=1)
        #print(log_prob_cont.shape, log_prob_disc.shape)
        entropy = entropy_cont + torch.sum(entropy_disc, axis=1)

        action_info = dict(
                log_prob=torch_to_np(log_prob),
                entropy=torch_to_np(entropy),
            val=torch_to_np(val)
        )
        #print("cd", action_cont.shape, action_discrete.shape)
        action = np.concatenate((torch_to_np(action_cont), torch_to_np(action_discrete)), axis=1)
        #print("action:", action)

        return action, action_info

    def get_act_val(self, ob, *args, **kwargs):
        if type(ob) is dict:
            ob = {key: torch_float(ob[key], device=cfg.alg.device) for key in ob}
        else:
            ob = torch_float(ob, device=cfg.alg.device)
        act_dist_cont, act_dist_disc, body_out = self.actor(ob)
        if self.same_body:
            val, body_out = self.critic(body_x=body_out)
        else:
            val, body_out = self.critic(x=ob)
        val = val.squeeze(-1)
        return act_dist_cont, act_dist_disc, val


    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=cfg.alg.device)
        ob = data['ob']
        state = data['state']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        act_dist_cont, act_dist_disc, val = self.get_act_val({"ob": ob, "state": state})
        action_cont = action[:, :self.dim_cont]
        action_discrete = action[:, self.dim_cont:]
        log_prob_disc = action_log_prob(action_discrete, act_dist_disc)
        log_prob_cont = action_log_prob(action_cont, act_dist_cont)
        entropy_disc = action_entropy(act_dist_disc, log_prob_disc)
        entropy_cont = action_entropy(act_dist_cont, log_prob_cont)
        #print("cont:", torch_to_np(log_prob_cont).reshape(-1, 1))
        log_prob = log_prob_cont + torch.sum(log_prob_disc, axis=1)
        entropy = entropy_cont + torch.sum(entropy_disc, axis=1)

        if not all([x.ndim == 1 for x in [val, entropy, log_prob]]):
            raise ValueError('val, entropy, log_prob should be 1-dim!')
        processed_data = dict(
            val=val,
            old_val=old_val,
            ret=ret,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            adv=adv,
            entropy=entropy
        )
        return processed_data