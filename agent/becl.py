import collections
import math
from collections import OrderedDict

import hydra
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class BECL(nn.Module):
    def __init__(self, tau_dim, feature_dim, hidden_dim):
        super().__init__()
        
        self.embed = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, feature_dim))

        self.project_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, feature_dim))
        self.apply(utils.weight_init)

    def forward(self, tau):
        features = self.embed(tau)
        features = self.project_head(features)
        return features


class BECLAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, 
                 update_encoder, contrastive_update_rate, temperature, skill, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.update_encoder = update_encoder
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature
        # specify skill in fine-tuning stage if needed
        self.skill = int(skill) if skill >= 0 else np.random.choice(self.skill_dim)
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.batch_size = kwargs['batch_size']
        # create actor and critic
        super().__init__(**kwargs)

        # net
        self.becl = BECL(self.obs_dim - self.skill_dim,
                                           self.skill_dim,
                                           kwargs['hidden_dim']).to(kwargs['device'])

        # optimizers
        self.becl_opt = torch.optim.Adam(self.becl.parameters(), lr=self.lr)

        self.becl.train()


    def get_meta_specs(self):
        return specs.Array((self.skill_dim,), np.float32, 'skill'),

    def init_meta(self):
        skill = np.zeros(self.skill_dim).astype(np.float32)
        if not self.reward_free:
            skill[self.skill] = 1.0
        else:
            skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_contrastive(self, state, skills):
        metrics = dict()
        features = self.becl(state)
        logits = self.compute_info_nce_loss(features, skills)
        loss = logits.mean()

        self.becl_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.becl_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['contrastive_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, skills, state, metrics):

        # compute contrastive reward
        features = self.becl(state)
        contrastive_reward = torch.exp(-self.compute_info_nce_loss(features, skills))

        intr_reward = contrastive_reward
        if self.use_tb or self.use_wandb:
            metrics['contrastive_reward'] = contrastive_reward.mean().item()

        return intr_reward

    def compute_info_nce_loss(self, features, skills):
        # features: (b,c), skills :(b, skill_dim)
        # label positives samples
        labels = torch.argmax(skills, dim=-1) #(b, 1)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long() #(b,b)
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1) #(b,c)
        similarity_matrix = torch.matmul(features, features.T) #(b,b)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)    #(b,b-1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) #(b,b-1)

        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None]
        similarity_matrix = torch.exp(similarity_matrix)

        pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        
        positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True) #(b,1)
        negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)  #(b,1)
        eps = torch.as_tensor(1e-6)
        loss = -torch.log(positives / (negatives + eps) + eps) #(b,1)

        return loss


    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if self.reward_free:

            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

            metrics.update(self.update_contrastive(next_obs, skill))

            for _ in range(self.contrastive_update_rate - 1):
                batch = next(replay_iter)
                obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
                obs = self.aug_and_encode(obs)
                next_obs = self.aug_and_encode(next_obs)

                metrics.update(self.update_contrastive(next_obs, skill))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, metrics)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()

            reward = intr_reward
        else:
            batch = next(replay_iter)

            obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
                batch, self.device)
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
