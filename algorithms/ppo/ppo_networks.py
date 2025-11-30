import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def orthogonal_init(layer, gain=np.sqrt(2)):
    # init weights using orthogonal method instead of random
    # helps network start in better state for learning
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ActorCritic(nn.Module):
    # actor-critic network for PPO
    # actor picks actions, critic estimates how good states are
    
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()

        # shared layers extract features from observation
        # layer norm helps with different observation scales
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

        # actor network outputs action logits (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh(), 
            nn.Linear(hidden_size, action_dim)
        )

        # critic network outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh(), 
            nn.Linear(hidden_size, 1)
        )
        
        # init all layers with orthogonal method
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        # smaller gain for policy head so initial actions arent too random
        orthogonal_init(self.actor[-1], gain=0.01)
        # normal gain for value head
        orthogonal_init(self.critic[-1], gain=1.0)

    def forward(self, x):
        # pass observation through network
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def get_action(self, obs):
        # sample action from current policy
        # also get log prob and value for training later
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob, value
    
    def evaluate_actions(self, obs, actions):
        # re-evaluate old actions under current policy
        # needed for PPO update to compute ratio
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
