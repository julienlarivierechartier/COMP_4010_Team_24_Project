import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.ppo_networks import ActorCritic
from algorithms.rollout_buffer import RolloutBuffer


class PPO:
    def __init__(self, obs_dim, action_dim,
                 lr=3e-4, gamma=0.99, clip=0.2,
                 gae_lambda=0.95, K=4):

        self.gamma = gamma
        self.clip = clip
        self.K = K
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]  # bootstrap

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [adv + values[i] for i, adv in enumerate(advantages)]
        return advantages, returns

    def update(self):
        states = torch.stack(self.buffer.states)
        actions = torch.tensor(self.buffer.actions)
        old_logprobs = torch.tensor(self.buffer.logprobs)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(self.K):
            logits, new_values = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()
