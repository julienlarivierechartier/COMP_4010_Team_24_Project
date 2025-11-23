import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .ppo_networks import ActorCritic
from .rollout_buffer import RolloutBuffer
from ..base import BaseAlgorithm

class PPO():
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


class PPOAgent(BaseAlgorithm):
    def __init__(self, env, lr=3e-4, gamma=0.99, clip=0.2, gae_lambda=0.95, K=4):
        """
        Wrap the PPO class to conform to BaseAlgorithm interface
        env: Gymnasium environment
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.ppo = PPO(obs_dim, action_dim, lr, gamma, clip, gae_lambda, K)

    def reset(self):
        # PPO doesn't have episode-specific internal state, but we could clear buffer
        self.ppo.buffer.clear()

    def select_action(self, obs):
        # Convert obs to torch tensor if needed
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action, logprob, value = self.ppo.policy.get_action(obs_tensor)
        
        # Store step info in buffer
        self.ppo.buffer.states.append(obs_tensor)
        self.ppo.buffer.actions.append(action)
        self.ppo.buffer.logprobs.append(logprob)
        self.ppo.buffer.values.append(value.squeeze())
        
        return action

    def train_step(self, transition=None):
        """
        For PPO, training happens in batches. We can train once the buffer
        is populated. The transition argument is optional here.
        """
        # Only update if we have enough transitions (or can do every step)
        self.ppo.update()

    def save(self, path: Path | str):
        torch.save(self.ppo.policy.state_dict(), Path(path) / "ppo_model.pt")

    def load(self, path: Path | str):
        self.ppo.policy.load_state_dict(torch.load(Path(path) / "ppo_model.pt"))