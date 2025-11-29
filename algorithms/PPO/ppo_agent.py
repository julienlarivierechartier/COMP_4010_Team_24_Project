import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import gymnasium as gym

from .ppo_networks import ActorCritic
from .rollout_buffer import RolloutBuffer
from ..base import BaseAlgorithm


class PPO:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        gae_lambda=0.95,
        K=4,
        device=None,
    ):

        # Added this to compute on GPU ewhen possible
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.gamma = gamma
        self.clip = clip
        self.K = K
        self.gae_lambda = gae_lambda

        self.buffer = RolloutBuffer()

        # Send to device allows executing on the GPU
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]  # bootstrap

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [adv + values[i] for i, adv in enumerate(advantages)]
        return advantages, returns

    def update(self):

        device = self.device

        states = torch.stack(self.buffer.states).to(device)
        actions = torch.tensor(self.buffer.actions, device=device)
        old_logprobs = torch.tensor(self.buffer.logprobs, device=device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

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
    def __init__(
        self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip=0.2, gae_lambda=0.95, K=4, device=None
    ):
        """
        Wrap the PPO class to conform to BaseAlgorithm interface
        """

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"PPOAgent initialized on device {self.device}")

        self.ppo = PPO(
            obs_dim, action_dim, lr, gamma, clip, gae_lambda, K, device=self.device
        )

    def reset(self):
        # PPO doesn't have episode-specific internal state, but we could clear buffer
        self.ppo.buffer.clear()

    def select_action(self, obs:np.ndarray, training:bool=True):
        # Convert obs to torch tensor if needed
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action, logprob, value = self.ppo.policy.get_action(obs_tensor)

        # Store step info in buffer
        self.ppo.buffer.states.append(obs_tensor.cpu())
        self.ppo.buffer.actions.append(action)
        self.ppo.buffer.logprobs.append(logprob.cpu())
        self.ppo.buffer.values.append(value.cpu().squeeze())

        return action

    def train_step(self, transition:tuple=None):
        """
        For PPO, training happens in batches. We can train once the buffer
        is populated. The transition argument is unused here.
        """
        self.ppo.update()

    def save(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.ppo.policy.state_dict(), path)

    def load(self, path: Path | str):
        if Path(path).exists():
            self.ppo.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.ppo.policy.to(self.device)
        else:
            print(f"Cannot load algo at path {path} because it does not exist.")

    def set_env(self, env: gym.Env):
        pass