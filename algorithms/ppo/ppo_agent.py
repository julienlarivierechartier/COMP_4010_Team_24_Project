import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from .ppo_networks import ActorCritic
from .rollout_buffer import RolloutBuffer
from ..base import BaseAlgorithm


class PPO:
    # PPO algorithm for traffic control
    # actor-critic with clipped objective
    # handles vehicles and pedestrians
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        gae_lambda=0.95,
        K=10,
        device=None,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        normalize_advantages=True,
    ):
        # setup PPO hyperparameters
        # lr = learning rate for adam
        # gamma = discount for future rewards
        # clip = how much policy can change
        # gae_lambda = for advantage calc
        # K = epochs to train per episode
        # entropy_coef = exploration bonus
        # value_loss_coef = weight for critic
        # max_grad_norm = clip gradients
        # target_kl = stop if policy changes too much
        # normalize_advantages = scale advantages
        
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.gamma = gamma
        self.clip = clip
        self.K = K
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages

        self.buffer = RolloutBuffer()

        # actor-critic network
        self.policy = ActorCritic(obs_dim, action_dim, hidden_size=256).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        # track how many updates
        self.updates = 0

    def compute_advantages(self, rewards, values, dones):
        # compute advantages using GAE
        # goes backwards through episode calculating advantage estimates
        advantages = []
        gae = 0
        values = values + [0]  # bootstrap for last state

        for i in reversed(range(len(rewards))):
            # TD error
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            # accumulate gae
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        # returns = advantages + baseline
        returns = [adv + values[i] for i, adv in enumerate(advantages)]
        return advantages, returns

    def update(self):
        # main PPO update after collecting full episode
        # runs K epochs of training on the batch
        if self.buffer.size() == 0:
            return  # nothing to learn from

        device = self.device

        # get all data from buffer and move to device
        states = torch.stack(self.buffer.states).to(device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = [v.item() if torch.is_tensor(v) else v for v in self.buffer.values]

        # compute advantages and returns using GAE
        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        old_values = torch.tensor(values[:-1] if len(values) > len(returns) else values, 
                                   dtype=torch.float32, device=device)

        # normalize advantages to mean 0 std 1
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # train for K epochs
        for epoch in range(self.K):
            # get new log probs and values from current policy
            new_logprobs, new_values, entropy = self.policy.evaluate_actions(states, actions)
            new_values = new_values.squeeze()

            # compute ratio of new policy to old policy
            ratio = torch.exp(new_logprobs - old_logprobs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # value loss with clipping
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip, self.clip
            )
            value_losses = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            # entropy bonus for exploration
            entropy_loss = entropy.mean()

            # combine losses
            loss = (
                actor_loss 
                + self.value_loss_coef * critic_loss 
                - self.entropy_coef * entropy_loss
            )

            # backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            
            # clip gradients if they get too big
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()

            # check if policy changed too much, stop early if so
            with torch.no_grad():
                kl = (old_logprobs - new_logprobs).mean().item()
                if kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch+1} due to reaching max KL: {kl:.4f}")
                    break

        self.updates += 1
        self.buffer.clear()


class PPOAgent(BaseAlgorithm):
    # wrapper for PPO to work with the training loop
    # interfaces with SUMO environment
    
    def __init__(
        self, 
        obs_dim,    
        action_dim,     
        lr=3e-4, 
        gamma=0.99, 
        clip=0.2, 
        gae_lambda=0.95, 
        K=10,
        entropy_coef=0.01,
        device=None
    ):
        # init PPO agent with hyperparameters
        
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"PPOAgent initialized on device: {self.device}")

        self.ppo = PPO(
            obs_dim, 
            action_dim, 
            lr, 
            gamma, 
            clip, 
            gae_lambda, 
            K, 
            device=self.device,
            entropy_coef=entropy_coef,
        )
        
        # for logging
        self.episode_rewards = []

    def reset(self):
        # called at beginning of episode
        """JLC: moved the code to the "episode_end" function below to to prevent 
        crashing when in eval mode."""
        pass
            
    def end_episode(self, training:bool):
        """Do something at the end of episode (independent of reset)"""
        if training and self.ppo.buffer.size() > 0:
            self.ppo.update()
        else:
            # Just clear the buffer during evaluation
            self.ppo.buffer.clear()

    def select_action(self, obs:np.ndarray, training:bool=True):
        # pick traffic light phase based on current state
        # stores everything in buffer for training
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # sample action from policy
        action, logprob, value = self.ppo.policy.get_action(obs_tensor)

        # store for later
        self.ppo.buffer.states.append(obs_tensor.cpu())
        self.ppo.buffer.actions.append(action)
        self.ppo.buffer.logprobs.append(logprob.cpu().item())
        self.ppo.buffer.values.append(value.cpu().item())

        return action

    def train_step(self, transition:tuple):
        # called after each step to store reward and done
        state, action, reward, next_state, done = transition
        
        # add to buffer
        self.ppo.buffer.add_reward_and_done(float(reward), float(done))

    def save(self, path: Path | str):
        # save model checkpoint
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.ppo.policy.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'updates': self.ppo.updates,
        }, path)

    def load(self, path: Path | str):
        # load saved checkpoint
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.ppo.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ppo.updates = checkpoint.get('updates', 0)
            self.ppo.policy.to(self.device)
            print(f"Loaded PPO checkpoint from {path} (updates: {self.ppo.updates})")
        else:
            print(f"Cannot load algo at path {path} because it does not exist.")
    
    def set_env(self, env):
        pass

    def end_episode(self, training:bool) -> None:
        if training:
            self.ppo.update()
        else:
            self.ppo.buffer.clear()