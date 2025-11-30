import torch
import numpy as np


class RolloutBuffer:
    # stores all the trajectory data for one episode
    # PPO needs full episode before updating
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # computed later during training
        self.advantages = []
        self.returns = []

    def clear(self):
        # reset buffer after update
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def size(self):
        # how many timesteps we have stored
        return len(self.states)
    
    def add_reward_and_done(self, reward, done):
        # store reward and done flag after each step
        self.rewards.append(reward)
        self.dones.append(done)
