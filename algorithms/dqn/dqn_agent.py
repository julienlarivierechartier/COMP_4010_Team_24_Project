import torch
from torch.nn import MSELoss
from torch.optim import Adam
import random
import gymnasium as gym
import numpy as np
from pathlib import Path
from .replay_buffer import ReplayBuffer
from .q_network import QNetwork
from ..base import BaseAlgorithm


class DQNAgent(BaseAlgorithm):
    """Implementation of DQN agent that does Q-learning with neural network function
    approximation with experience replay and target network for stable learning on
    continuous state spaces with discrete actions. The target network updates less
    frequently (every target_update_step) by copying the weights from the main network
    which updates at every step. This prevents "chasing a moving target". The replay
    buffer solves the sequential correlation problem by uniformly sampling mini-batches
    (controlled by batch_size) of experiences from recent and older transitions, which
    decorrelates training data and allows reusing experiences multiple times."""

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon:float = 0.995,
        buffer_capacity: float = 5000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = None,
    ):
        """
        Initialize the DQN agent algorithm with the given hyperparameters.

        Parameters:
            env: Gym environment (our CustomSumoEnvironment).
            lr: Gradient descent step size (alpha).
            gamma: Discount factor for future rewards.
            epsilon: Sets when to exploit versus explore.
            buffer_capacity: Maximum size of replay buffer.
            batch_size: Size of the mini-batches when sampling from buffer.
            target_update_freq: Number of main network updates before copying weights
                to the target network
            device: Device to run on (string "cuda" or "cpu")
        """

        # Check if the model can use the GPU
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set the environment and extract its dimensions
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Initialize the learning hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon

        # Initialize the empty replay buffer with the given capacity
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        """Initialize two identical networks (main and target). The random weight 
        initialization is done automatically by pytorch."""
        self.q_network = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.obs_dim, self.action_dim).to(self.device)

        """Copy the weights from the main network to target network because since they 
        were initialized separately in the above lines, their initial weights differ 
        and we need both networks to be initialized with the exact same weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the optimizer for gradient descent and the loss_function as MSE
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.loss_function = MSELoss()

        """Initialize update_counter to check if it is time to copy the weights from 
        the main network to the target network"""
        self.update_counter = 0

    def select_action(self, obs: np.ndarray, training: bool = True):
        """Select action using epsilon-greedy policy. Returns the action index."""

        # Convert observation to torch.Tensor to input in the network forward pass
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        """When training and exploring, choose a random action. Otherwise choose the 
        greedy action."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                """Choose action with maximum Q-value via argmax_a q_hat(s, a; w). When
                getting the argmax, need to convert from 1d tensor to regular python
                int."""
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()

    def train_step(self, transition: tuple):
        """Perform a training step using the transition tuple (state, action, reward,
        next_state, done). Add the transition and sample a mini-batch from the buffer.
        Compute the targets using the target network and perform gradient descent.
        Checks if it is time to update the target network and do the update then."""

        # Add the transition to the replay buffer.
        self.replay_buffer.add(transition)

        # Only train if we have enough samples to form a batch
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a mini-batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Move the batch tensors to the device (GPU)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        """Get the Q-values from the main network as a matrix with shape (batch_size, 
        action_dim)"""
        q_values = self.q_network(states)

        """Create row indices 0 to batch_size - 1 and use it to select the batch from 
        the q-values matrix to get a tensor with shape (batch_size,)"""
        batch_indices = torch.arange(self.batch_size, device=self.device)
        current_q_values = q_values[batch_indices, actions]

        # Compute the targets using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            """Compute the target Q-values (ensure that the terminal state doesn't 
            bootstrap, i.e. gets immediate rewards only.)"""
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss as squared TD error
        loss = self.loss_function(current_q_values, target_q_values)

        """Perform stochastic gradient descent (Adam optimizer) using pytorch to compute 
        gradients automatically and perform backpropagation through the main network."""

        # Clear out the older gradients (because torch accumulates them)
        self.optimizer.zero_grad()

        # Compute the gradients w.r.t. all network parameters.
        loss.backward()

        # Update the weights using the computed gradients
        self.optimizer.step()

        # Increment the target network update counter
        self.update_counter += 1

        """If it is time to update the target network, Copy the weights from the main 
        network to the target network."""
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def reset(self):
        pass

    def save(self, path: Path):
        """Create a checkpoint by saving main and target network, as well as training
        state for future use."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_counter": self.update_counter,
            },
            path,
        )

    def load(self, path: Path):
        """Allow resuming from checkpoint by loading main, target network, and training
        state from saved file."""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.update_counter = checkpoint["update_counter"]
            self.q_network.to(self.device)
            self.target_network.to(self.device)
        else:
            print(f"Cannot load DQN algorithm from {path} because it does not exist.")
