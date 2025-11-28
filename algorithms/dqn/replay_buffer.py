from random import sample
import torch
from collections import deque


class ReplayBuffer:
    """Implementation of the experience replay buffer that uses a double-ended queue to
    store past transitions and sample them uniformly as mini-batches for training. When
    the buffer is full, it discards the oldest transitions."""

    def __init__(self, capacity: int = 500):
        # Initialize the buffer deque
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: tuple) -> None:
        # Add the transition (state, action, reward, next_state, done) to the buffer
        self.buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a mini-batch of transitions from the buffer and returns them
        as a tuple of tensors for neural network training."""

        # Sample a batch (list of transition tuples) from the buffer
        batch = sample(self.buffer, batch_size)

        """Unpack the list into vectors of each component and convert NumPy arrays 
        to tensors for the network. States have shape (batch_size, obs_dim) because 
        they are NumPy arrays that get converted and stacked, while the rest (all 
        scalars) have shape (batch_size,)"""
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states]),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        # Accessor for the length (prevents having to do len(buffer.buffer in DQNAgent).
        return len(self.buffer)
