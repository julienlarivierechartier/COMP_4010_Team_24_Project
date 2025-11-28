from torch import nn, Tensor


class QNetwork(nn.Module):
    """Implementation of Q-Network for DQN as a fully connected neural network with 2
    hidden layers using ReLU activation to take the observation tensor and output the
    predicted Q-values for each possible action."""

    def __init__(self, obs_dim: int, action_dim: int):

        # Initialize the nn.Module
        super().__init__()

        # Define the fully connected network with hidden layers
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass takes the state batch (batch_size, obs_dim) as input and returns
        the Q-values with shape (batch_size, obs_dim)"""
        return self.network(x)
