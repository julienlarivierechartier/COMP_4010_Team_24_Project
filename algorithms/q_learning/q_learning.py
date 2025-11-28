import numpy as np
import gymnasium as gym
from pathlib import Path
from custom_env import CUSTOM_ENV_ID
from ..base import BaseAlgorithm


# Q-learning agent
class QLearningAgent(BaseAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        eps_decay=0.995,
        eps_min=0.01,
    ):

        # Extract this from the env (like the other algos)
        self.obs_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        # learning params
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # q table (JLC: is this the shape we want?)
        self.q_table = np.zeros(self.obs_space + [self.action_space])

    def update_q(self, state, action, reward, next_state):
        # basic Q-learning update
        next_best = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][next_best]
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay(self):
        # decrease epsilon each episode
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ---------------------------
    # BaseAlgorithm interface
    # ---------------------------

    # **Needed by BaseAlgorithm
    def reset(self):
        """Resets agent internal state. This is called at the start of each episode. In
        the case of Q-Learning, this could reset the decayed epsilon if we wanted, or it
        could just do nothing (pass)."""
        # self.epsilon = 1.0
        pass

    # **Needed by BaseAlgorithm
    def select_action(self, obs, training:bool=True):
        """Selecting an action based on the observation tuple"""
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[obs])

    # **Needed by BaseAlgorithm
    def train_step(self, transition: tuple):
        """
        transition = (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition
        # Convert to tuples because Q-table indexing expects tuples
        self.update_q(state, action, reward, next_state)
        self.decay()

    # **Needed by BaseAlgorithm
    def save(self, path: Path | str):
        """Save Q-table and parameters."""
        np.savez(
            Path(path),
            q_table=self.q_table,
            epsilon=self.epsilon,
            lr=self.lr,
            gamma=self.gamma,
        )

    # **Needed by BaseAlgorithm
    def load(self, path: Path | str):
        """Load Q-table and parameters."""
        data = np.load(Path(path))
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"])
        self.lr = float(data["lr"])
        self.gamma = float(data["gamma"])


# ------------------------------------
# Testing the training independently
# ------------------------------------


def train(agent: BaseAlgorithm, env: gym.Env, episodes=1000):
    """Rewrote to use the BaseAlgorithm implementation."""
    for ep in range(episodes):
        obs, _ = env.reset()
        agent.reset()

        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            agent.train_step((obs, action, reward, next_obs, terminated or truncated))

            obs = next_obs
            total_reward += reward

            done = terminated or truncated

        print(f"Episode: {ep+1} Reward: {total_reward}")


if __name__ == "__main__":
    env = gym.make(CUSTOM_ENV_ID)
    obs_shape = list(env.observation_space.shape)
    n_actions = env.action_space.n

    agent = QLearningAgent(obs_shape, n_actions)
    train(agent, env)
