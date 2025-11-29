import numpy as np
import gymnasium as gym
from pathlib import Path
from custom_env import CUSTOM_ENV_ID
from ..base import BaseAlgorithm


class QLearningAgent(BaseAlgorithm):
    """
    Tabular Q-learning with discretization: bin continuous obs to ints, then use a Q-table.
    """
    def __init__(
        self,
        obs_space,
        action_space,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        eps_decay=0.995,
        eps_min=0.01,
        bins: int | list[int] = 10,
    ):
        self.obs_space = obs_space
        self.action_space = action_space

        # learning hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # binning config: allow single int or per-feature list
        if isinstance(bins, int):
            self.bins_per_feature = np.array([bins] * self.obs_dim, dtype=int)
        else:
            if len(bins) != self.obs_dim:
                raise ValueError("bins length must match obs dim")
            self.bins_per_feature = np.array(bins, dtype=int)

        # observation bounds for linear binning
        self.obs_low = np.array(env.observation_space.low, dtype=np.float32)
        self.obs_high = np.array(env.observation_space.high, dtype=np.float32)
        self.obs_range = np.where(
            (self.obs_high - self.obs_low) == 0,
            1e-8,
            self.obs_high - self.obs_low,
        )
        # sparse Q-table: only store visited states
        self.q_table: dict[tuple, np.ndarray] = {}


    # helpers
    def _discretize(self, obs: np.ndarray) -> tuple:
        obs = np.asarray(obs, dtype=np.float32)
        # normalize to [0,1], then cut into bins
        norm = (obs - self.obs_low) / self.obs_range
        norm = np.clip(norm, 0.0, 0.999999)
        idx = np.floor(norm * self.bins_per_feature).astype(int)
        # clamp to avoid overflow
        idx = np.clip(idx, 0, self.bins_per_feature - 1)
        return tuple(idx.tolist())

    def _get_q_values(self, state: tuple) -> np.ndarray:
        # return Q(s,·); init zeros if unseen
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space, dtype=np.float32)
        return self.q_table[state]

    def update_q(self, state, action, reward, next_state, done: bool):
        # standard Q-learning target, and no bootstrap if done
        if done:
            target = reward
        else:
            next_q = self._get_q_values(next_state)
            next_best = np.argmax(next_q)
            target = reward + self.gamma * next_q[next_best]
        current_q = self._get_q_values(state)
        current_q[action] += self.lr * (target - current_q[action])

    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # BaseAlgorithm interface
    def reset(self):
        #per-episode hook; keep decayed epsilon
        pass

    def select_action(self, obs, training: bool = True):
        state = self._discretize(obs)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def train_step(self, transition: tuple):
        #transition = (state, action, reward, next_state, done)
        obs, action, reward, next_obs, done = transition
        s = self._discretize(obs)
        ns = self._discretize(next_obs)
        self.update_q(s, action, reward, ns, done)
        self.decay()

    def save(self, path: Path | str):
        # turn sparse Q-table into a savable list
        table_items = [(state, values) for state, values in self.q_table.items()]
        np.savez(
            Path(path),
            q_table=np.array(table_items, dtype=object),
            epsilon=self.epsilon,
            lr=self.lr,
            gamma=self.gamma,
            eps_decay=self.eps_decay,
            eps_min=self.eps_min,
            bins_per_feature=self.bins_per_feature,
            obs_low=self.obs_low,
            obs_high=self.obs_high,
        )

    def load(self, path: Path | str):
        #load Q-table and hyperparams
        data = np.load(Path(path), allow_pickle=True)
        raw_table = data["q_table"]
        self.q_table = {tuple(state): values for state, values in raw_table}
        self.epsilon = float(data["epsilon"])
        self.lr = float(data["lr"])
        self.gamma = float(data["gamma"])
        self.eps_decay = float(data["eps_decay"])
        self.eps_min = float(data["eps_min"])
        self.bins_per_feature = data["bins_per_feature"]
        self.obs_low = data["obs_low"]
        self.obs_high = data["obs_high"]
        self.obs_range = np.where(
            (self.obs_high - self.obs_low) == 0,
            1e-8,
            self.obs_high - self.obs_low,
        )

    def set_env(env:gym.Env):
        pass

# ------------------------------------
# Testing the training independently
# ------------------------------------


def train(agent: BaseAlgorithm, env: gym.Env, episodes=1000):
    for ep in range(episodes):
        obs, _ = env.reset()
        agent.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            agent.train_step((obs, action, reward, next_obs, terminated or truncated))

            obs = next_obs
            total_reward += reward
            done = terminated or truncated
        print(f"Episode: {ep+1} Reward: {total_reward}")

if __name__ == "__main__":
    env = gym.make(CUSTOM_ENV_ID)
    agent = QLearningAgent(env)
    train(agent, env)
