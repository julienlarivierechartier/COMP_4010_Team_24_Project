import numpy as np
import gymnasium as gym
from custom_env import CUSTOM_ENV_ID

# Q-learning agent
class QLearningAgent:
    def __init__(self, state_space, action_space, lr=0.1, gamma=0.99,
                 epsilon=1.0, eps_decay=0.995, eps_min=0.01):
        self.state_space = state_space
        self.action_space = action_space

        # learning params
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # q table
        self.q_table = np.zeros(state_space + [action_space])

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        # basic Q-learning update
        next_best = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][next_best]
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay(self):
        # decrease epsilon each episode
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

def train(agent, env, episodes=1000):
    for ep in range(episodes):
        state, _ = env.reset()
        state = tuple(state)  # the env gives array so convert it

        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state)

            agent.update_q(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                done = True
        agent.decay()

        # Just print reward to see if training improves
        print("Episode:", ep + 1, "Reward:", total_reward)

if __name__ == "__main__":
    env = gym.make(CUSTOM_ENV_ID)
    obs_shape = list(env.observation_space.shape)
    n_actions = env.action_space.n

    agent = QLearningAgent(obs_shape, n_actions)
    train(agent, env)
