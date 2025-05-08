import numpy as np
import gymnasium as gym
import time
import random
import matplotlib.pyplot as plt

# Create environment
env = gym.make('Taxi-v3')

# Hyperparameters
alpha = 0.8  # learning rate
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
max_steps = 200  # per episode

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

num_episodes = 1000
rewards = []


def get_action(state):
    return np.argmax(q_table[state])


# Training loop
for episode in range(num_episodes):
    # your code
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Choose action (epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = get_action(state)

        # Execute action and find the next state, reward
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_rewards += reward

        maxNextActionValue = np.max(q_table[observation])
        q_table[state][action] = q_table[state][action] + alpha * (
                    reward + (gamma * maxNextActionValue) - q_table[state][action])

        state = observation

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards.append(total_rewards)

# Plot
moving_avg = np.convolve(rewards, np.ones((100,)) / 100, mode='valid')
plt.plot(moving_avg)
plt.title('Taxi-v3: Moving Average Rewards')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid()
plt.show()

# Play using trained Q-table
env = gym.make('Taxi-v3', render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    time.sleep(0.5)
env.close()