import numpy as np
import gymnasium as gym
import time
import random
import matplotlib.pyplot as plt

# Create environment
env = gym.make('Taxi-v3')

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))  # Q-table initialization

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000
rewards = []

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False

    while not done and not truncated:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(Q[state])       # Exploit: best action from Q-table

        # Execute action and observe next state/reward
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        # Update Q-table using Q-learning formula
        max_next_q = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * max_next_q - Q[state, action])

        state = next_state  # Transition to next state

    rewards.append(total_reward)
    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plot moving average of rewards
moving_avg = np.convolve(rewards, np.ones((100,))/100, mode='valid')
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
    action = np.argmax(Q[state])
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    time.sleep(0.5)
env.close()