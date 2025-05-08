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

num_episodes = 1000
rewards = []


# Training loop
for episode in range(num_episodes):
    #your code

# Plot
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