import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Hyperparameters
alpha = 0.8        # learning rate
gamma = 0.95       # discount factor
epsilon = 1.0      # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 50000
max_steps = 100    # per episode
rewards = []


#your code
def get_action():
    pass

for episode in range(num_episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Choose action (epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = get_action(state)

        # execute the action and find next state and reward
        # your code

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plotting rewards
moving_avg = np.convolve(rewards, np.ones((100,))/100, mode='valid')

plt.plot(moving_avg)
plt.title('Average Reward vs Episode (Moving Average over 100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid()
plt.show()


# --- After training ---

# --- Play using the learned Q-table ---
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="human")
state, _ = env.reset()
done = False

print("\nAgent navigating the FrozenLake:\n")
env.render()

while not done:
    action = get_action(state)  # Take best action
    next_state, reward, done, truncated, _ = env.step(action)
    env.render()
    state = next_state

print("\nEpisode finished!")