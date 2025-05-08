import numpy as np
import random
import time
import csv
from ple import PLE
from ple.games.flappybird import FlappyBird

# Initialize Flappy Bird Environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

# Define actions
ACTIONS = env.getActionSet()  # [None, Flap]
n_actions = len(ACTIONS)

# Define discretization
def discretize_state(state):
    vertical_distance = int(state['player_y'] - state['next_pipe_bottom_y']) // 10
    horizontal_distance = int(state['next_pipe_dist_to_player']) // 10
    velocity = int(state['player_vel'])  # Already integer
    return (vertical_distance, horizontal_distance, velocity)

# Q-table
Q = {}

# Hyperparameters
alpha = 0.7          # learning rate
gamma = 0.95         # discount factor
epsilon = 1.0        # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 10000
max_steps = 500
rewards = []

# Training loop
for episode in range(num_episodes):
    env.reset_game()
    state = discretize_state(game.getGameState())
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, n_actions - 1)
        else:
            # Exploitation: Select action with highest Q-value
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action_idx = np.argmax(Q[state])

        action = ACTIONS[action_idx]
        reward = env.act(action)
        total_reward += reward  # Track cumulative reward
        next_state = discretize_state(game.getGameState())

        # Initialize states in Q-table if missing
        if state not in Q:
            Q[state] = np.zeros(n_actions)
        if next_state not in Q:
            Q[next_state] = np.zeros(n_actions)

        # Q-learning update
        current_q = Q[state][action_idx]
        max_next_q = np.max(Q[next_state])
        Q[state][action_idx] += alpha * (reward + gamma * max_next_q - current_q)

        # Transition to next state
        state = next_state

        if env.game_over():
            break

    rewards.append(total_reward)  # Save episode reward
    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")

print("\nTraining completed!")

# Saving Q-table
with open('q_table_flappybird.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['vertical_dist', 'horizontal_dist', 'velocity', 'action_0_value', 'action_1_value'])  # Header
    for state, action_values in Q.items():
        row = list(state) + list(action_values)
        writer.writerow(row)

print("Q-table saved to q_table_flappybird.csv!")