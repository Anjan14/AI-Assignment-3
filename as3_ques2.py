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

# Training loop
for episode in range(num_episodes):
    env.reset_game()
    state = discretize_state(game.getGameState())
    total_reward = 0

    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, n_actions - 1)
        else:
            action_idx = None #your code

        action = ACTIONS[action_idx]
        reward = env.act(action)
        next_state = discretize_state(game.getGameState())

        #your code to update Q table

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