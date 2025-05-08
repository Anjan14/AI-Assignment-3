import numpy as np
import random
import csv
import time
from ple import PLE
from ple.games.flappybird import FlappyBird

def discretize_state(state):
    vertical_distance = int(state['player_y'] - state['next_pipe_bottom_y']) // 10
    horizontal_distance = int(state['next_pipe_dist_to_player']) // 10
    velocity = int(state['player_vel'])  # Already integer
    return (vertical_distance, horizontal_distance, velocity)


Q = {}

with open('q_table_flappybird.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        state = (int(row['vertical_dist']), int(row['horizontal_dist']), int(row['velocity']))
        action_values = np.array([float(row['action_0_value']), float(row['action_1_value'])])
        Q[state] = action_values

print("Q-table loaded successfully!")


# ---------------------------------------------------------
# Play Flappy Bird using the learned Q-table (Greedy Policy)
# ---------------------------------------------------------

# Initialize Flappy Bird Environment
game = FlappyBird()

env = PLE(game, fps=30, display_screen=True)
env.init()

# Define actions
ACTIONS = env.getActionSet()  # [None, Flap]
n_actions = len(ACTIONS)

env.reset_game()

state = discretize_state(game.getGameState())
done = False

print("\nAgent playing with learned Q-table...\n")

while not env.game_over():
    if state not in Q:
        action_idx = random.randint(0, n_actions - 1)
    else:
        action_idx = np.argmax(Q[state])

    action = ACTIONS[action_idx]
    reward = env.act(action)
    next_state = discretize_state(game.getGameState())
    state = next_state

    time.sleep(0.02)

print("\nEpisode finished!")