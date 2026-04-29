import numpy as np
import random
from GridWorld import GridWorld, Agent, train

# ─────────────────────────────────────────────
#  HELPER: print the best action per cell
# ─────────────────────────────────────────────
def print_policy(agent, world, actions):
    rows, cols = world.world.shape
    print("  Policy (best action per cell):")
    for i in range(rows):
        row_str = "  "
        for j in range(cols):
            cell = world.world[i][j]
            if cell == 'X':
                row_str += " X  "
            elif cell == 'G':
                row_str += " G  "
            else:
                best_action = max(actions, key=lambda a: agent.qtable[((i, j), a)])
                row_str += f" {best_action}  "
        print(row_str)
    print()


# ─────────────────────────────────────────────
#  WORLD DEFINITIONS
# ─────────────────────────────────────────────

worlds = {
    "2x2": (
        np.array([
            ['-', 'X'],
            ['-', 'G']
        ]),
        (0, 0)
    ),
    "3x3": (
        np.array([
            ['-', '-', '-'],
            ['-', 'X', '-'],
            ['-', '-', 'G']
        ]),
        (0, 0)
    ),
    "4x4": (
        np.array([
            ['-', '-', '-', '-'],
            ['-', 'X', 'X', '-'],
            ['-', '-', '-', '-'],
            ['-', 'X', '-', 'G']
        ]),
        (0, 0)
    ),
    "5x5": (
        np.array([
            ['-', '-', '-', '-', '-'],
            ['-', 'X', 'X', '-', '-'],
            ['-', '-', '-', 'X', '-'],
            ['-', 'X', '-', '-', '-'],
            ['-', '-', '-', 'X', 'G']
        ]),
        (0, 0)
    ),
}

rewards = {'-': -1, 'G': 10}
actions = ['U', 'D', 'L', 'R']

# ─────────────────────────────────────────────
#  RUN EACH WORLD
# ─────────────────────────────────────────────

for name, (world_array, start) in worlds.items():
    print(f"{'='*40}")
    print(f"  World: {name}")
    print(f"{'='*40}")
    print("  Grid:")
    for row in world_array:
        print("  ", " ".join(row))
    print()

    grid = GridWorld(world_array, start, rewards, 0.9)
    agent = Agent(grid, actions)
    trained_agent = train(grid, agent, episodes=5000, epsilon=0.3, alpha=0.1, gamma=0.9)

    print_policy(trained_agent, grid, actions)