'''data.py
This module contains the PacmanDataset class for loading and transforming
the pickled dataset into a format suitable for training the architecture.
which includes the state_to_tensor function for feature engineering
from the GameState object.
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave ronic donkeng

version: 1.0

'''
import pickle

import torch
from torch.utils.data import Dataset
from pacman_module.util import manhattanDistance
from pacman_module.game import Directions


def state_to_tensor(state):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    # 1 Extract information about dimensions ot the walls , food and capsules
    walls = state.getWalls()
    width = walls.width
    height = walls.height
    food = state.getFood()
    capsules = state.getCapsules()

    score = state.getScore()
    num_food = state.getNumFood()
    num_agents = state.getNumAgents()
    num_ghosts = num_agents - 1  # Pacman is agent 0
    num_capsules = len(capsules)

    ''' Feature channels (22 total):
        === SPATIAL FEATURES ===
        0: walls
        1: food positions
        2: capsule positions
        3: pacman position
        4: scared ghosts (with timer intensity)
        5: normal ghosts
        6: distance to nearest ghost (normalized heatmap)
        7: adjacent cells to pacman
        8: distance to nearest food (heatmap)
        9: legal actions mask

        === DIRECTION FEATURES ===
        10: pacman direction (encoded as grid)
        11: ghost directions (encoded as vectors on grid)

        === SCALAR FEATURES ===
        12: score (normalized)
        13: food remaining ratio
        14: num ghosts (normalized)
        15: num capsules (normalized)
        16: is winning state
        17: is losing state
        18: has scared ghost
        19: ghost visibility

        === DANGER FEATURES ===
        20: ghost danger zones (within 3 cells)
        21: safe zones (far from ghosts)
    '''
    feature_size = 22

    # create a feature vector
    tensor = torch.zeros((feature_size, height, width), dtype=torch.float32)

    # 2 fill walls(0),food(1) and capsules(2) positions
    for i in range(width):
        for j in range(height):
            if walls[i][j]:
                tensor[0, j, i] = 1.0  # wall position (note: j,i)
            if food[i][j]:
                tensor[1, j, i] = 1.0  # food position
            if (i, j) in capsules:
                tensor[2, j, i] = 1.0  # capsule position

    # 3 Pacman position
    px, py = state.getPacmanPosition()
    px, py = int(px), int(py)
    tensor[3, py, px] = 1.0  # pacman position(convert to int in case of float)

    # 4 Ghost position and distance
    ghost_positions = []
    scared_ghost_positions = []

    for ghost in state.getGhostStates():
        gx, gy = ghost.getPosition()
        g_x, g_y = int(gx), int(gy)

        # check if positions are valid
        if 0 <= g_x < width and 0 <= g_y < height:
            if ghost.scaredTimer > 0:
                tensor[4, g_y, g_x] = ghost.scaredTimer / \
                    40  # scared ghost position (normalized)
                scared_ghost_positions.append((g_x, g_y))
            else:
                tensor[5, g_y, g_x] = 1.0   # normal ghost position
                ghost_positions.append((g_x, g_y))

    # 5 Distance to nearest ghost (normalized)
    ghost = state.getGhostPositions()
    if ghost:
        dists = [
            manhattanDistance(
                (px, py), (int(gx), int(gy))) for (
                gx, gy) in ghost]
        min_dist = min(dists)
        # normalized distance to nearest ghost
        tensor[6, :, :] = (min_dist / (width + height))

    # 6 Adjacent cells to pacman (indicates where pacman can go)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = px + dx, py + dy
        if 0 <= nx < width and 0 <= ny < height:
            tensor[7, ny, nx] = 1.0  # marks adjacent cells to pacman

    # 7 Distance to nearest food (guide toxwards food)
    food_positions = [(x, y) for x in range(width)
                      for y in range(height) if food[x][y]]
    if food_positions:
        for i in range(width):
            for j in range(height):
                if not walls[i][j]:  # only consider non-wall cells
                    # distance to nearest food
                    min_food_dist = min([manhattanDistance((i, j), food_pos)
                                        for food_pos in food_positions])
                    # create a heatmap where closer food results in higher
                    # values
                    tensor[8, j, i] = 1.0 / (min_food_dist + 1.0)

    # 8 Legal actions mask (indicates which moves are possible)
    legal_actions = state.getLegalActions(0)
    directions = {'North': (0, 1), 'South': (
        0, -1), 'East': (1, 0), 'West': (-1, 0), 'Stop': (0, 0)}
    for action in legal_actions:
        if action in directions:
            dx, dy = directions[action]
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                tensor[9, ny, nx] = 1.0  # marks legal action positions

    # =====================================================
    # === DIRECTION FEATURES ===
    # =====================================================

    # 10: Pacman direction (mark direction on grid around pacman)
    pacman_state = state.getPacmanState()
    pacman_dir = pacman_state.configuration.getDirection()
    direction_vectors = {
        'North': (0, 1),
        'South': (0, -1),
        'East': (1, 0),
        'West': (-1, 0),
        'Stop': (0, 0)
    }
    if pacman_dir in direction_vectors:
        dx, dy = direction_vectors[pacman_dir]
        dir_x, dir_y = px + dx, py + dy
        if 0 <= dir_x < width and 0 <= dir_y < height:
            tensor[10, dir_y, dir_x] = 1.0
    # direction at pacman position with value based on direction
    dir_encoding = {
        'North': 0.25,
        'South': 0.5,
        'East': 0.75,
        'West': 1.0,
        'Stop': 0.0}
    tensor[10, py, px] = dir_encoding.get(pacman_dir, 0.0)

    # 11: Ghost directions (mark direction vectors on grid)
    ghost_states = state.getGhostStates()
    has_scared_ghost = False
    all_ghosts_visible = True

    for ghost in ghost_states:
        gx, gy = ghost.getPosition()
        g_x, g_y = int(gx), int(gy)

        if 0 <= g_x < width and 0 <= g_y < height:
            # Get ghost direction
            ghost_dir = ghost.configuration.getDirection()
            if ghost_dir in direction_vectors:
                dx, dy = direction_vectors[ghost_dir]
                # Mark where ghost is heading
                next_x, next_y = g_x + dx, g_y + dy
                if 0 <= next_x < width and 0 <= next_y < height:
                    tensor[11, next_y, next_x] = 1.0
            # Encode direction at ghost position
            tensor[11, g_y, g_x] = dir_encoding.get(ghost_dir, 0.0)

            # Check scared status
            if ghost.scaredTimer > 0:
                has_scared_ghost = True

            # Check visibility
            if hasattr(ghost.configuration, 'visible'):
                if not ghost.configuration.visible:
                    all_ghosts_visible = False

    # =====================================================
    # === SCALAR FEATURES (uniform across grid) ===
    # =====================================================

    # 12: Score (normalized to [-1, 1])
    normalized_score = max(-1.0, min(1.0, score / 1000.0))
    tensor[12, :, :] = normalized_score

    # 13: Food remaining ratio
    total_cells = width * height
    max_food_estimate = total_cells * 0.3
    food_ratio = num_food / max(max_food_estimate, 1.0)
    tensor[13, :, :] = min(1.0, food_ratio)

    # 14: Number of ghosts (normalized, typically 1-4)
    tensor[14, :, :] = num_ghosts / 4.0

    # 15: Capsules remaining (normalized, typically 0-4)
    tensor[15, :, :] = num_capsules / 4.0

    # 16: Is winning state
    tensor[16, :, :] = 1.0 if state.isWin() else 0.0

    # 17: Is losing state
    tensor[17, :, :] = 1.0 if state.isLose() else 0.0

    # 18: Has scared ghost (any ghost is scared)
    tensor[18, :, :] = 1.0 if has_scared_ghost else 0.0

    # 19: Ghost visibility (all ghosts visible = 1, some hidden = 0)
    tensor[19, :, :] = 1.0 if all_ghosts_visible else 0.0

    # 20: Ghost danger zones (within 3 cells)
    danger_radius = 3
    for (gx, gy) in ghost_positions:
        for i in range(width):
            for j in range(height):
                if not walls[i][j]:
                    dist = manhattanDistance((i, j), (gx, gy))
                    if dist <= danger_radius:
                        intensity = 1.0 - (dist / danger_radius)
                        tensor[20, j, i] = max(tensor[20, j, i], intensity)

    # 21: Safe zones (far from ghosts)
    safe_distance = 5
    if ghost_positions:
        for i in range(width):
            for j in range(height):
                if not walls[i][j]:
                    min_ghost_dist = min([manhattanDistance((i, j), (gx, gy))
                                          for (gx, gy) in ghost_positions])
                    if min_ghost_dist >= safe_distance:
                        tensor[21, j, i] = min(
                            min_ghost_dist / (width + height), 1.0)
    else:
        tensor[21, :, :] = 1.0

    return tensor


class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]

        # predict action as an integer label
        action = self.actions[idx]

        action_str = str(action)

        if action_str in ['North', 'South', 'East', 'West', 'Stop']:
            action_map = {
                'North': 0,
                'South': 1,
                'East': 2,
                'West': 3,
                'Stop': 4}
            label = action_map[action_str]
        else:
            label = 4  # default to 'Stop' for unknown

        y = torch.tensor(label, dtype=torch.long)
        return x, y
