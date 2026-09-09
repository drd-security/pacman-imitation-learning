'''run.py
Script to run the Pacman game using the trained neural network agent.
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave ronic donkeng

version: 1.0
'''
import numpy as np
import random
import torch

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path_to_saved_model = "models/pacman_model.pth"

# Feel free to add code here depending on your implementation

# Architecture
model = PacmanNetwork(input_channels=22, conv_hidden1=32, conv_hidden2=64,
                      mlp_hidden1=128, mlp_hidden2=64, output_size=5)
model.load_state_dict(torch.load(path_to_saved_model, map_location="cpu"))
model.eval()

pacman_agent = PacmanAgent(model)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
