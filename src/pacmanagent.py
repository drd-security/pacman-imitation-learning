'''pacmanagent.py
Neural Network Pacman Agent action baseed on current game state.
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave ronic donkeng

version: 1.0
'''
from pacman_module.game import Agent

from data import state_to_tensor
import torch



class PacmanAgent(Agent):
    def __init__(self, model):
        """
        Initialize the neural network Pacman agent.

        Arguments:
            model: The trained neural network model.
        """
        super().__init__()

        self.model = model.eval()
        self.action_map = {
            0: 'North',
            1: 'South',
            2: 'East',
            3: 'West',
            4: 'Stop'
        }

    def get_action(self, state):
        """
        Return the action chosen by the neural network given the
        current state.

        Arguments:
            state: a GameState object
        """
        with torch.no_grad():
            x = state_to_tensor(state).unsqueeze(0)
            output = self.model(x)
            action_index = torch.argmax(output, dim=1).item()

        return self.action_map[action_index]
