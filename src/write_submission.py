'''write_submission.py
Script to write submission CSV file for Pacman neural network agent.
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave ronic donkeng
version: 1.0
'''
import pickle
import torch
import pandas as pd

from data import state_to_tensor
from architecture import PacmanNetwork


class SubmissionWriter:
    def __init__(self, test_set_path, model_path):
        """
        Initialize the writing of your submission.
        Pay attention that the test set only contains GameState objects,
        it's no longer (GameState, action) pairs.

        Arguments:
            test_set_path: The file path to the pickled test set.
            model_path: The file path to the trained model.
        """
        with open(test_set_path, "rb") as f:
            self.test_set = pickle.load(f)

        sample_input = state_to_tensor(self.test_set[0])
        input_channels = sample_input.shape[0]

        self.model = PacmanNetwork(
            input_channels=input_channels,
            conv_hidden1=32,
            conv_hidden2=64,
            mlp_hidden1=128,
            mlp_hidden2=64,
            output_size=5)

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict_on_testset(self):
        """
        Generate predictions for the test set.

        !!! Your predicted actions should follow the same order
        as the test set provided.
        """
        actions = []
        action_map = {
            0: 'North',
            1: 'South',
            2: 'East',
            3: 'West',
            4: 'Stop'
        }
        for state in self.test_set:
            x = state_to_tensor(state).unsqueeze(0)
            output = self.model(x)
            predicted_action = torch.argmax(output, dim=1).item()
            actions.append(action_map[predicted_action])

        return actions

    def write_csv(self, actions, file_name="submission"):
        """
        ! Do not modify !

        Write the predicted actions (North, South, ...)
        to a CSV file.

        """
        submission = pd.DataFrame(
            data={
                'ACTION': actions,
            },
            columns=["ACTION"]
        )

        submission.to_csv(file_name + ".csv", index=False)


if __name__ == "__main__":
    writer = SubmissionWriter(
        test_set_path="datasets/pacman_test.pkl",
        model_path="pacman_model.pth"  # change if needed
    )
    predictions = writer.predict_on_testset()
    writer.write_csv(predictions)
