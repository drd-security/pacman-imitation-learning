'''train.py
Training pipeline for the Pacman neural network agent.
author:
    -leslie lucynda tingue
    -bruno sandele
    -dave ronic donkeng
version: 1.0
'''
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
            val_split: Fraction of data for validation (default 20%)
        """
        super().__init__()

        self.path = path
        full_dataset = PacmanDataset(self.path)

        # Split into train and validation
        validate_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - validate_size
        self.train_dataset, self.validate_dataset = random_split(
            full_dataset, [train_size, validate_size]
        )

        # Calculate input size based on the first sample
        sample_input, _ = full_dataset[0]
        input_size = sample_input.shape[0]
        conv_hidden1 = 32
        conv_hidden2 = 64
        mlp_hidden1 = 128
        mlp_hidden2 = 64
        output_size = 5  # 5 possible actions: North, South, East, West, Stop

        self.model = PacmanNetwork(
            input_channels=input_size,
            conv_hidden1=conv_hidden1,
            conv_hidden2=conv_hidden2,
            mlp_hidden1=mlp_hidden1,
            mlp_hidden2=mlp_hidden2,
            output_size=output_size)

        # mesure how far the predictions are from the true labels
        self.criterion = nn.CrossEntropyLoss()
        # adjust the weights of the network based on the loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # scheduler to adjust learning rate during training
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=8,
            verbose=True
        )

    '''
    epochs: number of times the entire dataset is
    passed forward and backward through the neural network
    batch_size: number of samples processed before
    the model is updated
    learning_rate: step size at each iteration while
    moving toward a minimum of a loss function
    '''

    # Training loop
    def train(self, n_epochs=200, batch_size=64):
        print("Beginning of the training of your network...")
        print(f"Epochs: {n_epochs}, Batch size: {batch_size}")
        print("-" * 50)

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        validate_loader = DataLoader(
            self.validate_dataset, batch_size=batch_size, shuffle=False)

        best_validate_acc = 0.0
        patience = 50
        counter = 0

        for epoch in range(n_epochs):

            # ===== TRAIN =====
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            train_loss = total_loss / len(train_loader)
            train_acc = 100.0 * correct / total

            # ===== VALIDATION =====
            self.model.eval()
            validate_correct = 0
            validate_total = 0

            with torch.no_grad():
                for inputs, targets in validate_loader:
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    validate_total += targets.size(0)
                    validate_correct += (predicted == targets).sum().item()

            validate_acc = 100.0 * validate_correct / validate_total

            print(
                f"Epoch [{epoch + 1:3d}/{n_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {validate_acc:.2f}%"
            )

            # ===== SAVE BEST + STOP IF NON IMPROVEMENT =====
            if validate_acc > best_validate_acc:
                best_validate_acc = validate_acc
                torch.save(self.model.state_dict(), "pacman_model.pth")
                counter = 0
                print(f"Best model saved! (Val Acc: {validate_acc:.2f}%)")
            else:
                counter += 1

            # ===== SCHEDULER =====
            self.scheduler.step(validate_acc)

            if counter >= patience:
                print("no improvement for 50 epochs, stopping training.")
                break

        print("-" * 50)
        print(
            f"Training complete! Best Validation Accuracy: "
            f"{best_validate_acc:.2f}%"
        )
        # Load best model at the end
        self.model.load_state_dict(torch.load("pacman_model.pth"))


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
