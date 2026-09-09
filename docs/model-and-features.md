# Model and feature engineering

## Why combine CNN and MLP layers?

The input is spatial: walls, food, Pacman, ghosts, and derived local risk information all live on a grid. Convolutions therefore capture local relationships while sharing parameters across positions. Adaptive pooling converts the spatial representation to a fixed-size feature vector, and the MLP maps that representation to the five action classes.

## Derived state channels

The state tensor contains raw spatial information plus engineered signals intended to make imitation easier, such as ghost danger regions, safe regions, normalized score, remaining-food ratio, remaining capsules, ghost count, terminal-state indicators, and visibility/scared-state information.

## Training safeguards

Dropout and batch normalization improve optimization/regularization. A validation split is used to choose the best checkpoint, the learning rate is reduced when validation accuracy plateaus, and training stops after prolonged lack of improvement.
