# Pacman Imitation Learning

A PyTorch imitation-learning agent that learns to reproduce expert Pacman actions from game-state/action demonstrations using engineered spatial features and a hybrid **CNN + MLP** classifier.

## Highlights

- Converts rich Pacman game states into a 22-channel spatial tensor.
- Uses convolutional layers to learn local/spatial patterns.
- Uses adaptive average pooling so the network can process varying map sizes.
- Uses an MLP head to classify five actions: North, South, East, West, Stop.
- Includes training/validation split, cross-entropy loss, Adam optimization, learning-rate scheduling, checkpointing, and early stopping.
- Includes a trained model checkpoint from the academic submission.

## Model architecture

```text
22-channel state tensor
        |
Conv2D + BatchNorm + ReLU + Dropout
        |
Conv2D + BatchNorm + ReLU + Dropout + MaxPool
        |
Conv2D + BatchNorm + ReLU + Dropout + MaxPool
        |
AdaptiveAvgPool2D(1 x 1)
        |
Linear(64 -> 128) + ReLU + Dropout
        |
Linear(128 -> 64) + ReLU + Dropout
        |
Linear(64 -> 5 actions)
```

## Feature engineering

`data.py` derives spatial and global channels from the game state, including walls, food, capsules, Pacman position, ghost-related information, score/remaining-object ratios, terminal-state indicators, danger regions, and safe regions.

## Training strategy

- 80/20 training-validation split.
- Cross-entropy classification objective.
- Adam optimizer.
- `ReduceLROnPlateau` scheduler driven by validation accuracy.
- Best-checkpoint saving.
- Early stopping after extended validation stagnation.
- Fixed random seeds for reproducibility where practical.

## Repository structure

```text
src/
  architecture.py
  data.py
  train.py
  pacmanagent.py
  run.py
  write_submission.py
models/
  pacman_model.pth
requirements.txt
```

## Running

This project depends on the university Pacman framework and the original pickled demonstration dataset, neither of which is redistributed here.

With authorized access to those dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/run.py
```

Paths may need to be adjusted so the external `pacman_module` and dataset are visible to the scripts.

## Academic context and contribution

Three-person AI project. I completed the majority of the implementation and integration work; both teammates also contributed meaningfully. Existing source headers retain team attribution.

## Portfolio cleanup

The grading `submission.csv` and submission metadata were removed. An unnecessary import-time load of the training dataset in `pacmanagent.py` was also removed because it was not used for inference and made the agent harder to reuse.

## Publication status

See [NOTICE.md](NOTICE.md).
