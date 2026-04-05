# RelCube

Transformer-based heuristic learning for solving the 3x3 Rubik's Cube. The model learns to predict the cost-to-goal (number of moves to solved) from a compact cube state encoding, enabling A* and Beam Search to find solutions.

## Model

The architecture uses a Transformer encoder (8 layers, d_model=256, 8 heads) over 20 piece tokens (8 corners + 12 edges) plus a CLS token. Each piece is embedded using separate lookup tables for slot, piece identity, and orientation. The model outputs both a value (distance to solved) and a policy (next-move probabilities).

## Usage

**Training:**

```bash
python train.py                  # basic training
python train.py --wandb          # with Weights & Biases logging
```

Default hyperparameters: batch size 10,000, buffer size 1,000,000, learning rate 1e-4, max scramble depth 100. Checkpoints are saved every 10 epochs under `checkpoints/`.

**Evaluation:**

```bash
python eval.py solve-rate              # A* solve-rate at various scramble depths
python eval.py astar --num_test 5      # Batched A* on pre-computed test states
python eval.py beam --num_test 5       # Beam search on pre-computed test states
python eval.py heuristic --num_test 100 # Quick heuristic quality check
```

All modes accept `--checkpoint` to specify a model checkpoint (default: `checkpoints/checkpoint_latest.pt`). Run `python eval.py <mode> --help` for additional options.

## Results

The model achieves strong performance on low-to-moderate scramble depths:

| Scramble Depth | Solve Rate | Avg Steps |
|----------------|------------|-----------|
| 3              | 100%       | 3.0       |
| 5              | 100%       | 5.0       |
| 8              | 100%       | 7.8       |
| 10             | 100%       | 10.0      |
| 12             | 100%       | 11.4      |
| 15             | 80%        | 14.2      |
| 18             | 0%         | --        |

The heuristic plateaus around 10-12 predicted moves, which limits guidance for deeper scrambles (18+). Scaling the architecture without additional tuning led to training instability. Future work includes better regularization, curriculum learning, and architecture search to extend the effective range of the learned heuristic.

## Project Structure

```
train.py              # Training loop with buffer generation and checkpointing
model.py              # RelCube Transformer model (encoder, transformer, value/policy heads)
eval.py               # Unified evaluation suite (solve-rate, astar, beam, heuristic)
utils/                # Cube encoding, vectorization, and database utilities
checkpoints/          # Saved model checkpoints
```
