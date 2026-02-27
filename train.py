"""
RelCube training loop using Bellman-style value iteration with curriculum learning.

For each batch of states:
  1. Apply all 12 moves to get neighbor states (10k × 12 = 120k)
  2. Forward pass all neighbors in one shot
  3. Label: 0 if solved, else 1 + min(neighbor_values)
  4. Train on the 10k (state, label) pairs
"""

import torch
import torch.nn as nn
import numpy as np
import time
from model import EmbeddingLayer, TransformerBlocks, ValueHead
from utils.cube_encoding import (
    MOVE_INFO, ADJUST, apply_move_to_encoding
)


# All 12 quarter-turn moves
ALL_MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]

SOLVED_PIECES = np.arange(20, dtype=np.int32)
SOLVED_ORIENTS = np.zeros(20, dtype=np.int32)


class RelCubeValueNet(nn.Module):
    """RelCube model that takes (piece_ids, orientations) directly."""

    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.transformer = TransformerBlocks()
        self.value_head = ValueHead()

    def forward(self, piece_ids, orientations):
        embedded = self.embedding(piece_ids, orientations)
        transformed = self.transformer(embedded)
        value = self.value_head(transformed)
        return value.squeeze(-1)


def is_solved(pieces, orientations):
    """Check if a state is the solved state."""
    return np.array_equal(pieces, SOLVED_PIECES) and np.array_equal(orientations, SOLVED_ORIENTS)


# Replace with DataLoader
def generate_scrambles(num_states, depth):
    """Generate states by scrambling from solved to a given depth."""
    pieces_batch = np.zeros((num_states, 20), dtype=np.int32)
    orients_batch = np.zeros((num_states, 20), dtype=np.int32)

    for i in range(num_states):
        p = SOLVED_PIECES.copy()
        o = SOLVED_ORIENTS.copy()
        prev_face = None
        for _ in range(depth):
            # avoid immediately undoing the last move
            while True:
                move = ALL_MOVES[np.random.randint(12)]
                if move[0] != prev_face:
                    break
            prev_face = move[0]
            apply_move_to_encoding(move, p, o)
        pieces_batch[i] = p
        orients_batch[i] = o

    return pieces_batch, orients_batch


def get_all_neighbors(pieces_batch, orients_batch):
    """
    Apply all 12 moves to each state.

    Args:
        pieces_batch: (N, 20)
        orients_batch: (N, 20)

    Returns:
        neighbor_pieces: (N * 12, 20)
        neighbor_orients: (N * 12, 20)
    """
    n = pieces_batch.shape[0]
    all_p = np.zeros((n * 12, 20), dtype=np.int32)
    all_o = np.zeros((n * 12, 20), dtype=np.int32)

    for i in range(n):
        for j, move in enumerate(ALL_MOVES):
            p = pieces_batch[i].copy()
            o = orients_batch[i].copy()
            apply_move_to_encoding(move, p, o)
            idx = i * 12 + j
            all_p[idx] = p
            all_o[idx] = o

    return all_p, all_o


def compute_labels(pieces_batch, orients_batch, neighbor_values):
    """
    Bellman labels: 0 if solved, else 1 + min(neighbor values).

    Args:
        pieces_batch: (N, 20)
        orients_batch: (N, 20)
        neighbor_values: (N, 12) - model predictions for each neighbor

    Returns:
        labels: (N,)
    """
    n = pieces_batch.shape[0]
    labels = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if is_solved(pieces_batch[i], orients_batch[i]):
            labels[i] = 0.0
        else:
            labels[i] = 1.0 + neighbor_values[i].min().item()

    return labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RelCubeValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Curriculum: scramble depths and how many epochs at each depth
    curriculum = [
        (1, 200),
        (2, 200),
        (3, 200),
        (4, 200),
        (5, 200),
        (6, 300),
        (7, 300),
        (8, 400),
        (9, 400),
        (10, 500),
        (12, 500),
        (14, 600),
        (16, 600),
        (18, 700),
        (20, 800),
    ]

    batch_size = 10000
    inference_batch = 60000  # sub-batch for neighbor forward passes

    print(f"\nBatch size: {batch_size}")
    print(f"Neighbors per batch: {batch_size * 12:,}")
    print(f"Curriculum: {len(curriculum)} stages\n")

    global_step = 0
    for depth, num_epochs in curriculum:
        print(f"{'='*60}")
        print(f"Curriculum depth={depth}, epochs={num_epochs}")
        print(f"{'='*60}")

        for epoch in range(num_epochs):
            t0 = time.time()

            # 1. Generate scrambled states at current depth
            pieces, orients = generate_scrambles(batch_size, depth)

            # 2. Get all 12 neighbors for each state
            nbr_p, nbr_o = get_all_neighbors(pieces, orients)

            # 3. Inference on all neighbors (120k states) — no grad
            model.eval()
            with torch.no_grad():
                n_total = nbr_p.shape[0]
                nbr_vals = torch.zeros(n_total)

                for start in range(0, n_total, inference_batch):
                    end = min(start + inference_batch, n_total)
                    p_t = torch.from_numpy(nbr_p[start:end]).long().to(device)
                    o_t = torch.from_numpy(nbr_o[start:end]).long().to(device)
                    nbr_vals[start:end] = model(p_t, o_t).cpu()

                # Reshape to (N, 12)
                nbr_vals = nbr_vals.view(batch_size, 12)

            # 4. Compute Bellman labels
            labels = compute_labels(pieces, orients, nbr_vals)
            labels_t = torch.from_numpy(labels).float().to(device)

            # 5. Training pass on 10k states
            model.train()
            p_t = torch.from_numpy(pieces).long().to(device)
            o_t = torch.from_numpy(orients).long().to(device)
            preds = model(p_t, o_t)

            loss = loss_fn(preds, labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            dt = time.time() - t0

            if epoch % 20 == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    mean_label = labels.mean()
                    mean_pred = preds.mean().item()
                    solved_mask = labels == 0.0
                    n_solved = solved_mask.sum()

                print(
                    f"  epoch {epoch:4d} | loss {loss.item():.4f} | "
                    f"label_mean {mean_label:.2f} | pred_mean {mean_pred:.2f} | "
                    f"solved_in_batch {n_solved} | {dt:.1f}s"
                )

        # Save checkpoint after each curriculum stage
        ckpt_path = f"data/relcube_d{depth}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "depth": depth,
            "global_step": global_step,
        }, ckpt_path)
        print(f"  Saved {ckpt_path}\n")

    print("Training complete.")

if __name__ == "__main__":
    train()
