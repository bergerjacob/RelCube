import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
import shutil
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from model import RelCube
from utils.cube_encoding import apply_move_to_encoding


ALL_MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]

CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
MILESTONE_INTERVAL = 10

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 10000
BUFFER_SIZE = 1000000
INFERENCE_BATCH_SIZE = 120000
MAX_DEPTH = 40  # Used for buffer generation

SOLVED_PIECES = np.arange(20, dtype=np.int32)
SOLVED_ORIENTS = np.zeros(20, dtype=np.int32)

def pack_raw_state(edges, orients, device):
    raw = np.stack([edges, orients], axis=1)
    return torch.from_numpy(raw).long().to(device)


def is_solved(pieces, orientations):
    p_solved = (pieces == SOLVED_PIECES).all(axis=-1)
    o_solved = (orientations == SOLVED_ORIENTS).all(axis=-1)
    return p_solved & o_solved

# --- Precompute Valid Moves (Lookup Table) ---
# State is defined by: (prev_move_idx, consecutive_count)
VALID_MOVES_LUT = {}
for prev_idx in range(-1, 12):
    counts = [0] if prev_idx == -1 else [1, 2]
    for count in counts:
        prev_f = prev_idx // 2 if prev_idx != -1 else -1
        prev_a = prev_f // 2 if prev_f != -1 else -1
        
        valid = []
        for move_idx in range(12):
            f = move_idx // 2
            a = f // 2
            
            if f == prev_f and move_idx != prev_idx: continue
            if f == prev_f and move_idx == prev_idx and count == 2: continue
            if a == prev_a and f < prev_f: continue
            
            valid.append(move_idx)
            
        # Store as a tuple for fast indexing
        VALID_MOVES_LUT[(prev_idx, count)] = tuple(valid)


def generate_scrambles(num_states, max_depth):
    # Stores the entire trajectory: (num_states, max_depth, 20)
    pieces_batch = np.zeros((num_states, max_depth, 20), dtype=np.int32)
    orients_batch = np.zeros((num_states, max_depth, 20), dtype=np.int32)
    
    import random 

    for i in range(num_states):
        p = SOLVED_PIECES.copy()
        o = SOLVED_ORIENTS.copy()
        
        prev_move_idx = -1
        consecutive_count = 0
        
        for d in range(max_depth):
            valid_moves = VALID_MOVES_LUT[(prev_move_idx, consecutive_count)]
            chosen_idx = random.choice(valid_moves)
            
            if (chosen_idx // 2) == (prev_move_idx // 2):
                consecutive_count += 1
            else:
                consecutive_count = 1
                
            prev_move_idx = chosen_idx
            
            apply_move_to_encoding(ALL_MOVES[chosen_idx], p, o)
            
            # Save the state at this depth
            pieces_batch[i, d] = p
            orients_batch[i, d] = o

    return pieces_batch, orients_batch


def get_all_neighbors(pieces_batch, orients_batch):
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


def compute_labels(pieces_batch, orients_batch, nbr_p, nbr_o, neighbor_values):
    if torch.is_tensor(neighbor_values):
        neighbor_values = neighbor_values.cpu().numpy()

    n = pieces_batch.shape[0]

    solved_nbr_mask = is_solved(nbr_p, nbr_o).reshape(n, 12)
    neighbor_values[solved_nbr_mask] = 0.0

    labels = 1.0 + neighbor_values.min(axis=1)

    solved_parents_mask = is_solved(pieces_batch, orients_batch)
    labels[solved_parents_mask] = 0.0

    return labels.astype(np.float32)


def save_checkpoint(model, target_model, optimizer, epoch, global_step, latest_path, milestone_path=None):
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    temp_path = latest_path + ".tmp"
    checkpoint = {
        "model_state": model.state_dict(),
        "target_model_state": target_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, temp_path)
    shutil.move(temp_path, latest_path)
    if milestone_path:
        shutil.copy(latest_path, milestone_path)


def load_checkpoint(model, target_model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    target_model.load_state_dict(checkpoint["target_model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["global_step"]


# def test_model(model, buffer_pieces, buffer_orients, buffer_depths, test_pkl_path=None):
#     from utils.cube_facelet import apply_move, parse_moves, SOLVED_STATE, get_piece_encoding
#
#     num_test = 1000
#
#     if test_pkl_path and os.path.exists(test_pkl_path):
#         import pickle
#         with open(test_pkl_path, 'rb') as f:
#             test_data = pickle.load(f)
#         test_pieces = np.stack([s[0] for s in test_data['states']], axis=0)
#         test_orients = np.stack([s[1] for s in test_data['states']], axis=0)
#         test_depths = np.array(test_data['num_back_steps'])
#         if len(test_pieces) < num_test:
#             num_test = len(test_pieces)
#         test_pieces = test_pieces[:num_test]
#         test_orients = test_orients[:num_test]
#         test_depths = test_depths[:num_test]
#     else:
#         test_indices = np.random.choice(len(buffer_pieces), num_test, replace=False)
#
#         test_pieces = buffer_pieces[test_indices]
#         test_orients = buffer_orients[test_indices]
#         test_depths = buffer_depths[test_indices]
#
#     state_list = []
#     for i in range(num_test):
#         state = SOLVED_STATE
#         moves = []
#         for j in range(int(test_depths[i])):
#             move = ALL_MOVES[np.random.randint(12)]
#             moves.append(move)
#
#         for m in moves:
#             state = apply_move(state, m)
#
#         pieces, orients = get_piece_encoding(state)
#         state_list.append((pieces, orients))
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#
#     predictions = []
#     batch_size = 100
#     for start in range(0, num_test, batch_size):
#         end = min(start + batch_size, num_test)
#
#         batch_pieces = np.stack([state_list[i][0] for i in range(start, end)], axis=0)
#         batch_orients = np.stack([state_list[i][1] for i in range(start, end)], axis=0)
#         raw = pack_raw_state(batch_pieces, batch_orients, device)
#
#         with torch.no_grad():
#             value, _ = model(raw)
#             predictions.extend(value.squeeze(-1).cpu().numpy())
#
#     predictions = np.array(predictions)
#
#     print("\n--- Test Results ---")
#     print(f"Number of test states: {num_test}")
#     print(f"Min Depth: {test_depths.min()}, Max Depth: {test_depths.max()}")
#     print(f"Predicted Values - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
#     print(f"Label Mean: {test_depths.mean():.2f}")
#
#     depth_diff = np.abs(predictions - test_depths)
#     print(f"MAE: {depth_diff.mean():.2f}, RMSE: {np.sqrt((depth_diff**2).mean()):.2f}")
#     print("--------------------\n")


def train(use_wandb=False):
    if use_wandb:
        if "WANDB_PROJECT" not in os.environ:
            raise ValueError("Please set your project first! Run: export WANDB_PROJECT='relcube'")
            
        import wandb
        wandb.init(
            # id="dk8my6h6",         # Add your specific Run ID here
            # resume="must",         # Tell it to resume
            config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "buffer_size": BUFFER_SIZE,
                "inference_batch_size": INFERENCE_BATCH_SIZE,
                "max_depth": MAX_DEPTH
        })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RelCube().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Buffer size: {BUFFER_SIZE:,}")
    print(f"Neighbors per batch: {BATCH_SIZE * 12:,}\n")

    buffer_pieces = None
    buffer_orients = None
    
    global_step = 0
    epoch = 0

    import copy
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    checkpoint_latest = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
    if os.path.exists(checkpoint_latest):
        print(f"Resuming from checkpoint: {checkpoint_latest}")
        epoch, global_step = load_checkpoint(model, target_model, optimizer, checkpoint_latest)
        print(f"Resumed from epoch {epoch}, step {global_step}")
    else:
        print("Starting fresh training")
    
    while True:
        epoch += 1
        epoch_start = time.time()
        
        print(f"Epoch {epoch}: Generating buffer...")
        gen_start = time.time()
        
        num_puzzles = BUFFER_SIZE // MAX_DEPTH
        all_pieces = []
        all_orients = []
        all_depths = []
        
        for cycle in range(max(1, BUFFER_SIZE // (num_puzzles * MAX_DEPTH))):
            pieces_traj, orients_traj = generate_scrambles(num_puzzles, MAX_DEPTH)
            
            for depth in range(1, MAX_DEPTH + 1):
                # We simply pull the column corresponding to the depth we want
                all_pieces.append(pieces_traj[:, depth - 1, :])
                all_orients.append(orients_traj[:, depth - 1, :])
                all_depths.append(np.full(num_puzzles, depth, dtype=np.int32))
        
        buffer_pieces = np.concatenate(all_pieces, axis=0)
        buffer_orients = np.concatenate(all_orients, axis=0)
        buffer_depths = np.concatenate(all_depths, axis=0)
        
        gen_time = time.time() - gen_start
        print(f"  Buffer generated in {gen_time:.1f}s with {len(buffer_pieces):,} states")
        
        indices = np.random.permutation(len(buffer_pieces))
        num_batches = len(indices) // BATCH_SIZE
        print(f"  {num_batches} batches per epoch\n")
        
        batch_start = time.time()
        
        for batch_idx in range(num_batches):
            t0 = time.time()
            
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_indices = indices[start_idx:end_idx]
            
            pieces = buffer_pieces[batch_indices]
            orients = buffer_orients[batch_indices]
            
            nbr_p, nbr_o = get_all_neighbors(pieces, orients)
            
            model.eval()
            with torch.no_grad():
                n_total = nbr_p.shape[0]
                nbr_vals = torch.zeros(n_total)

                for start in range(0, n_total, INFERENCE_BATCH_SIZE):
                    end = min(start + INFERENCE_BATCH_SIZE, n_total)
                    raw = pack_raw_state(nbr_p[start:end], nbr_o[start:end], device)
                    value, _ = target_model(raw)
                    nbr_vals[start:end] = value.squeeze(-1).cpu()

                nbr_vals = nbr_vals.view(BATCH_SIZE, 12)

            labels = compute_labels(pieces, orients, nbr_p, nbr_o, nbr_vals)
            labels_t = torch.from_numpy(labels).float().to(device)
            
            model.train()
            raw = pack_raw_state(pieces, orients, device)
            value, _ = model(raw)
            preds = value.squeeze(-1)

            loss = loss_fn(preds, labels_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            global_step += 1
            dt = time.time() - t0
            
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                mean_pred = preds.mean().item()
                batch_avg = (time.time() - batch_start) / (batch_idx + 1)
                
                with torch.no_grad():
                    solved_mask = labels == 0.0
                    n_solved = int(solved_mask.sum())

                print(
                    f"  batch {batch_idx:5d}/{num_batches} | step {global_step:6d} | "
                    f"loss {loss.item():.4f} | pred_mean {mean_pred:.2f} | "
                    f"label_mean {labels.mean():.2f} | solved {n_solved} | "
                    f"{dt:.2f}s/batch | avg {batch_avg:.2f}s"
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/pred_mean": mean_pred,
                        "train/label_mean": labels.mean(),
                        "train/solved": n_solved,
                        "epoch": epoch,
                    }, step=global_step)
            # if batch_idx % 1 == 0:
                # test_pkl_path = os.path.join(project_root, "test_data/cube3_test.pkl")
                # test_model(model, buffer_pieces, buffer_orients, buffer_depths, test_pkl_path)

        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s\n")
        
        target_model.load_state_dict(model.state_dict())
        
        milestone_path = None
        if epoch % MILESTONE_INTERVAL == 0:
            milestone_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
        save_checkpoint(model, target_model, optimizer, epoch, global_step, checkpoint_latest, milestone_path)
        
        buffer_pieces = None
        buffer_orients = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RelCube")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    
    train(use_wandb=args.wandb)
