import torch
import numpy as np
import time
import sys
import os
import heapq

# Setup path so imports work identically to train.py
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from model import RelCube
from utils.cube_encoding import apply_move_to_encoding

ALL_MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]
SOLVED_PIECES = np.arange(20, dtype=np.int32)
SOLVED_ORIENTS = np.zeros(20, dtype=np.int32)

def pack_raw_state(edges, orients, device):
    raw = np.stack([edges, orients], axis=1)
    return torch.from_numpy(raw).long().to(device)

def is_solved(p, o):
    return np.array_equal(p, SOLVED_PIECES) and np.array_equal(o, SOLVED_ORIENTS)

def generate_scramble(depth):
    """Generates a random scramble of a specific depth."""
    p = SOLVED_PIECES.copy()
    o = SOLVED_ORIENTS.copy()
    prev_face = None
    moves_applied = []
    
    for _ in range(depth):
        while True:
            move = ALL_MOVES[np.random.randint(12)]
            if move[0] != prev_face:
                break
        prev_face = move[0]
        apply_move_to_encoding(move, p, o)
        moves_applied.append(move)
        
    return p, o, moves_applied

def format_path(start_h, path_with_h):
    """Helper to format the path with heuristic guesses nicely."""
    if not path_with_h:
        return f"Start (h={start_h:.2f}) -> [No moves]"
    
    out = f"Start (h={start_h:.2f})"
    for move, h in path_with_h:
        out += f" -> {move} (h={h:.2f})"
    return out

def solve_astar(model, start_p, start_o, device, max_nodes=2000):
    """
    A* Search that tracks the heuristic h(n) for debugging.
    """
    if is_solved(start_p, start_o):
        return True, 0, [], 0.0, []

    # Get the starting heuristic value
    raw_start = pack_raw_state(np.expand_dims(start_p, 0), np.expand_dims(start_o, 0), device)
    with torch.no_grad():
        start_val, _ = model(raw_start)
        h_start = start_val.item()

    # Priority queue: (f_score, counter, g_score, pieces, orientations, path_with_h)
    # path_with_h will store tuples of (move, h_value)
    counter = 0
    open_set = []
    heapq.heappush(open_set, (h_start, counter, 0, start_p.copy(), start_o.copy(), []))
    
    start_hash = tuple(start_p.tolist()) + tuple(start_o.tolist())
    best_g = {start_hash: 0}

    # Tracking for failures: keep the path that got to the lowest h value
    best_h_seen = h_start
    closest_path = []
    nodes_expanded = 0

    while open_set and nodes_expanded < max_nodes:
        _, _, g, current_p, current_o, path = heapq.heappop(open_set)
        nodes_expanded += 1

        nbr_p = np.zeros((12, 20), dtype=np.int32)
        nbr_o = np.zeros((12, 20), dtype=np.int32)
        
        for i, move in enumerate(ALL_MOVES):
            p_copy = current_p.copy()
            o_copy = current_o.copy()
            apply_move_to_encoding(move, p_copy, o_copy)
            nbr_p[i] = p_copy
            nbr_o[i] = o_copy

        # Quick check if any neighbor is solved
        for i, move in enumerate(ALL_MOVES):
            if is_solved(nbr_p[i], nbr_o[i]):
                final_path = path + [(move, 0.0)]
                return True, len(final_path), final_path, h_start, final_path

        # Evaluate all 12 neighbors in one batch
        raw = pack_raw_state(nbr_p, nbr_o, device)
        with torch.no_grad():
            values, _ = model(raw)
            values = values.squeeze(-1).cpu().numpy()

        for i, move in enumerate(ALL_MOVES):
            p_next = nbr_p[i]
            o_next = nbr_o[i]
            state_hash = tuple(p_next.tolist()) + tuple(o_next.tolist())
            
            tentative_g = g + 1
            
            if state_hash not in best_g or tentative_g < best_g[state_hash]:
                best_g[state_hash] = tentative_g
                h = values[i]
                f = tentative_g + h
                
                # Update our debugging tracker if we found a state closer to solved
                if h < best_h_seen:
                    best_h_seen = h
                    closest_path = path + [(move, h)]
                
                counter += 1
                new_path = path + [(move, h)]
                heapq.heappush(open_set, (f, counter, tentative_g, p_next, o_next, new_path))

    # Failed to find a solution within max_nodes
    return False, nodes_expanded, [], h_start, closest_path

def run_tests(checkpoint_path, num_tests_per_depth=5, test_depths=[3, 5, 8, 10, 12, 15, 18, 21, 25, 29, 35]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = RelCube().to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("\n--- Starting Evaluation (A* Search with Heuristic Debugging) ---")
    for depth in test_depths:
        successes = 0
        total_solve_steps = 0
        
        print(f"\n=========================================")
        print(f"  Testing Scramble Depth: {depth}")
        print(f"=========================================")
        
        for i in range(num_tests_per_depth):
            p, o, scramble_moves = generate_scramble(depth)
            print(f"\n[Test {i+1}] Scramble: {' '.join(scramble_moves)}")
            
            solved, steps_taken, path, start_h, closest_path = solve_astar(model, p, o, device)
            
            if solved and steps_taken <= depth:
                successes += 1
                total_solve_steps += steps_taken
                print(f"  -> SUCCESS! Solved in {steps_taken} moves.")
                print(f"  -> Path: {format_path(start_h, path)}")
            
            elif solved:
                print(f"  -> FAILED (Suboptimal). Solved in {steps_taken} moves (expected <= {depth}).")
                print(f"  -> Path: {format_path(start_h, path)}")
            
            else:
                print(f"  -> FAILED (Timeout). Expanded {steps_taken} nodes without solving.")
                print(f"  -> The closest the model thought it got was h={closest_path[-1][1]:.2f} if it had taken this path:")
                print(f"  -> Best Path Attempt: {format_path(start_h, closest_path)}")
                print(f"  -> (Notice if the 'h' values stop decreasing—this indicates flat values or a local minimum).")

        win_rate = (successes / num_tests_per_depth) * 100
        avg_steps = (total_solve_steps / successes) if successes > 0 else 0
        print(f"\nDepth {depth} Summary: {win_rate:.1f}% Strict Solve Rate | Avg steps of successful solves: {avg_steps:.1f}")

if __name__ == "__main__":
    CHECKPOINT_FILE = "checkpoint_latest.pt" 
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE) if 'CHECKPOINT_DIR' in globals() else os.path.join(project_root, "checkpoints", CHECKPOINT_FILE)
    run_tests(ckpt_path)
