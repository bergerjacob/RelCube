#!/usr/bin/env python3
"""
Beam Search Test for RelCube Model
Uses wide beam to explore multiple paths in parallel.

Usage:
    python test_astar.py --checkpoint checkpoints/checkpoint_latest.pt --num_test 5 --beam_width 100
"""

import torch
import numpy as np
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RelCube
from utils.cube_encoding import apply_move_to_encoding

ALL_MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]
SOLVED_PIECES = np.arange(20, dtype=np.int32)
SOLVED_ORIENTS = np.zeros(20, dtype=np.int32)


def is_solved(p, o):
    return np.array_equal(p, SOLVED_PIECES) and np.array_equal(o, SOLVED_ORIENTS)


def solve_beam_search(
    model, start_p, start_o, expected, device, beam_width=100, max_depth=30
):
    """Wide beam search - keep top beam_width states at each depth."""
    start_time = time.time()

    if is_solved(start_p, start_o):
        return {
            "solved": True,
            "solution_len": 0,
            "nodes": 0,
            "time": 0,
            "expected": expected,
        }

    # Initialize beam with starting state
    # Each entry: (pieces, orients, path)
    beam = [(start_p.copy(), start_o.copy(), [])]
    visited = {tuple(start_p.tolist()) + tuple(start_o.tolist())}
    total_nodes = 0

    for depth in range(max_depth):
        if not beam:
            break

        # Generate all neighbors
        all_nbr_p = []
        all_nbr_o = []
        all_path = []

        for p, o, path in beam:
            for move in ALL_MOVES:
                p_t = torch.from_numpy(p).to(device)
                o_t = torch.from_numpy(o).to(device)
                p_t, o_t = apply_move_to_encoding(move, p_t, o_t)
                p_n = p_t.cpu().numpy()
                o_n = o_t.cpu().numpy()

                # Check if solved immediately
                if is_solved(p_n, o_n):
                    return {
                        "solved": True,
                        "solution_len": len(path) + 1,
                        "nodes": total_nodes,
                        "time": time.time() - start_time,
                        "expected": expected,
                    }

                state_key = tuple(p_n.tolist()) + tuple(o_n.tolist())
                if state_key not in visited:
                    visited.add(state_key)
                    all_nbr_p.append(p_n)
                    all_nbr_o.append(o_n)
                    all_path.append(path + [move])

        if not all_nbr_p:
            break

        total_nodes += len(all_nbr_p)

        # Batch inference for all neighbors
        raw = np.stack([all_nbr_p, all_nbr_o], axis=1)
        raw_torch = torch.from_numpy(raw).long().to(device)

        with torch.no_grad():
            h_values = model(raw_torch)[0].squeeze(-1).cpu().numpy()

        # Sort by heuristic and keep top beam_width
        sorted_indices = np.argsort(h_values)
        beam = []
        for idx in sorted_indices[:beam_width]:
            beam.append((all_nbr_p[idx], all_nbr_o[idx], all_path[idx]))

    return {
        "solved": False,
        "solution_len": -1,
        "nodes": total_nodes,
        "time": time.time() - start_time,
        "expected": expected,
    }


def main():
    parser = argparse.ArgumentParser(description="Beam search test for RelCube")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt"
    )
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=30)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = RelCube().to(device)
    model.eval()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # Load test data
    data_path = "data_0/data_0.npz"
    print(f"Loading: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    pieces = data["pieces"][: args.num_test]
    orientations = data["orientations"][: args.num_test]
    solution_lengths = data["solution_lengths"][: args.num_test]

    total = len(pieces)

    print(f"\n{'=' * 80}")
    print(
        f"Beam Search Test: {total} states (beam_width={args.beam_width}, max_depth={args.max_depth})"
    )
    print(f"{'=' * 80}\n")

    # Process all states
    total_start = time.time()
    results = []

    for i in range(total):
        result = solve_beam_search(
            model,
            pieces[i],
            orientations[i],
            int(solution_lengths[i]),
            device,
            args.beam_width,
            args.max_depth,
        )
        results.append(result)

        status = (
            "OK"
            if result["solved"] and result["solution_len"] <= result["expected"]
            else ("SUBOPT" if result["solved"] else "FAIL")
        )
        sol_len = result["solution_len"] if result["solved"] else "FAIL"
        print(
            f"State {i + 1:3d}: Exp={result['expected']:2d} | Got={sol_len:>5s} | "
            f"Nodes={result['nodes']:7d} | Time={result['time']:.2f}s | {status}"
        )

    total_time = time.time() - total_start

    # Analyze results
    solved_count = sum(1 for r in results if r["solved"])
    optimal_count = sum(
        1 for r in results if r["solved"] and r["solution_len"] <= r["expected"]
    )

    solved_results = [r for r in results if r["solved"]]
    avg_nodes = np.mean([r["nodes"] for r in solved_results]) if solved_results else 0
    avg_time = np.mean([r["time"] for r in solved_results]) if solved_results else 0

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Total states:         {total}")
    print(f"Solved:               {solved_count} ({100 * solved_count / total:.1f}%)")
    print(f"Optimal solutions:    {optimal_count} ({100 * optimal_count / total:.1f}%)")
    if solved_results:
        print(f"Avg nodes expanded:   {avg_nodes:.0f}")
        print(f"Avg solve time:       {avg_time:.2f}s")
    print(f"Total time:           {total_time:.2f}s ({total_time / total:.2f}s/state)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
