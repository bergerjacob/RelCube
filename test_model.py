#!/usr/bin/env python3
"""
Quick Model Evaluation Script
Evaluates heuristic quality on NPZ data - fast for training monitoring.

Usage:
    python test_model.py --checkpoint checkpoints/checkpoint_latest.pt --num_test 100
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


def pack_raw_state(edges, orients, device):
    edges = np.asarray(edges)
    orients = np.asarray(orients)
    if edges.ndim == 1:
        edges = edges[np.newaxis, :]
        orients = orients[np.newaxis, :]
    raw = np.stack([edges, orients], axis=1)
    return torch.from_numpy(raw).long().to(device)


def is_solved(p, o):
    return np.array_equal(p, SOLVED_PIECES) and np.array_equal(o, SOLVED_ORIENTS)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RelCube model")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt')
    parser.add_argument('--npz', type=str, default='data_0/data_0.npz')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = RelCube().to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    data = np.load(args.npz, allow_pickle=True)
    pieces = data['pieces'][:args.num_test]
    orientations = data['orientations'][:args.num_test]
    solution_lengths = data['solution_lengths'][:args.num_test]
    
    n = len(pieces)
    
    # Batched heuristic inference
    start_time = time.time()
    heuristics = np.zeros(n)
    for i in range(0, n, args.batch_size):
        end = min(i + args.batch_size, n)
        raw = pack_raw_state(pieces[i:end], orientations[i:end], device)
        with torch.no_grad():
            heuristics[i:end] = model(raw)[0].squeeze(-1).cpu().numpy()
    eval_time = time.time() - start_time
    
    # Analyze heuristic quality
    print(f"\nHeuristic evaluation: {eval_time:.3f}s ({eval_time/n*1000:.1f}ms/state)")
    print(f"\nSolution length vs heuristic:")
    print(f"  h range: [{heuristics.min():.1f}, {heuristics.max():.1f}]")
    print(f"  h mean:  {heuristics.mean():.1f}")
    print(f"  h std:   {heuristics.std():.2f}")
    
    # Correlation
    if len(np.unique(solution_lengths)) > 1:
        corr = np.corrcoef(heuristics, solution_lengths)[0, 1]
        print(f"  corr(h, length): {corr:.3f}")
    
    # Check heuristic improvement rate (how often a neighbor has lower h)
    print("\nNeighborhood analysis (first 10 states):")
    improve_counts = []
    for i in range(min(10, n)):
        p, o = pieces[i], orientations[i]
        h_start = heuristics[i]
        
        improved = 0
        for move in ALL_MOVES:
            p_t = torch.from_numpy(p).to(device)
            o_t = torch.from_numpy(o).to(device)
            p_n, o_n = apply_move_to_encoding(move, p_t, o_t)
            p_n, o_n = p_n.cpu().numpy(), o_n.cpu().numpy()
            raw = pack_raw_state(p_n, o_n, device)
            with torch.no_grad():
                h_n = model(raw)[0].item()
            if h_n < h_start:
                improved += 1
        
        improve_counts.append(improved)
        exp = int(solution_lengths[i])
        print(f"  {i+1:2d}: exp={exp:2d}, h={h_start:5.1f}, neighbors_better={improved}/12")
    
    avg_improve = np.mean(improve_counts)
    print(f"\n  Avg neighbors with better h: {avg_improve:.1f}/12")
    
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
