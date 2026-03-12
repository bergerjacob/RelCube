#!/usr/bin/env python3
"""
A* Test Script for RelCube Model (DeepCubeA-style)
Replicates DeepCubeA testing protocol with our NPZ data format.

Usage:
    python test_astar.py --npz data_0/data_0.npz --checkpoint checkpoints/checkpoint_latest.pt --num_test 100
"""

import torch
import numpy as np
import argparse
import time
import heapq
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


def solve_astar(model, start_p, start_o, device, weight=1.0, max_nodes=100000):
    """Weighted A* search following DeepCubeA protocol."""
    start_time = time.time()
    h_start = 0.0
    
    if is_solved(start_p, start_o):
        return True, 0, 0, time.time() - start_time, [], h_start

    raw_start = pack_raw_state(np.expand_dims(start_p, 0), np.expand_dims(start_o, 0), device)
    with torch.no_grad():
        start_val, _ = model(raw_start)
        h_start = start_val.item()

    counter = 0
    open_set = []
    heapq.heappush(open_set, (weight * 0 + h_start, counter, 0, start_p.copy(), start_o.copy(), []))
    
    start_hash = tuple(start_p.tolist()) + tuple(start_o.tolist())
    closed_dict = {start_hash: 0}
    nodes_expanded = 0

    while open_set and nodes_expanded < max_nodes:
        _, _, g, current_p, current_o, path = heapq.heappop(open_set)
        nodes_expanded += 1

        nbr_p = np.zeros((12, 20), dtype=np.int32)
        nbr_o = np.zeros((12, 20), dtype=np.int32)
        
        for i, move in enumerate(ALL_MOVES):
            p_copy = torch.from_numpy(current_p).to(device)
            o_copy = torch.from_numpy(current_o).to(device)
            p_copy, o_copy = apply_move_to_encoding(move, p_copy, o_copy)
            nbr_p[i] = p_copy.cpu().numpy()
            nbr_o[i] = o_copy.cpu().numpy()

        for i, move in enumerate(ALL_MOVES):
            if is_solved(nbr_p[i], nbr_o[i]):
                solve_time = time.time() - start_time
                return True, len(path) + 1, nodes_expanded, solve_time, path + [move], h_start

        raw = pack_raw_state(nbr_p, nbr_o, device)
        with torch.no_grad():
            values, _ = model(raw)
            values = values.squeeze(-1).cpu().numpy()

        for i, move in enumerate(ALL_MOVES):
            p_next = nbr_p[i]
            o_next = nbr_o[i]
            state_hash = tuple(p_next.tolist()) + tuple(o_next.tolist())
            
            tentative_g = g + 1
            
            if state_hash not in closed_dict or tentative_g < closed_dict[state_hash]:
                closed_dict[state_hash] = tentative_g
                h = values[i]
                f = weight * tentative_g + h
                
                counter += 1
                new_path = path + [(move, h)]
                heapq.heappush(open_set, (f, counter, tentative_g, p_next, o_next, new_path))

    solve_time = time.time() - start_time
    return False, nodes_expanded, nodes_expanded, solve_time, [], h_start


def main():
    parser = argparse.ArgumentParser(description="A* test for RelCube model (DeepCubeA-style)")
    parser.add_argument('--npz', type=str, default='data_0/data_0.npz', help='NPZ file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt', help='Checkpoint')
    parser.add_argument('--num_test', type=int, default=100, help='Number of states')
    parser.add_argument('--weight', type=float, default=1.0, help='Weight for path cost (1.0 = A*)')
    parser.add_argument('--max_nodes', type=int, default=100000, help='Max nodes per state')
    
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
    
    print(f"Loading: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    
    pieces = data['pieces']
    orientations = data['orientations']
    solution_lengths = data['solution_lengths']
    
    total = min(args.num_test, len(pieces))
    
    print(f"\n{'='*70}")
    print(f"A* Test: {total} states (weight={args.weight}, max_nodes={args.max_nodes})")
    print(f"{'='*70}")
    
    results = {'solved': [], 'sol_costs': [], 'nodes': [], 'times': [], 'expected': []}
    
    for i in range(total):
        p, o = pieces[i], orientations[i]
        expected = int(solution_lengths[i])
        
        solved, sol_len, nodes, t, path, _ = solve_astar(model, p, o, device, args.weight, args.max_nodes)
        
        results['solved'].append(solved)
        results['sol_costs'].append(sol_len if solved else -1)
        results['nodes'].append(nodes)
        results['times'].append(t)
        results['expected'].append(expected)
        
        status = "OK" if solved and sol_len <= expected else ("SUBOPT" if solved else "FAIL")
        got = str(sol_len) if solved else "FAIL"
        print(f"State {i+1:4d}: Exp={expected:2d} | Got={got:>5s} | Nodes={nodes:7d} | Time={t:6.2f}s | {status}")
    
    solved_count = sum(results['solved'])
    optimal_count = sum(1 for s, c, e in zip(results['solved'], results['sol_costs'], results['expected']) if s and c <= e)
    
    avg_nodes = np.mean([n for s, n in zip(results['solved'], results['nodes']) if s])
    avg_time = np.mean([t for s, t in zip(results['solved'], results['times']) if s])
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total:           {total}")
    print(f"Solved:          {solved_count} ({100*solved_count/total:.1f}%)")
    print(f"Optimal:         {optimal_count} ({100*optimal_count/total:.1f}%)")
    if solved_count > 0:
        print(f"Avg nodes:       {avg_nodes:.0f}")
        print(f"Avg time:        {avg_time:.2f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
