#!/usr/bin/env python3
"""
RelCube Evaluation Suite

Modes:
  solve-rate   A* solve-rate evaluation at various scramble depths
  astar        Batched A* search on pre-computed test states
  beam         Beam search on pre-computed test states
  heuristic    Quick heuristic quality check (no search)

Usage:
  python eval.py solve-rate
  python eval.py astar --num_test 5
  python eval.py beam --num_test 5
  python eval.py heuristic --num_test 100
"""

import torch
import numpy as np
import argparse
import time
import os
import sys
import heapq
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RelCube
from utils.cube_encoding import apply_move_to_encoding

ALL_MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]
SOLVED_PIECES = np.arange(20, dtype=np.int32)
SOLVED_ORIENTS = np.zeros(20, dtype=np.int32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def pack_raw_state(pieces, orients, device):
    pieces = np.asarray(pieces)
    orients = np.asarray(orients)
    if pieces.ndim == 1:
        pieces = pieces[np.newaxis, :]
        orients = orients[np.newaxis, :]
    raw = np.stack([pieces, orients], axis=1)
    return torch.from_numpy(raw).long().to(device)


def is_solved(p, o):
    return np.array_equal(p, SOLVED_PIECES) and np.array_equal(o, SOLVED_ORIENTS)


def load_model(checkpoint_path, device):
    model = RelCube().to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def generate_scramble(depth):
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


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def solve_astar(model, start_p, start_o, device, batch_size=100, max_nodes=2000):
    if is_solved(start_p, start_o):
        return True, 0, []

    raw_start = pack_raw_state(start_p, start_o, device)
    with torch.no_grad():
        h_start = model(raw_start)[0].item()

    counter = itertools.count()
    open_set = []
    heapq.heappush(open_set, (h_start, next(counter), 0, start_p.copy(), start_o.copy(), []))
    start_hash = tuple(start_p.tolist()) + tuple(start_o.tolist())
    best_g = {start_hash: 0}
    nodes_expanded = 0

    while open_set and nodes_expanded < max_nodes:
        current_batch = []
        while open_set and len(current_batch) < batch_size:
            current_batch.append(heapq.heappop(open_set))

        for f, _, g, p, o, path in current_batch:
            if is_solved(p, o):
                return True, len(path), path

        all_child_p, all_child_o = [], []
        all_child_g, all_child_path = [], []

        for f, _, g, p, o, path in current_batch:
            for move in ALL_MOVES:
                p_t = torch.from_numpy(p).to(device)
                o_t = torch.from_numpy(o).to(device)
                p_t, o_t = apply_move_to_encoding(move, p_t, o_t)
                p_n, o_n = p_t.cpu().numpy(), o_t.cpu().numpy()

                state_key = tuple(p_n.tolist()) + tuple(o_n.tolist())
                child_g = g + 1

                if state_key not in best_g or child_g < best_g[state_key]:
                    best_g[state_key] = child_g
                    all_child_p.append(p_n)
                    all_child_o.append(o_n)
                    all_child_g.append(child_g)
                    all_child_path.append(path + [move])

        if not all_child_p:
            continue

        nodes_expanded += len(all_child_p)

        raw = pack_raw_state(np.array(all_child_p), np.array(all_child_o), device)
        with torch.no_grad():
            h_values = model(raw)[0].squeeze(-1).cpu().numpy()

        for i in range(len(all_child_p)):
            child_f = all_child_g[i] + h_values[i]
            heapq.heappush(open_set, (
                child_f, next(counter), all_child_g[i],
                all_child_p[i], all_child_o[i], all_child_path[i]
            ))

    return False, nodes_expanded, []


def solve_beam_search(model, start_p, start_o, device, beam_width=200, max_depth=30):
    if is_solved(start_p, start_o):
        return True, 0, []

    beam = [(start_p.copy(), start_o.copy(), [])]
    visited = {tuple(start_p.tolist()) + tuple(start_o.tolist())}
    total_nodes = 0

    for depth in range(max_depth):
        if not beam:
            break

        all_nbr_p, all_nbr_o, all_path = [], [], []

        for p, o, path in beam:
            for move in ALL_MOVES:
                p_t = torch.from_numpy(p).to(device)
                o_t = torch.from_numpy(o).to(device)
                p_t, o_t = apply_move_to_encoding(move, p_t, o_t)
                p_n, o_n = p_t.cpu().numpy(), o_t.cpu().numpy()

                if is_solved(p_n, o_n):
                    return True, len(path) + 1, path + [move]

                state_key = tuple(p_n.tolist()) + tuple(o_n.tolist())
                if state_key not in visited:
                    visited.add(state_key)
                    all_nbr_p.append(p_n)
                    all_nbr_o.append(o_n)
                    all_path.append(path + [move])

        if not all_nbr_p:
            break

        total_nodes += len(all_nbr_p)

        raw = pack_raw_state(np.array(all_nbr_p), np.array(all_nbr_o), device)
        with torch.no_grad():
            h_values = model(raw)[0].squeeze(-1).cpu().numpy()

        sorted_indices = np.argsort(h_values)
        beam = []
        for idx in sorted_indices[:beam_width]:
            beam.append((all_nbr_p[idx], all_nbr_o[idx], all_path[idx]))

    return False, total_nodes, []


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def eval_solve_rate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    test_depths = [3, 5, 8, 10, 12, 15, 18]

    print(f"\n{'Depth':<8} {'Solved':<10} {'Rate':<10} {'Avg Steps':<10}")
    print("-" * 40)

    results = []
    for depth in test_depths:
        successes = 0
        total_steps = 0

        for _ in range(args.num_test):
            p, o, _ = generate_scramble(depth)
            solved, steps, _ = solve_astar(model, p, o, device,
                                           batch_size=args.batch_size,
                                           max_nodes=args.max_nodes)
            if solved and steps <= depth:
                successes += 1
                total_steps += steps

        rate = (successes / args.num_test) * 100
        avg = (total_steps / successes) if successes > 0 else 0
        print(f"{depth:<8} {successes}/{args.num_test:<6} {rate:>6.1f}%    {avg:>5.1f}")
        results.append((depth, rate, avg))

    print("-" * 40)
    print("\nSummary:")
    print(f"{'Depth':<8} {'Rate':<10}")
    for depth, rate, _ in results:
        bar = "#" * int(rate / 5)
        print(f"{depth:<8} {rate:>5.1f}% {bar}")


def eval_astar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    data = np.load(args.data, allow_pickle=True)
    pieces = data["pieces"][:args.num_test]
    orientations = data["orientations"][:args.num_test]
    solution_lengths = data["solution_lengths"][:args.num_test]
    total = len(pieces)

    print(f"\nA* Search: {total} states (batch_size={args.batch_size}, max_nodes={args.max_nodes})\n")

    total_start = time.time()
    results = []
    for i in range(total):
        result = solve_astar(model, pieces[i], orientations[i], device,
                             batch_size=args.batch_size, max_nodes=args.max_nodes)
        solved, nodes, path = result
        expected = int(solution_lengths[i])
        results.append({"solved": solved, "solution_len": len(path) if solved else -1,
                        "nodes": nodes, "expected": expected})

        status = "OK" if solved and len(path) <= expected else ("SUBOPT" if solved else "FAIL")
        sol_len = str(len(path)) if solved else "FAIL"
        print(f"State {i+1:3d}: Exp={expected:2d} | Got={sol_len:>5s} | "
              f"Nodes={nodes:7d} | {status}")

    total_time = time.time() - total_start
    solved_count = sum(1 for r in results if r["solved"])
    optimal_count = sum(1 for r in results if r["solved"] and r["solution_len"] <= r["expected"])
    solved_results = [r for r in results if r["solved"]]

    print(f"\n{'='*60}")
    print(f"Solved: {solved_count}/{total} ({100*solved_count/total:.1f}%)")
    print(f"Optimal: {optimal_count}/{total} ({100*optimal_count/total:.1f}%)")
    if solved_results:
        print(f"Avg nodes: {np.mean([r['nodes'] for r in solved_results]):.0f}")
        print(f"Avg time:  {total_time/total:.2f}s/state")
    print(f"{'='*60}")


def eval_beam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    data = np.load(args.data, allow_pickle=True)
    pieces = data["pieces"][:args.num_test]
    orientations = data["orientations"][:args.num_test]
    solution_lengths = data["solution_lengths"][:args.num_test]
    total = len(pieces)

    print(f"\nBeam Search: {total} states (beam_width={args.beam_width}, max_depth={args.max_depth})\n")

    total_start = time.time()
    results = []
    for i in range(total):
        result = solve_beam_search(model, pieces[i], orientations[i], device,
                                   beam_width=args.beam_width, max_depth=args.max_depth)
        solved, nodes, path = result
        expected = int(solution_lengths[i])
        results.append({"solved": solved, "solution_len": len(path) if solved else -1,
                        "nodes": nodes, "expected": expected})

        status = "OK" if solved and len(path) <= expected else ("SUBOPT" if solved else "FAIL")
        sol_len = str(len(path)) if solved else "FAIL"
        print(f"State {i+1:3d}: Exp={expected:2d} | Got={sol_len:>5s} | "
              f"Nodes={nodes:7d} | {status}")

    total_time = time.time() - total_start
    solved_count = sum(1 for r in results if r["solved"])
    optimal_count = sum(1 for r in results if r["solved"] and r["solution_len"] <= r["expected"])
    solved_results = [r for r in results if r["solved"]]

    print(f"\n{'='*60}")
    print(f"Solved: {solved_count}/{total} ({100*solved_count/total:.1f}%)")
    print(f"Optimal: {optimal_count}/{total} ({100*optimal_count/total:.1f}%)")
    if solved_results:
        print(f"Avg nodes: {np.mean([r['nodes'] for r in solved_results]):.0f}")
        print(f"Avg time:  {total_time/total:.2f}s/state")
    print(f"{'='*60}")


def eval_heuristic(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    data = np.load(args.data, allow_pickle=True)
    pieces = data["pieces"][:args.num_test]
    orientations = data["orientations"][:args.num_test]
    solution_lengths = data["solution_lengths"][:args.num_test]
    n = len(pieces)

    start_time = time.time()
    heuristics = np.zeros(n)
    for i in range(0, n, args.batch_size):
        end = min(i + args.batch_size, n)
        raw = pack_raw_state(pieces[i:end], orientations[i:end], device)
        with torch.no_grad():
            heuristics[i:end] = model(raw)[0].squeeze(-1).cpu().numpy()
    eval_time = time.time() - start_time

    print(f"\nHeuristic evaluation: {eval_time:.3f}s ({eval_time/n*1000:.1f}ms/state)")
    print(f"\n  h range: [{heuristics.min():.1f}, {heuristics.max():.1f}]")
    print(f"  h mean:  {heuristics.mean():.1f}")
    print(f"  h std:   {heuristics.std():.2f}")

    if len(np.unique(solution_lengths)) > 1:
        corr = np.corrcoef(heuristics, solution_lengths)[0, 1]
        print(f"  corr(h, length): {corr:.3f}")

    print("\nNeighborhood analysis (first 10 states):")
    improve_counts = []
    for i in range(min(10, n)):
        h_start = heuristics[i]
        improved = 0
        for move in ALL_MOVES:
            p_t = torch.from_numpy(pieces[i]).to(device)
            o_t = torch.from_numpy(orientations[i]).to(device)
            p_n, o_n = apply_move_to_encoding(move, p_t, o_t)
            raw = pack_raw_state(p_n.cpu().numpy(), o_n.cpu().numpy(), device)
            with torch.no_grad():
                h_n = model(raw)[0].item()
            if h_n < h_start:
                improved += 1
        improve_counts.append(improved)
        exp = int(solution_lengths[i])
        print(f"  {i+1:2d}: exp={exp:2d}, h={h_start:5.1f}, neighbors_better={improved}/12")

    print(f"\n  Avg neighbors with better h: {np.mean(improve_counts):.1f}/12")
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RelCube Evaluation Suite")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # solve-rate
    p_sr = subparsers.add_parser("solve-rate", help="A* solve-rate at various depths")
    p_sr.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    p_sr.add_argument("--num_test", type=int, default=10)
    p_sr.add_argument("--batch_size", type=int, default=100)
    p_sr.add_argument("--max_nodes", type=int, default=2000)

    # astar
    p_as = subparsers.add_parser("astar", help="Batched A* on pre-computed states")
    p_as.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    p_as.add_argument("--data", type=str, default="data_0/data_0.npz")
    p_as.add_argument("--num_test", type=int, default=5)
    p_as.add_argument("--batch_size", type=int, default=100)
    p_as.add_argument("--max_nodes", type=int, default=100000)

    # beam
    p_bm = subparsers.add_parser("beam", help="Beam search on pre-computed states")
    p_bm.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    p_bm.add_argument("--data", type=str, default="data_0/data_0.npz")
    p_bm.add_argument("--num_test", type=int, default=5)
    p_bm.add_argument("--beam_width", type=int, default=200)
    p_bm.add_argument("--max_depth", type=int, default=30)

    # heuristic
    p_he = subparsers.add_parser("heuristic", help="Quick heuristic quality check")
    p_he.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    p_he.add_argument("--data", type=str, default="data_0/data_0.npz")
    p_he.add_argument("--num_test", type=int, default=100)
    p_he.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    modes = {
        "solve-rate": eval_solve_rate,
        "astar": eval_astar,
        "beam": eval_beam,
        "heuristic": eval_heuristic,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
