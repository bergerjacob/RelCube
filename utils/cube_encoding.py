"""
Rubik's Cube Move to Piece-Based Encoding Converter
Vectorized Torch Implementation - HPC Ready.
"""

import argparse
import sys
import torch
from typing import List, Tuple
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MOVE_DATA on CPU so it works with tensors on any device
# PyTorch will handle device placement automatically during indexing
MOVE_DATA = {
    "U": (
        torch.tensor(
            [3, 0, 1, 2, 4, 5, 6, 7, 11, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
        ),
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ),
    "D": (
        torch.tensor(
            [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11, 13, 14, 15, 12, 16, 17, 18, 19]
        ),
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ),
    "L": (
        torch.tensor(
            [0, 5, 1, 3, 4, 6, 2, 7, 8, 9, 18, 11, 12, 13, 17, 15, 16, 10, 14, 19]
        ),
        torch.tensor([0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ),
    "R": (
        torch.tensor(
            [4, 1, 2, 0, 7, 5, 6, 3, 16, 9, 10, 11, 19, 13, 14, 15, 12, 17, 18, 8]
        ),
        torch.tensor([1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ),
    "F": (
        torch.tensor(
            [1, 4, 2, 3, 5, 0, 6, 7, 8, 17, 10, 11, 12, 16, 14, 15, 9, 13, 18, 19]
        ),
        torch.tensor([1, 2, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
    ),
    "B": (
        torch.tensor(
            [0, 1, 3, 7, 4, 5, 2, 6, 8, 9, 10, 19, 12, 13, 14, 18, 16, 17, 11, 15]
        ),
        torch.tensor([0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]),
    ),
}


def get_piece_encoding_from_moves(moves_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate piece encoding using Torch but returning NumPy for compatibility."""
    # Start on the GPU/CPU device
    pieces = torch.arange(20, device=device)
    orientations = torch.zeros(20, dtype=torch.long, device=device)

    move_list = parse_moves(moves_str)

    for move in move_list:
        pieces, orientations = apply_move_to_encoding(move, pieces, orientations)

    # Convert back to NumPy for the rest of the script (printing/saving)
    return pieces.cpu().numpy(), orientations.cpu().numpy()


def parse_moves(moves_str: str) -> List[str]:
    """Parse move string into list of moves."""
    valid = "URFDLB"
    moves, i = [], 0
    while i < len(moves_str):
        if moves_str[i] in valid:
            m = moves_str[i]
            if i + 1 < len(moves_str) and moves_str[i + 1] in "'2":
                m += moves_str[i + 1]
                i += 1
            moves.append(m)
        i += 1
    return moves
    return moves


def apply_move_to_encoding(move: str, pieces, orientations):
    """
    Vectorized version of move application.
    Works with both numpy arrays and torch tensors on any device.
    Returns the same type as input.
    """
    face = move[0]
    times = 1
    if len(move) > 1:
        if move[1] == "'":
            times = 3
        elif move[1] == "2":
            times = 2

    perm, delta = MOVE_DATA[face]

    # Check if input is numpy array
    is_numpy = isinstance(pieces, np.ndarray)

    if is_numpy:
        # Convert perm/delta to numpy if they're tensors
        perm = perm.cpu().numpy() if torch.is_tensor(perm) else perm
        delta = delta.cpu().numpy() if torch.is_tensor(delta) else delta

        for _ in range(times):
            pieces = pieces[perm]
            orientations = orientations[perm] + delta
            orientations[:8] = orientations[:8] % 3
            orientations[8:] = orientations[8:] % 2
    else:
        # Torch tensors - ensure perm/delta are on same device
        if perm.device != pieces.device:
            perm = perm.to(pieces.device)
        if delta.device != orientations.device:
            delta = delta.to(orientations.device)

        for _ in range(times):
            pieces = pieces[perm]
            orientations = orientations[perm] + delta
            orientations[:8] %= 3
            orientations[8:] %= 2

    return pieces, orientations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("moves", type=str, help='Move sequence (e.g., "R U\' F2")')
    parser.add_argument("-s", "--save", type=str, help="Save to .npz file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    try:
        pieces, orientations = get_piece_encoding_from_moves(args.moves)

        if args.save:
            np.savez(args.save, pieces=pieces, orientations=orientations)
            print(f"Saved to {args.save}")
        else:
            print(f"Pieces: {pieces}")
            print(f"Orients: {orientations}")

            if args.verbose:
                print("\nPiece details:")
                for i in range(20):
                    p = pieces[i]
                    o = orientations[i]
                    print(f"Piece {p:2d} at slot {i:2d}, orient={o}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
