"""
Rubik's Cube Move to Piece-Based Encoding Converter
Piece ID and orientation only - no facelet manipulation.
"""

import argparse
import sys
from typing import List, Tuple
import numpy as np


# MOVE_INFO: face -> (corner_cycle, edge_cycle)
# Cycle lists show slots in clockwise order: slot[i] receives from slot[(i-1)%4]
MOVE_INFO = {
    'U': ([0, 1, 2, 3], [8, 9, 10, 11]),
    'D': ([7, 6, 5, 4], [15, 14, 13, 12]),
    'L': ([1, 5, 6, 2], [10, 17, 14, 18]),
    'R': ([0, 3, 7, 4], [8, 19, 12, 16]),
    'F': ([0, 4, 5, 1], [9, 16, 13, 17]),
    'B': ([2, 6, 7, 3], [11, 18, 15, 19]),
}

# ADJUST: face -> (corner_adjust, edge_adjust)
# corner_adjust: {from_slot: orient_delta} or None
# edge_adjust: {from_slot: orient_delta} or None
ADJUST = {
    'U': (None, None),
    'D': (None, None),
    'L': ({1: -1, 5: 1, 6: -1, 2: 1}, None),
    'R': ({0: 1, 3: -1, 7: 1, 4: -1}, None),
    'F': ({0: -1, 4: 1, 5: -1, 1: 1}, {9: 1, 16: 1, 13: 1, 17: 1}),
    'B': ({3: 1, 2: -1, 6: 1, 7: -1}, {11: 1, 18: 1, 15: 1, 19: 1}),
}


def get_piece_encoding_from_moves(moves_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate piece encoding after applying moves from solved state."""
    pieces = np.arange(20, dtype=np.int32)
    orientations = np.zeros(20, dtype=np.int32)
    
    move_list = parse_moves(moves_str)
    
    for move in move_list:
        apply_move_to_encoding(move, pieces, orientations)
    
    return pieces, orientations


def parse_moves(moves_str: str) -> List[str]:
    """Parse move string into list of moves."""
    valid = 'URFDLB'
    moves = []
    i = 0
    while i < len(moves_str):
        if moves_str[i] in valid:
            m = moves_str[i]
            if i + 1 < len(moves_str) and moves_str[i+1] in "'2":
                m += moves_str[i+1]
                i += 1
            moves.append(m)
        i += 1
    return moves


def apply_move_to_encoding(move: str, 
                           pieces: np.ndarray, 
                           orientations: np.ndarray) -> None:
    """Apply a move to the piece encoding in-place."""
    face = move[0]
    times = 1
    
    if len(move) > 1:
        if move[1] == "'": 
            times = 3
        elif move[1] == '2': 
            times = 2
    
    corner_cycle, edge_cycle = MOVE_INFO[face]
    corner_adjust, edge_adjust = ADJUST[face]
    
    for _ in range(times):
        _cycle_items(corner_cycle, pieces, orientations, corner_adjust, 3)
        _cycle_items(edge_cycle, pieces, orientations, edge_adjust, 2)


def _cycle_items(cycle: List[int], 
                 pieces: np.ndarray, 
                 orientations: np.ndarray,
                 adjust: dict,
                 mod: int) -> None:
    """Cycle items (corners or edges) with orientation adjustments."""
    saved_pieces = [pieces[s] for s in cycle]
    saved_orients = [orientations[s] for s in cycle]
    
    for i in range(len(cycle)):
        to_slot = cycle[i]
        from_idx = (i - 1) % len(cycle)
        
        pieces[to_slot] = saved_pieces[from_idx]
        
        new_orient = saved_orients[from_idx]
        
        if adjust is not None and cycle[from_idx] in adjust:
            delta = adjust[cycle[from_idx]]
            new_orient = (new_orient + delta) % mod
        
        orientations[to_slot] = new_orient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('moves', type=str, help='Move sequence (e.g., "R U\' F2")')
    parser.add_argument('-s', '--save', type=str, help='Save to .npz file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
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
