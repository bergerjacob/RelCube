#!/usr/bin/env python3
"""
Rubik's Cube Move to Facelet Converter

Converts a series of cube moves into facelet format.
Input: Cube notation moves (R, U', F2, etc.)
Output: 54-character facelet string (UURF...)
"""

import argparse
import sys
from typing import List


# Initial solved cube state - 54 facelets
# Order: U1-9, R1-9, F1-9, D1-9, L1-9, B1-9
# Within each face:
#   1 2 3     0 1 2
#   4 5 6  =  3 4 5
#   7 8 9     6 7 8
SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

# Face indices (start of each face in the string)
FACE_START = {'U': 0, 'R': 9, 'F': 18, 'D': 27, 'L': 36, 'B': 45}

# Clockwise face rotation mapping
# 0 1 2      6 3 0
# 3 4 5  <-  7 4 1
# 6 7 8      8 5 2
FACE_ROT_CW = [6, 3, 0, 7, 4, 1, 8, 5, 2]

# Edge cycles for each face (clockwise rotation)
# Each cycle lists 12 indices representing 4 groups of 3 stickers
# that cycle around the face
EDGE_CYCLES = {
    # U: F->R, R->B, B->L, L->F
    'U': [18, 19, 20, 9, 10, 11, 45, 46, 47, 36, 37, 38],
    # D: F->L, L->B, B->R, R->F
    'D': [24, 25, 26, 42, 43, 44, 51, 52, 53, 15, 16, 17],
    # R: U->F, F->D, D->B, B->U
    'R': [2, 5, 8, 20, 23, 26, 29, 32, 35, 51, 48, 45],
    # L: U->B, B->D, D->F, F->U
    'L': [0, 3, 6, 53, 50, 47, 27, 30, 33, 18, 21, 24],
    # F: U->R, R->D, D->L, L->U
    'F': [6, 7, 8, 11, 14, 17, 29, 28, 27, 36, 39, 42],
    # B: U->L, L->D, D->R, R->U
    'B': [2, 1, 0, 38, 41, 44, 33, 34, 35, 15, 12, 9],
}


def rotate_face(state: str, face: str, times: int = 1) -> str:
    """Rotate a face's stickers."""
    start = FACE_START[face]
    state_list = list(state)
    old_face = state_list[start:start + 9]
    
    for _ in range(times % 4):
        new_face = [old_face[FACE_ROT_CW[i]] for i in range(9)]
        old_face = new_face
    
    for i in range(9):
        state_list[start + i] = old_face[i]
    
    return ''.join(state_list)


def cycle_edges(state: str, face: str, times: int = 1) -> str:
    """Cycle edge stickers around a face."""
    cycle = EDGE_CYCLES[face]
    state_list = list(state)
    
    for _ in range(times % 4):
        temp = [state_list[cycle[i]] for i in range(3)]
        for i in range(3):
            state_list[cycle[i]] = state_list[cycle[3 + i]]
            state_list[cycle[3 + i]] = state_list[cycle[6 + i]]
            state_list[cycle[6 + i]] = state_list[cycle[9 + i]]
            state_list[cycle[9 + i]] = temp[i]
    
    return ''.join(state_list)


def apply_move(state: str, move: str) -> str:
    """Apply a single move to the cube state."""
    if not move:
        return state
    
    face = move[0]
    if face not in FACE_START:
        raise ValueError(f"Invalid face: {face}")
    
    # Determine rotation: 1=90° CW, 2=180°, 3=90° CCW
    if len(move) == 1:
        times = 1
    elif move[1] == "'":
        times = 3
    elif move[1] == '2':
        times = 2
    else:
        raise ValueError(f"Invalid move: {move}")
    
    state = rotate_face(state, face, times)
    state = cycle_edges(state, face, times)
    return state


def apply_moves(state: str, moves: List[str]) -> str:
    """Apply a list of moves to the cube state."""
    for move in moves:
        move = move.strip()
        if move:
            state = apply_move(state, move)
    return state


def parse_moves(moves_str: str) -> List[str]:
    """Parse a string of moves into a list."""
    moves = []
    i = 0
    moves_str = moves_str.strip()
    while i < len(moves_str):
        if moves_str[i] in 'URFDLB':
            move = moves_str[i]
            i += 1
            if i < len(moves_str) and moves_str[i] in "'2":
                move += moves_str[i]
                i += 1
            moves.append(move)
        elif moves_str[i].isspace():
            i += 1
        else:
            raise ValueError(f"Invalid character: {moves_str[i]} at position {i}")
    return moves


def moves_to_facelet(moves: str) -> str:
    """Convert a series of moves to a facelet string."""
    move_list = parse_moves(moves)
    return apply_moves(SOLVED_STATE, move_list)


def inverse_moves(moves: str) -> str:
    """Get the inverse of a series of moves."""
    move_list = parse_moves(moves)
    inverse = []
    for move in reversed(move_list):
        if len(move) == 1:
            inverse.append(move + "'")
        elif move[1] == "'":
            inverse.append(move[0])
        elif move[1] == '2':
            inverse.append(move)
    return ' '.join(inverse)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Rubik's cube moves to facelet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "R U R' U'"              # Convert moves to facelet string
  %(prog)s -i "R U R'"              # Get inverse of moves
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('moves', nargs='?', type=str,
                       help='Moves to convert (e.g., "R U R\' U\'")')
    group.add_argument('-i', '--inverse', type=str,
                       help='Get inverse of moves')
    
    args = parser.parse_args()
    
    if args.inverse:
        try:
            inv = inverse_moves(args.inverse)
            print(inv)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.moves:
        try:
            facelet = moves_to_facelet(args.moves)
            print(facelet)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
