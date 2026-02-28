
###################################################################################################################
# NOTE: I am not sure this is working it needs to be tested independantly, the other encoding utility def does work
###################################################################################################################

#!/usr/bin/env python3
"""
Rubik's Cube Move to Piece-Based Encoding Converter
Verified logic for physical consistency.
"""

import argparse
import sys
from typing import List, Tuple
import numpy as np

# Facelet definitions
# Order: U (0-8), R (9-17), F (18-26), D (27-35), L (36-44), B (45-53)
SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
FACE_START = {'U': 0, 'R': 9, 'F': 18, 'D': 27, 'L': 36, 'B': 45}
FACE_ROT_CW = [6, 3, 0, 7, 4, 1, 8, 5, 2]

# EDGE_CYCLES defines the 12 stickers on the 4 adjacent faces that move when a face is turned.
# Each list contains 4 groups of 3 stickers in Clockwise order.
EDGE_CYCLES = {
    'U': [45, 46, 47, 9, 10, 11, 18, 19, 20, 36, 37, 38],
    'D': [24, 25, 26, 15, 16, 17, 51, 52, 53, 42, 43, 44],
    'L': [0, 3, 6, 18, 21, 24, 27, 30, 33, 53, 50, 47],
    'R': [8, 5, 2, 45, 48, 51, 35, 32, 29, 26, 23, 20],
    'F': [6, 7, 8, 9, 12, 15, 29, 28, 27, 44, 41, 38],
    'B': [2, 1, 0, 36, 39, 42, 33, 34, 35, 17, 14, 11],
}

# Corners: (Primary U/D sticker, face2, face3)
CORNERS = [
    (8, 20, 9),   # 0: UFR
    (6, 38, 18),  # 1: UFL
    (0, 47, 36),  # 2: UBL
    (2, 11, 45),  # 3: UBR
    (29, 15, 26), # 4: DFR
    (27, 24, 44), # 5: DFL
    (33, 42, 53), # 6: DBL
    (35, 51, 17)  # 7: DBR
]

# Edges: (Primary U/D or F/B sticker, face2)
EDGES = [
    (5, 10),  # 8: UR
    (7, 19),  # 9: UF
    (3, 37),  # 10: UL
    (1, 46),  # 11: UB
    (32, 16), # 12: DR
    (28, 25), # 13: DF
    (30, 43), # 14: DL
    (34, 52), # 15: DB
    (23, 12), # 16: FR
    (21, 41), # 17: FL
    (50, 39), # 18: BL
    (48, 14)  # 19: BR
]

def get_piece_map():
    pm = {}
    for i, indices in enumerate(CORNERS + EDGES):
        colors = tuple(sorted([SOLVED_STATE[idx] for idx in indices]))
        pm[colors] = i
    return pm

PIECE_ID_MAP = get_piece_map()

def get_piece_encoding(state: str) -> Tuple[np.ndarray, np.ndarray]:
    pieces = np.zeros(20, dtype=np.int32)
    orientations = np.zeros(20, dtype=np.int32)
    all_slots = CORNERS + EDGES
    
    for i, indices in enumerate(all_slots):
        current_stickers = [state[idx] for idx in indices]
        sorted_colors = tuple(sorted(current_stickers))
        
        if sorted_colors not in PIECE_ID_MAP:
            raise ValueError(f"Impossible piece colors {sorted_colors} at slot {i}")
            
        pieces[i] = PIECE_ID_MAP[sorted_colors]
        primary = {'U', 'D'} if i < 16 else {'F', 'B'}
        for orient, sticker in enumerate(current_stickers):
            if sticker in primary:
                orientations[i] = orient
                break
    return pieces, orientations


def get_facelet_from_pieces(pieces: np.ndarray, orients: np.ndarray) -> str:
    state = list(SOLVED_STATE)
    
    for piece_id in range(20):
        faces = CORNERS + EDGES
        face_indices = faces[piece_id]
        
        color_idx = pieces[piece_id]
        orient = orients[piece_id]
        
        if piece_id < 8:
            corner_colors = ['U', 'R', 'F', 'D', 'B', 'L']
            col_idx = color_idx % 6
            base_color = corner_colors[col_idx]
            if color_idx >= 6:
                base_color = 'Y' if base_color in ['U', 'D'] else ('O' if base_color in ['R', 'L'] else 'G')
        else:
            edge_colors = ['U', 'R', 'F', 'D', 'B', 'L']
            col_idx = (color_idx - 8) % 6
            base_color = edge_colors[col_idx] if col_idx < 4 else ('O' if col_idx == 4 else 'G')
        
        for i, face_idx in enumerate(face_indices):
            state[face_idx] = base_color
    
    return ''.join(state)

def rotate_face(state: str, face: str, times: int = 1) -> str:
    start = FACE_START[face]
    state_list = list(state)
    face_stickers = [state_list[start + i] for i in range(9)]
    for _ in range(times % 4):
        new_face = [face_stickers[FACE_ROT_CW[i]] for i in range(9)]
        face_stickers = new_face
    for i in range(9):
        state_list[start + i] = face_stickers[i]
    return ''.join(state_list)

def cycle_edges(state: str, face: str, times: int = 1) -> str:
    cycle = EDGE_CYCLES[face]
    state_list = list(state)
    for _ in range(times % 4):
        temp = [state_list[cycle[i]] for i in range(3)]
        # Shift 4 groups of 3
        for i in range(3):
            state_list[cycle[i]] = state_list[cycle[9+i]]
            state_list[cycle[9+i]] = state_list[cycle[6+i]]
            state_list[cycle[6+i]] = state_list[cycle[3+i]]
            state_list[cycle[3+i]] = temp[i]
    return ''.join(state_list)

def apply_move(state: str, move: str) -> str:
    face = move[0]
    times = 1
    if len(move) > 1:
        if move[1] == "'": times = 3
        elif move[1] == '2': times = 2
    state = rotate_face(state, face, times)
    state = cycle_edges(state, face, times)
    return state

def parse_moves(moves_str: str) -> List[str]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('moves', type=str)
    parser.add_argument('-p', '--piece', action='store_true')
    parser.add_argument('-s', '--save', type=str)
    args = parser.parse_args()
    
    try:
        state = SOLVED_STATE
        for m in parse_moves(args.moves):
            state = apply_move(state, m)
        
        if args.piece or args.save:
            p, o = get_piece_encoding(state)
            if args.save:
                np.savez(args.save, pieces=p, orientations=o)
                print(f"Saved to {args.save}")
            else:
                print(f"Pieces: {p}\nOrients:{o}")
        else:
            print(state)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
