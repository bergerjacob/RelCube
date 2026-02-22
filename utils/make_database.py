# run with : python make_database.py

import numpy as np
from cube_converter import apply_move, get_piece_encoding, SOLVED_STATE, parse_moves

# Define our 6 states
scramble_list = [
    "R U R' U'",                             # 1. Easy
    "D L2 B F' R' U2",                       # 2. Easy
    "F2 U' L R' D2 B2 L' R U'",               # 3. Medium scramble
    "L B R U' D' L' B' D2 F R2",             # 4. Hard scramble
    "R2 L2 U2 D2 F2 B2",                     # 5. Checkerboard pattern
    "U U U U"                                # 6. solved state 
]

all_slots = []
all_pieces = []
all_orients = []

print(f"Encoding {len(scramble_list)} states...")

for i, moves_str in enumerate(scramble_list):
    # Start from solved and apply moves
    state = SOLVED_STATE
    for m in parse_moves(moves_str):
        state = apply_move(state, m)
    
    # Encode to piece-based tensors
    s, p, o = get_piece_encoding(state)
    
    all_slots.append(s)
    all_pieces.append(p)
    all_orients.append(o)
    
    print(f"State {i+1} done: {moves_str[:20]}...")

# Save all 6 states into one file
np.savez('cube_dataset_6.npz', 
         slots=np.array(all_slots), 
         pieces=np.array(all_pieces), 
         orientations=np.array(all_orients))

print("\nSuccess! 'cube_dataset_6.npz' created.")