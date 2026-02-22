import numpy as np
import argparse
import sys
from cube_encoding import get_piece_encoding_from_moves

def run_encoding(output_path):
    # Define our 6 states
    scramble_list = [
        "R U R' U'",                             # 1. Easy
        "D L2 B F' R' U2",                       # 2. Easy
        "F2 U' L R' D2 B2 L' R U'",               # 3. Medium scramble
        "L B R U' D' L' B' D2 F R2",             # 4. Hard scramble
        "R2 L2 U2 D2 F2 B2",                     # 5. Checkerboard pattern
        "U U U U"                                 # 6. solved state 
    ]

    all_pieces = []
    all_orients = []

    print(f"Encoding {len(scramble_list)} states...")

    for i, moves_str in enumerate(scramble_list):
        try:
            pieces, orients = get_piece_encoding_from_moves(moves_str)
            all_pieces.append(pieces)
            all_orients.append(orients)
            print(f"State {i+1} done: {moves_str[:20]}...")
        except Exception as e:
            print(f"Error processing state {i+1}: {e}")
            sys.exit(1)

    # Save logic
    if not output_path.endswith('.npz'):
        output_path += '.npz'

    np.savez(output_path, 
             pieces=np.array(all_pieces), 
             orientations=np.array(all_orients))

    print(f"\nSuccess! '{output_path}' created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Rubik's Cube states.")
    
    # Adding the --output flag and making it mandatory
    parser.add_argument("--output", 
                        required=True, 
                        help="The path/name for the output .npz file")
    
    args = parser.parse_args()
    
    # Run the program
    run_encoding(args.output)
