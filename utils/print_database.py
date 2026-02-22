import numpy as np
import argparse
import sys

def print_database(input_path):
    # Ensure the filename ends with .npz
    if not input_path.endswith('.npz'):
        input_path += '.npz'

    try:
        d = np.load(input_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)

    # Header for the table
    print(f"\nDatabase: {input_path}")
    print("Solved slot reference: " + " ".join(f"{i:<3}" for i in range(20)))
    print("-" * 130) # Adjusted width for readability

    # Get the number of states in the file
    num_states = d['pieces'].shape[0]

    print(f"{'Idx':<4} | {'Status':<10} | {'Pieces':<60} | {'Orients'}")
    print("-" * 130)

    for i in range(num_states):
        pieces = d['pieces'][i]
        orients = d['orientations'][i]
        
        # Check if the pieces are in their original positions (0-19)
        is_solved = np.array_equal(pieces, np.arange(20))
        status = "Solved" if is_solved else "Scrambled"
        
        pieces_str = " ".join(f"{int(x):<2}" for x in pieces)
        orients_str = " ".join(f"{int(x):<2}" for x in orients)
        
        print(f"{i:<4} | {status:<10} | {pieces_str:<60} | {orients_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and display a Rubik's Cube .npz database.")
    
    # Required input flag
    parser.add_argument("--input", 
                        required=True, 
                        help="The path to the .npz file to print")
    
    args = parser.parse_args()
    
    print_database(args.input)
