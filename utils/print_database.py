# make sure you made the database first with: python make_database.py
# run with: python print_database.py

import numpy as np
d = np.load('cube_dataset_6.npz')

print(f"{'State':<10} | {'Status':<15} | {'Pieces'}")
print("-" * 50)
for i in range(6):
    pieces = d['pieces'][i]
    is_solved = np.array_equal(pieces, np.arange(20))
    status = "Solved" if is_solved else "Scrambled"
    print(f"Index {i:<5} | {status:<15} | {pieces}")