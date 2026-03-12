"""Convert data_0.pkl (DeepCubeA format) to data_0.npz in cube_encoding format.

Outputs pieces (N, 20) and orientations (N, 20) compatible with cube_encoding.py.
Note: solutions use DeepCubeA move conventions (different face orientation from
cube_encoding.py), so solution_lengths is the reliable label for training.
"""

import pickle
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from cube_facelet import get_piece_encoding


# Empirically determined position mapping from DeepCubeA (54-facelet) to cube_facelet.py
# DCA_TO_CF[i] = the cube_facelet position corresponding to DCA position i
# DCA face order: U=0-8, D=9-17, L=18-26, R=27-35, B=36-44, F=45-53
# CF face order:  U=0-8, R=9-17, F=18-26, D=27-35, L=36-44, B=45-53
# Within each face, DCA uses a transposed layout relative to CF.
DCA_TO_CF = np.array([
     6,  3,  0,  7,  4,  1,  8,  5,  2,   # DCA U → CF U (transposed)
    33, 30, 27, 34, 31, 28, 35, 32, 29,   # DCA D → CF D
    42, 39, 36, 43, 40, 37, 44, 41, 38,   # DCA L → CF L
    15, 12,  9, 16, 13, 10, 17, 14, 11,   # DCA R → CF R
    51, 48, 45, 52, 49, 46, 53, 50, 47,   # DCA B → CF B
    24, 21, 18, 25, 22, 19, 26, 23, 20,   # DCA F → CF F
])

# DCA facelet value → face color letter
DCA_FACELET_TO_FACE = "UDLRBF"


class Cube3State:
    """Stub for environments.cube3.Cube3State so pickle can load without the module."""
    pass


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "environments.cube3" and name == "Cube3State":
            return Cube3State
        return super().find_class(module, name)


def colors_to_facelet_string(colors):
    """Convert a 54-int DeepCubeA color array to a facelet string in cube_facelet.py order."""
    # Build a CF-ordered facelet string:
    # 1. For each CF position, find the corresponding DCA position (invert DCA_TO_CF)
    # 2. Look up the DCA facelet value there and convert to face letter
    result = [''] * 54
    for dca_pos in range(54):
        cf_pos = DCA_TO_CF[dca_pos]
        face_idx = colors[dca_pos] // 9  # which DCA face the facelet belongs to
        result[cf_pos] = DCA_FACELET_TO_FACE[face_idx]
    return "".join(result)


def solution_to_moves(sol):
    """Convert [[face, dir], ...] to move strings like R, R', etc."""
    moves = []
    for face, direction in sol:
        if direction == -1:
            moves.append(face + "'")
        else:
            moves.append(face)
    return moves


def convert(pkl_path, npz_path):
    with open(pkl_path, "rb") as f:
        data = StubUnpickler(f).load()

    colors_list = [s.colors for s in data["states"]]
    n = len(colors_list)

    pieces = np.zeros((n, 20), dtype=np.int32)
    orientations = np.zeros((n, 20), dtype=np.int32)

    for i, colors in enumerate(colors_list):
        facelet_str = colors_to_facelet_string(colors)
        p, o = get_piece_encoding(facelet_str)
        pieces[i] = p
        orientations[i] = o

    # Convert solutions to move strings
    sol_lengths = np.array([len(s) for s in data["solutions"]], dtype=np.int32)
    sol_strs = np.array(
        [" ".join(solution_to_moves(s)) for s in data["solutions"]], dtype=object
    )

    times = np.array(data["times"])
    num_nodes = np.array(data["num_nodes_generated"])

    np.savez(
        npz_path,
        pieces=pieces,
        orientations=orientations,
        solution_lengths=sol_lengths,
        solutions=sol_strs,
        times=times,
        num_nodes_generated=num_nodes,
        allow_pickle=True,
    )
    print(f"Saved {npz_path}")
    print(f"  pieces:            {pieces.shape} {pieces.dtype}")
    print(f"  orientations:      {orientations.shape} {orientations.dtype}")
    print(f"  solution_lengths:  {sol_lengths.shape}  range [{sol_lengths.min()}, {sol_lengths.max()}]")
    print(f"  solutions:         {sol_strs.shape} (move strings)")
    print(f"  times:             {times.shape} {times.dtype}")
    print(f"  num_nodes:         {num_nodes.shape} {num_nodes.dtype}")


if __name__ == "__main__":
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "data_0/data_0.pkl"
    npz_path = sys.argv[2] if len(sys.argv) > 2 else "data_0/data_0.npz"
    convert(pkl_path, npz_path)
