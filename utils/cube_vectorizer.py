import torch
import torch.nn as nn
from typing import List

# when running locally for testing purposes it uses cpu but when ran on the HPC it uses cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CubeBatch:
    def __init__(self, batch_size: int, device: torch.device = device):
        self.batch_size = batch_size
        self.device = device
        
        self.pieces = torch.arange(20, device=device).repeat(batch_size, 1)
        self.orients = torch.zeros((batch_size, 20), dtype=torch.long, device=device)
        
        self._init_move_tensors()
        self._init_master_tensors()

    def _init_move_tensors(self):
        # Format: face -> [perm_indices], [orientation_deltas]
        base_moves = {
            'U': ([3, 0, 1, 2, 4, 5, 6, 7, 11, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'D': ([0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11, 13, 14, 15, 12, 16, 17, 18, 19], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'L': ([0, 5, 1, 3, 4, 6, 2, 7, 8, 9, 18, 11, 12, 13, 17, 15, 16, 10, 14, 19], 
                  [0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'R': ([4, 1, 2, 0, 7, 5, 6, 3, 16, 9, 10, 11, 19, 13, 14, 15, 12, 17, 18, 8], 
                  [1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'F': ([1, 4, 2, 3, 5, 0, 6, 7, 8, 17, 10, 11, 12, 16, 14, 15, 9, 13, 18, 19], 
                  [1, 2, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
            'B': ([0, 1, 3, 7, 4, 5, 2, 6, 8, 9, 10, 19, 12, 13, 14, 18, 16, 17, 11, 15], 
                  [0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1])
        }
        
        self.perms = {}
        self.deltas = {}
        
        for face, (p, d) in base_moves.items():
            self.perms[face] = torch.tensor(p, device=self.device)
            self.deltas[face] = torch.tensor(d, device=self.device)

    def apply_moves(self, move_str: str):
        """Apply a sequence of moves to the entire batch."""
        moves = self._parse_moves(move_str)
        for m in moves:
            face = m[0]
            times = 1
            if len(m) > 1:
                if m[1] == "'": times = 3
                elif m[1] == '2': times = 2
            
            for _ in range(times):
                self.pieces = torch.gather(self.pieces, 1, self.perms[face].expand(self.batch_size, -1))
                
                permuted_orients = torch.gather(self.orients, 1, self.perms[face].expand(self.batch_size, -1))
                new_orients = permuted_orients + self.deltas[face]
                
                new_orients[:, :8] %= 3
                new_orients[:, 8:] %= 2
                self.orients = new_orients

    def _parse_moves(self, moves_str: str) -> List[str]:
        valid = 'URFDLB'
        moves, i = [], 0
        while i < len(moves_str):
            if moves_str[i] in valid:
                m = moves_str[i]
                if i + 1 < len(moves_str) and moves_str[i+1] in "'2":
                    m += moves_str[i+1]; i += 1
                moves.append(m)
            i += 1
        return moves

    def get_encoding(self):
        """
        Returns the concatenated (Pieces, Orients) for the Transformer
        """
        return torch.stack([self.pieces, self.orients], dim=-1)
    
    def _init_master_tensors(self):
        self.master_perms = torch.stack([self.perms[f] for f in 'URFDLB'])
        self.master_deltas = torch.stack([self.deltas[f] for f in 'URFDLB'])

    def scramble(self, depth: int):
        """
        Applies 'depth' random moves to EACH cube in the batch independently
        """
        for _ in range(depth):
            move_indices = torch.randint(0, 6, (self.batch_size,), device=self.device)
            
            current_perms = self.master_perms[move_indices]
            current_deltas = self.master_deltas[move_indices]
            
            # 3. Apply to pieces
            self.pieces = torch.gather(self.pieces, 1, current_perms)
            
            # 4. Apply to orientations
            permuted_orients = torch.gather(self.orients, 1, current_perms)
            new_orients = permuted_orients + current_deltas
            
            new_orients[:, :8] %= 3
            new_orients[:, 8:] %= 2
            self.orients = new_orients

# Example Usage:
if __name__ == "__main__":
    # Simulate 10,000 cubes at once
    cubes = CubeBatch(batch_size=10000, device=device) 
    cubes.apply_moves("R U R' U'")
    
    state = cubes.get_encoding()
    print(f"Batch Shape: {state.shape}")
    print(f"Cube 0 Pieces: {state[0, :, 0]}")

    cubes = CubeBatch(batch_size=10000)
    cubes.scramble(depth=20) # 20 random moves per cube
    state = cubes.get_encoding() 

    print(f"Batch Shape: {state.shape}") 
    print(f"Cube 0 Pieces: {state[0, :, 0]}")
# 'state' is now a massive training set of 10,000 scrambled cubes