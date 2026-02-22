import unittest
import numpy as np

from cube_encoding import get_piece_encoding_from_moves


class TestCubeConverter(unittest.TestCase):

    def verify_state(self, test_name: str, moves_str: str, expected_pieces: np.ndarray, expected_orients: np.ndarray):
        """Helper to print details and assert the actual state matches expectations."""
        
        # 1. Run the black box function
        actual_pieces, actual_orients = get_piece_encoding_from_moves(moves_str)
        
        # 2. Print exactly what it did, what it expected, and what it was
        print(f"\n{'='*65}")
        print(f"TEST:  {test_name}")
        print(f"MOVES: '{moves_str}'")
        print("-" * 27 + " EXPECTED " + "-" * 28)
        print(f"Pieces:  {expected_pieces}")
        print(f"Orients: {expected_orients}")
        print("-" * 28 + " ACTUAL " + "-" * 29)
        print(f"Pieces:  {actual_pieces}")
        print(f"Orients: {actual_orients}")
        print(f"{'='*65}")

        # 3. Assert they match
        np.testing.assert_array_equal(
            actual_pieces, expected_pieces, 
            f"[{test_name}] Pieces mismatch after moves!"
        )
        np.testing.assert_array_equal(
            actual_orients, expected_orients, 
            f"[{test_name}] Orientations mismatch after moves!"
        )

    def assert_solved_state(self, test_name: str, moves_str: str):
        """Helper for tests that should return to a perfectly solved cube."""
        expected_pieces = np.arange(20, dtype=np.int32)
        expected_orients = np.zeros(20, dtype=np.int32)
        self.verify_state(test_name, moves_str, expected_pieces, expected_orients)

    # ==========================================
    # 1. BASIC MOVES & INVERSES
    # ==========================================

    def test_no_moves(self):
        """Zero moves should equal a solved state."""
        self.assert_solved_state("No Moves (Identity)", "")

    def test_face_order_4(self):
        """Turning any face 4 times is the identity."""
        for face in "URFDLB":
            self.assert_solved_state(f"Face {face} x4", f"{face} {face} {face} {face}")

    def test_face_double_turns(self):
        """Double turns done twice is the identity."""
        for face in "URFDLB":
            self.assert_solved_state(f"Face {face}2 x2", f"{face}2 {face}2")

    def test_move_and_inverse(self):
        """A move followed by its prime is the identity."""
        for face in "URFDLB":
            self.assert_solved_state(f"Move and Inverse ({face} {face}')", f"{face} {face}'")

    # ==========================================
    # 2. ALGORITHM CYCLES (Order N)
    # ==========================================

    def test_sexy_move_cycle(self):
        """(R U R' U') repeated 6 times returns the cube to a solved state."""
        sexy_move = "R U R' U' "
        self.assert_solved_state("Sexy Move x6", sexy_move * 6)

    def test_reverse_sexy_move_cycle(self):
        """(U R U' R') repeated 6 times is also solved."""
        rev_sexy = "U R U' R' "
        self.assert_solved_state("Reverse Sexy x6", rev_sexy * 6)

    def test_sune_cycle(self):
        """Sune repeated 6 times is order 6."""
        sune = "R U R' U R U2 R' "
        self.assert_solved_state("Sune x6", sune * 6)

    def test_t_perm_order_2(self):
        """The T-Perm swaps 2 corners and 2 edges. Doing it twice solves it."""
        t_perm = "R U R' U' R' F R2 U' R' U' R U R' F' "
        self.assert_solved_state("T-Perm x2", t_perm * 2)

    def test_y_perm_order_2(self):
        """The Y-Perm swaps 2 corners and 2 edges. Order 2."""
        y_perm = "F R U' R' U' R U R' F' R U R' U' R' F R F' "
        self.assert_solved_state("Y-Perm x2", y_perm * 2)

    def test_u_perm_order_3(self):
        """The Ua-Perm cycles 3 edges. Doing it 3 times solves it."""
        u_perm = "R U' R U R U R U' R' U' R2 "
        self.assert_solved_state("U-Perm x3", u_perm * 3)

    def test_a_perm_order_3(self):
        """The Aa-Perm cycles 3 corners. Doing it 3 times solves it."""
        # Rotationless A-Perm to avoid 'x' or 'y' slice moves
        a_perm = "R' F R' B2 R F' R' B2 R2 "
        self.assert_solved_state("A-Perm x3", a_perm * 3)

    def test_checkerboard_order_2(self):
        """Checkerboard pattern applied twice returns the cube to solved."""
        checkerboard = "U2 D2 F2 B2 L2 R2 "
        self.assert_solved_state("Checkerboard x2", checkerboard * 2)

    # ==========================================
    # 3. ALGORITHMS + EXACT INVERSES
    # ==========================================

    def test_sune_and_antisune(self):
        """Doing a Sune followed by an Anti-Sune solves the cube."""
        sune = "R U R' U R U2 R' "
        anti_sune = "R U2 R' U' R U' R' "
        self.assert_solved_state("Sune + Anti-Sune", sune + anti_sune)

    def test_scramble_and_inverse(self):
        """A heavy 20-move scramble followed by its exact inverse."""
        scramble = "R2 F2 U' L2 D B2 R2 U' R2 F2 D L' D' R B U' F' L2 B' U' "
        # The inverse reverses the order and flips the direction of every move
        inverse = "U B L2 F U B' R' D L D' F2 R2 U R2 B2 D' L2 U F2 R2 "
        
        self.assert_solved_state("20-Move Scramble + Exact Inverse", scramble + inverse)

    # ==========================================
    # 4. INVARIANT STATES (Pieces locked, orientations changed)
    # ==========================================

    def test_superflip(self):
        """The Superflip leaves every piece in its home slot but flips all 12 edges."""
        superflip = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
        
        expected_pieces = np.arange(20, dtype=np.int32)
        
        # Corners (0-7) stay 0, Edges (8-19) flip to 1
        expected_orients = np.zeros(20, dtype=np.int32)
        expected_orients[8:] = 1 
        
        self.verify_state("Superflip State", superflip, expected_pieces, expected_orients)


if __name__ == "__main__":
    # Prevent unittest from hiding standard print statements
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

