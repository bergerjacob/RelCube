## RelCube Architecture

### 1. Encoding
**Target:** Physical state extraction into discrete integers.

* **Input:** Raw state of the 3x3 Rubik's Cube.
* **Operation:** Extract the piece ID, slot ID, and orientation for each of the 20 physical positions (8 corners, 12 edges).
* **Outputs (3 integer tensors):**
    * **Slot IDs:** Values 0–19 (Mapping corners 0–7, edges 8–19).
        * **Shape:** `(Batch_Size, 20)`
    * **Piece IDs:** Values 0–19 (Same mappings as above for the “home” slot, so a correctly solved piece would have slot ID = piece ID).
        * **Shape:** `(Batch_Size, 20)`
    * **Orientations:** Values 0–2.
        * **Corners (0–2):** 0 if U/D sticker is on U/D face, 1 if clockwise, 2 if counter-clockwise.
        * **Edges (0–1):** ZZ-style Edge Orientation (EO). 0 if oriented (U/D sticker on U/D/F/B faces, or L/R sticker on L/R edges), 1 if misoriented.
        * **Shape:** `(Batch_Size, 20)`

---

### 2. Embedding
**Target:** Conversion of discrete integers into a fused, continuous vector space.

* **Input:** The `(Batch_Size, 20, 3)` integer tensor from the Encoding step.
* **Operation 1: Deterministic Slicing** (Hardware-friendly split)
    * **Corner Slice:** Extract indices 0–7 along the sequence dimension.
        * **Shape:** `(Batch_Size, 8, 3)`
    * **Edge Slice:** Extract indices 8–19 along the sequence dimension.
        * **Shape:** `(Batch_Size, 12, 3)`
* **Operation 2: Parallel Lookup & Feature Concatenation**
    * **Corner Path:** Pass through `Corner_Slot(8, 42)`, `Corner_Piece(8, 42)`, and `Corner_Orient(3, 44)` lookup tables. Concatenate results (42 + 42 + 44 = 128).
        * **Shape:** `(Batch_Size, 8, 128)`
    * **Edge Path:** Pass through `Edge_Slot(12, 42)`, `Edge_Piece(12, 42)`, and `Edge_Orient(2, 44)` lookup tables. Concatenate results (42 + 42 + 44 = 128).
        * **Shape:** `(Batch_Size, 12, 128)`
* **Operation 3: Sequence Recombination**
    * Concatenate Corner and Edge Paths back together (8 + 12 = 20 tokens).
        * **Shape:** `(Batch_Size, 20, 128)`
* **Layer 1: Linear Projection**
    * `Linear(128 -> 256)` to map to the Transformer's hidden dimension.
    * **Shape:** `(Batch_Size, 20, 256)`
    * > **Architecture Note: Learned Positional Fusion**
      > In standard Transformer architectures (*Attention Is All You Need*), positional encodings are often added to token embeddings. In RelCube, the `Slot ID` functions as our positional encoding, while the `Piece ID` and `Orientation` act as the semantic token. Rather than adding them, we concatenate these embeddings and use this `Linear(128 -> 256)` layer to mix them.
* **Output:** Continuous, context-free sequence of 20 piece tokens.

---

### 3. Transformer Blocks
**Target:** Long-range spatial dependencies and permutation routing.

* **Input:** The `(Batch_Size, 20, 256)` tensor from the Embedding layer.
* **Architecture:** 8 sequential Transformer Encoder Layers.
* **For Each Block (Repeated 8 times):**
    * **Sub-layer 1 (Self-Attention):** Multi-Head Attention (`d_model=256`, `nhead=8`).
        * **Operation:** Residual Add + Layer Normalization.
    * **Sub-layer 2 (Feedforward):** `Linear(256 -> 1024)` → GELU → `Linear(1024 -> 256)`.
        * **Operation:** Residual Add + Layer Normalization.
* **Output:** Context-aware sequence of 20 piece tokens.

---

### 4. Output Heads

#### Value Head (Regression)
**Target:** Puzzle entropy / distance to solved.

* **Operation:** Global Average Pooling (averages the 20 tokens).
    * **Shape:** `(Batch_Size, 256)`
* **Layer 1:** `Linear(256 -> 128)` + ReLU.
* **Layer 2:** `Linear(128 -> 1)`.
* **Output:** Continuous scalar (Expected moves to solved).

#### Policy Head (Classification)
**Target:** Best move selection based on spatial relationships.

* **Operation:** Flatten (20 * 256 = 5120).
    * **Shape:** `(Batch_Size, 5120)`
* **Layer 1:** `Linear(5120 -> 256)` + ReLU.
* **Layer 2:** `Linear(256 -> 12)` (Standard move set).
* **Output:** Logits (Probability distribution of next best move).
