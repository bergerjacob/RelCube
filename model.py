import torch
import torch.nn as nn


class CubeEncoder(nn.Module):
    """Encoding layer that extracts piece ID and orientation."""

    def __init__(self):
        super().__init__()

    def forward(self, raw_state):
        """
        Extract piece ID and orientation from raw cube state.

        Args:
            raw_state: (Batch_Size, 2, 20) - [0] = piece IDs, [1] = orientations

        Returns:
            piece_ids: (Batch_Size, 20) - Values 0-19
            orientations: (Batch_Size, 20) - Values 0-2 (corners) or 0-1 (edges)
        """
        piece_ids = raw_state[:, 0, :]
        orientations = raw_state[:, 1, :]
        return piece_ids, orientations


class EmbeddingLayer(nn.Module):
    """Embedding layer with deterministic slicing and parallel lookup tables."""

    def __init__(self, d_model=512):
        super().__init__()

        self.corner_slot_emb = nn.Embedding(8, 42)
        self.corner_piece_emb = nn.Embedding(8, 42)
        self.corner_orient_emb = nn.Embedding(3, 44)

        self.edge_slot_emb = nn.Embedding(12, 42)
        self.edge_piece_emb = nn.Embedding(12, 42)
        self.edge_orient_emb = nn.Embedding(2, 44)

        self.proj = nn.Linear(128, d_model)

    def forward(self, piece_ids, orientations):
        """
        Args:
            piece_ids: (Batch_Size, 20) - Piece IDs 0-19
            orientations: (Batch_Size, 20) - Orientations

        Returns:
            (Batch_Size, 20, 256) - Embedded tokens
        """
        batch_size = piece_ids.size(0)
        device = piece_ids.device

        corner_slice = 8
        edge_slice = 12

        corner_slot_ids = torch.arange(corner_slice, device=device).expand(
            batch_size, corner_slice
        )
        edge_slot_ids = torch.arange(edge_slice, device=device).expand(
            batch_size, edge_slice
        )

        corner_piece_ids = piece_ids[:, :corner_slice]
        corner_orientations = orientations[:, :corner_slice]

        edge_piece_ids = piece_ids[:, corner_slice:] - corner_slice
        edge_orientations = orientations[:, corner_slice:]

        corner_emb_slot = self.corner_slot_emb(corner_slot_ids)
        corner_emb_piece = self.corner_piece_emb(corner_piece_ids)
        corner_emb_orient = self.corner_orient_emb(corner_orientations)
        corner_embedded = torch.cat(
            [corner_emb_slot, corner_emb_piece, corner_emb_orient], dim=-1
        )

        edge_emb_slot = self.edge_slot_emb(edge_slot_ids)
        edge_emb_piece = self.edge_piece_emb(edge_piece_ids)
        edge_emb_orient = self.edge_orient_emb(edge_orientations)
        edge_embedded = torch.cat(
            [edge_emb_slot, edge_emb_piece, edge_emb_orient], dim=-1
        )

        embedded = torch.cat([corner_embedded, edge_embedded], dim=1)
        embedded = self.proj(embedded)

        return embedded


class TransformerBlock(nn.Module):
    """Single Transformer Encoder Block."""

    def __init__(self, d_model=512, nhead=16, dim_feedforward=2048):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class TransformerBlocks(nn.Module):
    """Multiple Transformer Encoder Blocks."""

    def __init__(self, num_layers=12, d_model=512, nhead=16, dim_feedforward=2048):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ValueHead(nn.Module):
    """Value head for regression (distance to solved).

    Takes the CLS token which has aggregated global context from all pieces.
    """

    def __init__(self, d_model=512):
        super().__init__()

        self.fc1 = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            x: (Batch_Size, d_model) - CLS token representation
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PolicyHead(nn.Module):
    """Policy head for move selection (classification).

    Uses the 20 piece tokens (excluding CLS) to predict move probabilities.
    """

    def __init__(self, d_model=512, num_tokens=20, num_moves=12):
        super().__init__()

        self.fc1 = nn.Linear(d_model * num_tokens, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_moves)

    def forward(self, x):
        """
        Args:
            x: (Batch_Size, 20, d_model) - Token representations (excluding CLS)
        """
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class RelCube(nn.Module):
    """Main RelCube model for Rubik's Cube state evaluation."""

    def __init__(self, encoder=None, d_model=512):
        super().__init__()

        self.encoder = encoder or CubeEncoder()
        self.embedding = EmbeddingLayer(d_model=512)
        self.transformer = TransformerBlocks(d_model=512)
        self.value_head = ValueHead(d_model=512)
        # self.policy_head = PolicyHead(d_model)

        # CLS token for global state aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, raw_state):
        """
        Forward pass through the entire model.

        Args:
            raw_state: Raw representation of 3x3 Rubik's Cube

        Returns:
            value: (Batch_Size, 1) - Expected moves to solved
            policy: (Batch_Size, 12) - Move probabilities
        """
        piece_ids, orientations = self.encoder(raw_state)
        embedded = self.embedding(piece_ids, orientations)

        # Prepend CLS token to sequence
        batch_size = embedded.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        # Process through transformer
        transformed = self.transformer(embedded)

        # Extract CLS token (first token) for value prediction
        cls_output = transformed[:, 0]
        value = self.value_head(cls_output)

        # # Policy uses the remaining tokens (excluding CLS)
        # policy_tokens = transformed[:, 1:]
        # policy = self.policy_head(policy_tokens)
        policy = None

        return value, policy
