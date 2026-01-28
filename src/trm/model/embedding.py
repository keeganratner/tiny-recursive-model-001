"""Grid embedding for ARC-AGI color grids."""
import torch
import torch.nn as nn


class GridEmbedding(nn.Module):
    """
    Embed ARC-AGI color grids into continuous vector space.

    Maps grid of shape (B, H, W) with color values 0-9 and padding value -1
    to embedded representation (B, H, W, hidden_dim).

    Uses 11 embedding vectors:
    - Indices 0-9: ARC colors (black, blue, red, green, yellow, gray, pink, orange, purple, brown)
    - Index 10: Padding token for variable-size grids

    Args:
        hidden_dim: Embedding dimension (typically 512)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 11 embeddings: 0-9 for colors, 10 for padding
        # No bias per paper specification
        self.embedding = nn.Embedding(
            num_embeddings=11,
            embedding_dim=hidden_dim,
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Embed color grid.

        Args:
            grid: Tensor of shape (B, H, W) with values 0-9 (colors) or -1 (padding)

        Returns:
            Tensor of shape (B, H, W, hidden_dim)
        """
        # Map PAD_VALUE (-1) to index 10
        # Valid colors 0-9 remain unchanged
        grid_mapped = torch.where(grid == -1, torch.tensor(10, dtype=grid.dtype, device=grid.device), grid)

        # Embed: (B, H, W) -> (B, H, W, hidden_dim)
        embedded = self.embedding(grid_mapped)

        return embedded
