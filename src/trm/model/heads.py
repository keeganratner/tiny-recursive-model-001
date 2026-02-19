"""Output and halting prediction heads for TRM network."""
import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Output head that predicts color logits for each grid cell.

    Maps hidden states to color class logits for ARC-AGI prediction.
    Produces unnormalized logits over 10 color classes.

    Args:
        hidden_dim: Model dimension (default 512)
        num_colors: Number of ARC color classes (default 10)
    """

    def __init__(self, hidden_dim: int = 512, num_colors: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors

        # Linear projection to color logits (no bias per paper spec)
        self.proj = nn.Linear(hidden_dim, num_colors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict color logits from hidden states.

        Args:
            x: Hidden states of shape (B, seq_len, hidden_dim)

        Returns:
            Logits of shape (B, seq_len, num_colors)
        """
        return self.proj(x)


class HaltingHead(nn.Module):
    """
    Halting head that predicts confidence in current answer (Q-head in paper).

    Produces single confidence score per batch item for early stopping.
    Uses global average pooling over sequence followed by sigmoid activation.

    Args:
        hidden_dim: Model dimension (default 512)
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear projection to scalar confidence (no bias per paper spec)
        self.proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Predict halting confidence from hidden states.

        Args:
            x:    Hidden states of shape (B, seq_len, hidden_dim)
            mask: (B, seq_len) bool, True=valid. If None, average all positions.

        Returns:
            Confidence scores of shape (B,) in range [0, 1]
        """
        if mask is not None:
            # Masked average: only valid tokens contribute to pooling
            mask_f = mask.unsqueeze(-1).float()          # (B, seq_len, 1)
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            # Global average pooling over sequence dimension
            pooled = x.mean(dim=1)  # (B, hidden_dim)

        # Project to scalar and apply sigmoid
        confidence = torch.sigmoid(self.proj(pooled))  # (B, 1)

        return confidence.squeeze(-1)  # (B,)
