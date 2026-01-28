"""Multi-head self-attention with rotary position embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryEmbedding


class Attention(nn.Module):
    """
    Multi-head self-attention with rotary position embeddings.

    Implements scaled dot-product attention with:
    - Separate Q, K, V projections (no bias)
    - Rotary position embeddings applied to Q and K
    - Multi-head attention with efficient F.scaled_dot_product_attention
    - Output projection (no bias)

    Args:
        hidden_dim: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V projections - no bias per paper
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection - no bias per paper
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Rotary position embedding
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim)
            mask: Optional attention mask of shape (B, seq_len) or (B, seq_len, seq_len)
                  where True/1 indicates valid positions
            positions: Optional position indices of shape (B, seq_len)
                      If None, uses simple sequential positions

        Returns:
            Output tensor of shape (B, seq_len, hidden_dim)
        """
        B, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, seq_len, hidden_dim)
        k = self.k_proj(x)  # (B, seq_len, hidden_dim)
        v = self.v_proj(x)  # (B, seq_len, hidden_dim)

        # Reshape to multi-head: (B, seq_len, hidden_dim) -> (B, seq_len, num_heads, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim)
        k = k.view(B, seq_len, self.num_heads, self.head_dim)
        v = v.view(B, seq_len, self.num_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        if positions is None:
            # Use sequential positions if not provided
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)

        q = self.rope(q, positions)  # (B, seq_len, num_heads, head_dim)
        k = self.rope(k, positions)  # (B, seq_len, num_heads, head_dim)

        # Transpose to (B, num_heads, seq_len, head_dim) for attention
        q = q.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)

        # Prepare attention mask for scaled_dot_product_attention
        # F.scaled_dot_product_attention expects mask where True means attend
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # (B, seq_len) -> (B, 1, 1, seq_len) for broadcasting
                # This masks out invalid keys
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (B, seq_len, seq_len) -> (B, 1, seq_len, seq_len)
                attn_mask = mask.unsqueeze(1)

        # Scaled dot-product attention
        # Output: (B, num_heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Transpose back and reshape: (B, num_heads, seq_len, head_dim) -> (B, seq_len, hidden_dim)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, seq_len, self.hidden_dim)

        # Output projection
        out = self.o_proj(out)

        return out
