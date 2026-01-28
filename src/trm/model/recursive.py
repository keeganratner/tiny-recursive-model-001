"""Recursive refinement loop implementing the core TRM algorithm."""
import torch
import torch.nn as nn

from .network import TRMNetwork
from .embedding import GridEmbedding


class RecursiveRefinement(nn.Module):
    """
    Recursive refinement module implementing the TRM's nested iteration loop.

    This is the core innovation of the TRM architecture: achieving effective depth
    through weight-shared recursion rather than parameter count. A single 2-layer
    network is applied repeatedly in a nested loop structure:

    Outer loop (T iterations): Refines answer state y
    Inner loop (n iterations): Refines latent state z

    Total network calls: T * (n + 1) = 3 * 7 = 21 for default config

    State representation:
        x: Input grid embedding (fixed throughout recursion)
        y: Answer state logits (B, H, W, num_colors) - refined each outer iteration
        z: Latent state logits (B, H, W, num_colors) - refined each inner iteration

    State combination:
        States are combined by ADDING their embeddings:
        - Inner loop: combined = embed(x) + embed(y) + embed(z)
        - Outer loop: combined = embed(y) + embed(z)

        Note: x is raw grid values, y and z are logits requiring argmax before embedding.

    Args:
        network: TRMNetwork instance (shared weights across all iterations)
        embedding: GridEmbedding instance (for state combination)
        outer_steps: Number of outer loop iterations T (default 3)
        inner_steps: Number of inner loop iterations n (default 6)
        num_colors: Number of ARC color classes (default 10)
        halt_threshold: Confidence threshold for early stopping (default 0.9)
        enable_halting: Whether to enable adaptive halting (default True)
    """

    def __init__(
        self,
        network: TRMNetwork,
        embedding: GridEmbedding,
        outer_steps: int = 3,
        inner_steps: int = 6,
        num_colors: int = 10,
        halt_threshold: float = 0.9,
        enable_halting: bool = True,
    ):
        super().__init__()
        self.network = network
        self.embedding = embedding
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.num_colors = num_colors
        self.halt_threshold = halt_threshold
        self.enable_halting = enable_halting

    def _combine_states(
        self,
        states: list[torch.Tensor],
        is_logits: list[bool]
    ) -> torch.Tensor:
        """
        Combine multiple states by adding their embeddings.

        States can be either raw grids (values 0-9) or logits (B, H, W, num_colors).
        Logits are converted to hard predictions via argmax before embedding.

        Args:
            states: List of state tensors to combine
            is_logits: List of booleans indicating if each state is logits (True) or raw (False)

        Returns:
            Combined embedding tensor of shape (B, H, W, hidden_dim)
        """
        embeddings = []

        for state, logit_flag in zip(states, is_logits):
            if logit_flag:
                # Convert logits to hard predictions: (B, H, W, num_colors) -> (B, H, W)
                state = torch.argmax(state, dim=-1)

            # Embed state: (B, H, W) -> (B, H, W, hidden_dim)
            embedded = self.embedding(state)
            embeddings.append(embedded)

        # Add all embeddings element-wise
        combined = sum(embeddings)

        return combined

    def _forward_through_network(self, combined_embedding: torch.Tensor) -> dict:
        """
        Forward combined embedding through transformer and heads.

        Bypasses TRMNetwork's internal embedding layer since we already have
        combined embeddings from state addition.

        Args:
            combined_embedding: Embedding tensor of shape (B, H, W, hidden_dim)

        Returns:
            Dictionary with logits (B, H, W, num_colors) and halt_confidence (B,)
        """
        B, H, W, hidden_dim = combined_embedding.shape

        # Flatten spatial dimensions to sequence: (B, H*W, hidden_dim)
        x = combined_embedding.view(B, H * W, hidden_dim)

        # Apply transformer stack
        x = self.network.transformer(x)

        # Output head: (B, H*W, hidden_dim) -> (B, H*W, num_colors)
        logits = self.network.output_head(x)

        # Reshape logits back to grid: (B, H, W, num_colors)
        logits = logits.view(B, H, W, self.num_colors)

        # Halting head: (B, H*W, hidden_dim) -> (B,)
        halt_confidence = self.network.halting_head(x)

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
        }

    def forward(self, x: torch.Tensor) -> dict:
        """
        Apply recursive refinement to input grid.

        Implements nested loop structure:
            for t in range(T):  # outer loop
                z = zeros
                for i in range(n):  # inner loop
                    z = network(combine(x, y, z))
                y = network(combine(y, z))
            return y

        Args:
            x: Input grid tensor of shape (B, H, W) with values 0-9 or -1 (padding)

        Returns:
            Dictionary containing:
                - logits: Final answer state of shape (B, H, W, num_colors)
                - halt_confidence: Halting confidence from last network call (B,)
                - iterations: Total number of network forward passes (int)
                - halted_early: Boolean indicating if halting stopped early (bool)
        """
        B, H, W = x.shape

        # Initialize answer state y to zeros (B, H, W, num_colors)
        y = torch.zeros(B, H, W, self.num_colors, device=x.device, dtype=torch.float32)

        # Track total iterations
        total_iterations = 0
        halted_early = False

        # Outer loop: Refine answer state y
        for t in range(self.outer_steps):
            # Initialize/reset latent state z to zeros at start of each outer iteration
            z = torch.zeros_like(y)

            # Inner loop: Refine latent state z
            for i in range(self.inner_steps):
                # Combine x (raw), y (logits), z (logits)
                combined = self._combine_states(
                    states=[x, y, z],
                    is_logits=[False, True, True]
                )

                # Forward through transformer and heads (bypassing embedding)
                output = self._forward_through_network(combined)
                z = output["logits"]
                total_iterations += 1

            # After inner loop completes, update answer state y
            # Combine only y and z (no x for answer update per paper)
            combined_yz = self._combine_states(
                states=[y, z],
                is_logits=[True, True]
            )

            # Forward through transformer and heads (bypassing embedding)
            output = self._forward_through_network(combined_yz)
            y = output["logits"]
            total_iterations += 1

            # Check halting condition after each outer iteration
            if self.enable_halting:
                halt_confidence = output["halt_confidence"]  # (B,)
                # All batch items must exceed threshold to halt
                if (halt_confidence >= self.halt_threshold).all():
                    halted_early = True
                    break

        # Return final answer state with metadata
        return {
            "logits": y,
            "halt_confidence": output["halt_confidence"],  # from last network call
            "iterations": total_iterations,
            "halted_early": halted_early,
        }
