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

        # Role embeddings for in-context mode: 0=demo_input, 1=demo_output, 2=test_input
        self.role_embedding = nn.Embedding(3, embedding.hidden_dim)

    def _combine_states(
        self,
        states: list[torch.Tensor],
        is_logits: list[bool]
    ) -> torch.Tensor:
        """
        Combine multiple states by adding their embeddings.

        States can be either raw grids (values 0-9) or logits (B, H, W, num_colors).
        Logit states use a soft (differentiable) embedding: a softmax-weighted sum
        over the color embedding vectors. This preserves gradient flow through the
        recursive state updates, unlike argmax which has zero gradient.

        Args:
            states: List of state tensors to combine
            is_logits: List of booleans indicating if each state is logits (True) or raw (False)

        Returns:
            Combined embedding tensor of shape (B, H, W, hidden_dim)
        """
        embeddings = []

        for state, logit_flag in zip(states, is_logits):
            if logit_flag:
                # Soft (differentiable) embedding: softmax-weighted sum over color embeddings.
                # state: (B, H, W, num_colors)
                probs = torch.softmax(state, dim=-1)
                # Color embedding vectors, excluding the pad token at index 10.
                # shape: (num_colors, hidden_dim)
                color_embs = self.embedding.embedding.weight[:self.num_colors]
                # Weighted sum: (B, H, W, num_colors) @ (num_colors, hidden_dim)
                # -> (B, H, W, hidden_dim)
                embedded = probs @ color_embs
            else:
                # Hard embedding for raw grid inputs (values 0-9 or -1 for padding)
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

    def forward_in_context(
        self,
        demo_inputs: torch.Tensor,   # (B, max_demos, H_d, W_d)
        demo_outputs: torch.Tensor,  # (B, max_demos, H_d, W_d)
        num_demos: torch.Tensor,     # (B,) int — actual demo count per batch item
        test_input: torch.Tensor,    # (B, H_t, W_t)
    ) -> dict:
        """
        Apply recursive refinement with demonstration context.

        Constructs a single flat sequence from all demo pairs and the test input:
            [demo1_in | demo1_out | demo2_in | demo2_out | ... | test_in]

        The recursive loop operates over this full sequence. At the end, only the
        logits for the test_input positions (the last H_t*W_t tokens) are returned.
        Role embeddings distinguish demo_input / demo_output / test_input tokens.

        Args:
            demo_inputs:  (B, max_demos, H_d, W_d) padded demo input grids
            demo_outputs: (B, max_demos, H_d, W_d) padded demo output grids
            num_demos:    (B,) actual number of valid demo pairs per batch item
            test_input:   (B, H_t, W_t) query input grid

        Returns:
            Dictionary containing:
                - logits: (B, H_t, W_t, num_colors) — logits for test grid only
                - halt_confidence: (B,)
                - iterations: total network forward passes
                - halted_early: bool
        """
        B, max_demos, H_d, W_d = demo_inputs.shape
        _, H_t, W_t = test_input.shape
        device = test_input.device

        demo_len = H_d * W_d    # tokens per demo grid
        test_len = H_t * W_t    # tokens for test input
        total_len = 2 * max_demos * demo_len + test_len

        # Color embedding matrix (shared with soft-embedding in recursive loop)
        color_embs = self.embedding.embedding.weight[:self.num_colors]  # (num_colors, hidden_dim)

        # Role embedding vectors (broadcast over sequence)
        role_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        r_demo_in  = self.role_embedding(role_ids[0])  # (hidden_dim,)
        r_demo_out = self.role_embedding(role_ids[1])  # (hidden_dim,)
        r_test_in  = self.role_embedding(role_ids[2])  # (hidden_dim,)

        # Build initial sequence embedding from demo pairs + test input
        segments = []
        for demo_idx in range(max_demos):
            di_emb = self.embedding(demo_inputs[:, demo_idx])   # (B, H_d, W_d, hidden_dim)
            di_emb = di_emb.view(B, demo_len, -1) + r_demo_in  # (B, demo_len, hidden_dim)
            segments.append(di_emb)

            do_emb = self.embedding(demo_outputs[:, demo_idx])  # (B, H_d, W_d, hidden_dim)
            do_emb = do_emb.view(B, demo_len, -1) + r_demo_out
            segments.append(do_emb)

        ti_emb = self.embedding(test_input).view(B, test_len, -1) + r_test_in
        segments.append(ti_emb)

        x_emb = torch.cat(segments, dim=1)  # (B, total_len, hidden_dim)

        # Attention mask: True = valid token, False = padded demo pair
        attn_mask = torch.ones(B, total_len, dtype=torch.bool, device=device)
        for b in range(B):
            n = int(num_demos[b].item())
            for demo_idx in range(n, max_demos):
                start = demo_idx * 2 * demo_len
                attn_mask[b, start:start + 2 * demo_len] = False

        # Initialize answer and latent states over TEST positions only.
        # Demo positions are read-only context — they always contribute x_emb and never
        # accumulate recursive state.  This prevents demo contamination where evolving
        # y/z logits over demo tokens bleed noise into the context signal.
        y = torch.zeros(B, test_len, self.num_colors, device=device, dtype=torch.float32)

        total_iterations = 0
        halted_early = False
        halt_confidence = None

        for t in range(self.outer_steps):
            z = torch.zeros_like(y)  # (B, test_len, num_colors)

            # Inner loop: refine latent z
            for i in range(self.inner_steps):
                y_emb = torch.softmax(y, dim=-1) @ color_embs  # (B, test_len, hidden_dim)
                z_emb = torch.softmax(z, dim=-1) @ color_embs  # (B, test_len, hidden_dim)
                # Demo positions keep their fixed x_emb; test positions get x + y + z
                combined = x_emb.clone()
                combined[:, -test_len:, :] = combined[:, -test_len:, :] + y_emb + z_emb
                hidden = self.network.transformer(combined, mask=attn_mask)
                # Only update z from test-position outputs
                z = self.network.output_head(hidden)[:, -test_len:, :]  # (B, test_len, num_colors)
                total_iterations += 1

            # Outer step: update answer y from (y, z) — no x at test positions per paper.
            # Demo positions still carry x_emb so the transformer retains context.
            y_emb = torch.softmax(y, dim=-1) @ color_embs
            z_emb = torch.softmax(z, dim=-1) @ color_embs
            combined_yz = x_emb.clone()
            combined_yz[:, -test_len:, :] = y_emb + z_emb   # overwrite test positions: y+z only
            hidden = self.network.transformer(combined_yz, mask=attn_mask)
            y = self.network.output_head(hidden)[:, -test_len:, :]   # (B, test_len, num_colors)
            halt_confidence = self.network.halting_head(hidden, mask=attn_mask)  # (B,)
            total_iterations += 1

            if self.enable_halting:
                if (halt_confidence >= self.halt_threshold).all():
                    halted_early = True
                    break

        # y already covers only test positions; reshape to spatial grid
        test_logits = y.contiguous().view(B, H_t, W_t, self.num_colors)

        return {
            "logits": test_logits,
            "halt_confidence": halt_confidence,
            "iterations": total_iterations,
            "halted_early": halted_early,
        }

    def forward_with_intermediates(self, x: torch.Tensor) -> dict:
        """
        Apply recursive refinement while collecting intermediate states at each outer iteration.

        Same nested loop structure as forward(), but stores y and halt_confidence after each
        outer iteration. This enables deep supervision where losses can be computed at
        intermediate reasoning steps, not just the final output.

        Args:
            x: Input grid tensor of shape (B, H, W) with values 0-9 or -1 (padding)

        Returns:
            Dictionary containing:
                - final_logits: Final answer state of shape (B, H, W, num_colors)
                - final_halt_confidence: Halting confidence from last network call (B,)
                - intermediate_states: List of dicts, one per outer iteration completed:
                    - logits: Answer state y at that iteration (B, H, W, num_colors)
                    - halt_confidence: Confidence at that iteration (B,)
                    - iteration: Outer iteration index (0-indexed)
                - iterations: Total number of network forward passes (int)
                - halted_early: Boolean indicating if halting stopped early (bool)
        """
        B, H, W = x.shape

        # Initialize answer state y to zeros (B, H, W, num_colors)
        y = torch.zeros(B, H, W, self.num_colors, device=x.device, dtype=torch.float32)

        # Track total iterations and intermediate states
        total_iterations = 0
        halted_early = False
        intermediate_states = []

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

            # Store intermediate state after this outer iteration (clone to avoid reference issues)
            intermediate_states.append({
                "logits": y.clone(),
                "halt_confidence": output["halt_confidence"].clone(),
                "iteration": t,
            })

            # Check halting condition after each outer iteration
            if self.enable_halting:
                halt_confidence = output["halt_confidence"]  # (B,)
                # All batch items must exceed threshold to halt
                if (halt_confidence >= self.halt_threshold).all():
                    halted_early = True
                    break

        # Return final answer state with intermediate states and metadata
        return {
            "final_logits": y,
            "final_halt_confidence": output["halt_confidence"],
            "intermediate_states": intermediate_states,
            "iterations": total_iterations,
            "halted_early": halted_early,
        }
