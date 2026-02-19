"""TRM training loop with terminal supervision."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..model import RecursiveRefinement


class TRMTrainer:
    """
    Training loop for TRM with terminal supervision only.

    This trainer implements the basic training mechanics:
    - Combined CrossEntropy + BCE loss
    - AdamW optimizer with paper betas (0.9, 0.95)
    - Single training step logic

    Terminal supervision means we only supervise the final output y,
    not intermediate states. Deep supervision (Phase 6) will extend
    this to supervise intermediate iterations.

    Args:
        model: RecursiveRefinement instance to train
        learning_rate: Learning rate for AdamW (default 1e-4)
        embed_lr: Separate learning rate for embedding layers (default: same as learning_rate).
            The paper uses 1e-2 for embeddings vs 1e-4 for the rest of the network.
        weight_decay: Weight decay for AdamW (default 0.01)
        beta1: First beta for AdamW (default 0.9)
        beta2: Second beta for AdamW (default 0.95)
        halting_loss_weight: Weight for BCE halting loss (default 0.1)
    """

    def __init__(
        self,
        model: RecursiveRefinement,
        learning_rate: float = 1e-4,
        embed_lr: float | None = None,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.95,
        halting_loss_weight: float = 0.1,
        use_amp: bool = False,
    ):
        self.model = model
        self.halting_loss_weight = halting_loss_weight
        self.use_amp = use_amp

        # Split parameters: embedding layers get a separate (higher) LR per the paper.
        # Paper uses 1e-2 for embeddings vs 1e-4 for the transformer weights.
        _embed_lr = embed_lr if embed_lr is not None else learning_rate
        embed_modules = [model.embedding, model.role_embedding]
        embed_param_ids = set(id(p) for m in embed_modules for p in m.parameters())
        embed_params = [p for p in model.parameters() if id(p) in embed_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in embed_param_ids]

        # Create AdamW optimizer with paper betas (TRAIN-05)
        self.optimizer = AdamW(
            [
                {"params": embed_params, "lr": _embed_lr},
                {"params": other_params, "lr": learning_rate},
            ],
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        halt_confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss for terminal supervision.

        Loss consists of two components:
        1. CrossEntropy loss for answer accuracy (TRAIN-03)
        2. BCE loss for halting prediction (TRAIN-04)

        The halting target is binary: 1.0 if the sample is 100% correct,
        0.0 otherwise. This teaches the model to output high confidence
        only when it has the right answer.

        Args:
            logits: Predicted logits of shape (B, H, W, num_colors)
            targets: Target grid of shape (B, H, W) with values 0-9
            halt_confidence: Halting confidence of shape (B,)
            mask: Boolean mask of shape (B, H, W), True for valid positions

        Returns:
            Dictionary containing:
                - total_loss: Combined CE + weighted BCE loss
                - ce_loss: CrossEntropy loss for answer accuracy
                - bce_loss: BCE loss for halting prediction
                - accuracy: Fraction of correctly predicted cells (masked)
        """
        B, H, W, num_colors = logits.shape

        # Flatten spatial dimensions: (B*H*W, num_colors) and (B*H*W,)
        logits_flat = logits.view(-1, num_colors)  # (B*H*W, num_colors)
        targets_flat = targets.view(-1)  # (B*H*W,)
        mask_flat = mask.view(-1)  # (B*H*W,)

        # Select only valid (non-padded) positions
        valid_logits = logits_flat[mask_flat]  # (N_valid, num_colors)
        valid_targets = targets_flat[mask_flat]  # (N_valid,)

        # Handle edge case where mask has no valid positions
        if valid_logits.numel() == 0:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            accuracy = torch.tensor(1.0, device=logits.device)
        else:
            # CrossEntropy loss for answer accuracy (TRAIN-03)
            ce_loss = F.cross_entropy(valid_logits, valid_targets)

            # Compute predictions for accuracy
            predictions = torch.argmax(valid_logits, dim=-1)  # (N_valid,)
            accuracy = (predictions == valid_targets).float().mean()

        # Compute per-sample accuracy for halting target
        # Reshape predictions to per-sample: (B, H*W)
        predictions_grid = torch.argmax(logits, dim=-1)  # (B, H, W)

        # Per-sample: count correct predictions vs total valid cells
        per_sample_correct = []
        for b in range(B):
            sample_mask = mask[b]  # (H, W)
            if sample_mask.sum() == 0:
                # No valid positions, consider it correct
                per_sample_correct.append(1.0)
            else:
                sample_preds = predictions_grid[b][sample_mask]  # (N_valid_b,)
                sample_targets = targets[b][sample_mask]  # (N_valid_b,)
                is_correct = (sample_preds == sample_targets).all().float().item()
                per_sample_correct.append(is_correct)

        # BCE halting loss (TRAIN-04)
        # Target is 1.0 if sample is 100% correct, 0.0 otherwise
        halt_target = torch.tensor(
            per_sample_correct, device=logits.device, dtype=torch.float32
        )
        # BCE must run in float32 (F.binary_cross_entropy is unsafe under BF16 autocast)
        with torch.autocast(device_type="cuda", enabled=False):
            bce_loss = F.binary_cross_entropy(halt_confidence.float(), halt_target.float())

        # Combine losses
        total_loss = ce_loss + self.halting_loss_weight * bce_loss

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "bce_loss": bce_loss,
            "accuracy": accuracy,
        }

    def train_step(
        self,
        input_grids: torch.Tensor,
        target_grids: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Single training step: forward, loss, backward, step.

        Args:
            input_grids: Input grid tensor of shape (B, H, W) with values 0-9 or -1
            target_grids: Target grid tensor of shape (B, H, W) with values 0-9
            mask: Boolean mask of shape (B, H, W), True for valid positions

        Returns:
            Dictionary containing:
                - total_loss: Combined loss value
                - ce_loss: CrossEntropy loss value
                - bce_loss: BCE loss value
                - accuracy: Fraction of correctly predicted cells
                - iterations: Number of network forward passes
        """
        # Set model to training mode
        self.model.train()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass + loss (BF16 autocast when use_amp=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            output = self.model(input_grids)
            logits = output["logits"]
            halt_confidence = output["halt_confidence"]
            loss_dict = self.compute_loss(logits, target_grids, halt_confidence, mask)

        # Backward pass
        loss_dict["total_loss"].backward()

        # Optimizer step
        self.optimizer.step()

        # Return metrics (detach for clean values)
        return {
            "total_loss": loss_dict["total_loss"].item(),
            "ce_loss": loss_dict["ce_loss"].item(),
            "bce_loss": loss_dict["bce_loss"].item(),
            "accuracy": loss_dict["accuracy"].item(),
            "iterations": output["iterations"],
        }

    def train_step_in_context(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        num_demos: torch.Tensor,
        test_input: torch.Tensor,
        target_grids: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Single in-context training step.

        Passes all demo pairs as context alongside the test input, then
        computes loss only on the test output positions.

        Args:
            demo_inputs:  (B, max_demos, H_d, W_d) padded demo inputs
            demo_outputs: (B, max_demos, H_d, W_d) padded demo outputs
            num_demos:    (B,) actual demo count per batch item
            test_input:   (B, H_t, W_t) query input
            target_grids: (B, H_t, W_t) expected test output
            mask:         (B, H_t, W_t) True for valid output positions

        Returns:
            Dictionary with total_loss, ce_loss, bce_loss, accuracy, iterations.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            output = self.model.forward_in_context(demo_inputs, demo_outputs, num_demos, test_input)
            logits = output["logits"]
            halt_confidence = output["halt_confidence"]
            loss_dict = self.compute_loss(logits, target_grids, halt_confidence, mask)

        loss_dict["total_loss"].backward()
        self.optimizer.step()

        return {
            "total_loss": loss_dict["total_loss"].item(),
            "ce_loss": loss_dict["ce_loss"].item(),
            "bce_loss": loss_dict["bce_loss"].item(),
            "accuracy": loss_dict["accuracy"].item(),
            "iterations": output["iterations"],
        }
