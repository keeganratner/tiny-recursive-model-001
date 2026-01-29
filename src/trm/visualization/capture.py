"""Iteration history capture for visualizing recursive refinement."""
from dataclasses import dataclass
from typing import List, Optional

import torch

from ..model.recursive import RecursiveRefinement


@dataclass
class IterationHistory:
    """
    History of model predictions across outer iterations.

    Stores the evolution of the answer state y through the recursive refinement
    process, enabling visualization of how the model iteratively improves its
    prediction.

    Attributes:
        input_grid: Original input grid (H, W) with values 0-9 or -1 (padding)
        iteration_grids: List of discrete predictions (H, W) at each outer iteration
        halt_confidences: Halting confidence scores at each iteration
        halted_early: Whether the model stopped before reaching max outer_steps
        final_prediction: Final discrete prediction (H, W) - same as iteration_grids[-1]
        iteration_losses: Optional list of accuracy values per iteration (0.0 to 1.0)
    """
    input_grid: torch.Tensor           # (H, W) original input
    iteration_grids: List[torch.Tensor] # List of (H, W) predictions at each outer iteration
    halt_confidences: List[float]       # Confidence at each iteration
    halted_early: bool                  # Whether model halted before max iterations
    final_prediction: torch.Tensor      # (H, W) final argmax prediction
    iteration_losses: Optional[List[float]] = None  # Optional accuracy per iteration


class IterationHistoryCapture:
    """
    Wraps RecursiveRefinement to capture iteration history for visualization.

    This class runs model inference and extracts intermediate predictions at each
    outer iteration, converting logits to discrete grids for visualization. It uses
    forward_with_intermediates() to collect intermediate states, then processes them
    into a visualization-friendly format.

    Usage:
        capture = IterationHistoryCapture(refinement_module)
        history = capture.capture(input_grid)
        # history now contains input, iterations, and metadata

    Args:
        refinement: RecursiveRefinement module to wrap
    """

    def __init__(self, refinement: RecursiveRefinement):
        """
        Initialize the capture wrapper.

        Args:
            refinement: RecursiveRefinement instance to capture from
        """
        self.refinement = refinement

    @torch.no_grad()
    def capture(self, input_grid: torch.Tensor) -> IterationHistory:
        """
        Run inference and capture intermediate states.

        Performs forward pass with intermediate state collection, then converts
        the logits at each iteration to discrete predictions via argmax. Handles
        both batched and single-example inputs.

        Args:
            input_grid: Input tensor (B, H, W) or (H, W) with values 0-9 or -1 (padding)

        Returns:
            IterationHistory with input, iterations, and metadata

        Raises:
            ValueError: If batch size > 1 (current implementation supports single examples)
        """
        # Ensure model is in eval mode
        self.refinement.eval()

        # Handle unbatched input: (H, W) -> (1, H, W)
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
            was_unbatched = True
        else:
            was_unbatched = False

        B, H, W = input_grid.shape

        # Current implementation supports single examples only
        if B > 1:
            raise ValueError(
                f"IterationHistoryCapture currently supports batch_size=1, got {B}. "
                "Process examples individually for visualization."
            )

        # Run inference with intermediate state collection
        output = self.refinement.forward_with_intermediates(input_grid)

        # Extract data
        intermediate_states = output["intermediate_states"]  # List of dicts
        halted_early = output["halted_early"]

        # Convert intermediate logits to discrete predictions
        iteration_grids = []
        halt_confidences = []

        for state_dict in intermediate_states:
            logits = state_dict["logits"]  # (1, H, W, num_colors)
            halt_conf = state_dict["halt_confidence"]  # (1,)

            # Convert logits to discrete predictions: argmax over color dimension
            discrete_pred = torch.argmax(logits, dim=-1)  # (1, H, W)

            # Store as CPU tensors for safe storage (clone to detach from computation graph)
            iteration_grids.append(discrete_pred.squeeze(0).cpu().clone())  # (H, W)
            halt_confidences.append(float(halt_conf.squeeze().cpu().item()))

        # Get final prediction (same as last iteration)
        final_prediction = iteration_grids[-1].clone()

        # Store input grid (squeeze batch dimension if it was added)
        stored_input = input_grid.squeeze(0).cpu().clone() if was_unbatched else input_grid[0].cpu().clone()

        return IterationHistory(
            input_grid=stored_input,
            iteration_grids=iteration_grids,
            halt_confidences=halt_confidences,
            halted_early=halted_early,
            final_prediction=final_prediction,
            iteration_losses=None,
        )

    @torch.no_grad()
    def capture_with_losses(
        self, input_grid: torch.Tensor, target_grid: torch.Tensor
    ) -> IterationHistory:
        """
        Run inference and capture intermediate states with loss computation.

        Same as capture() but also computes accuracy per iteration against target.
        This is useful for visualizing how prediction quality evolves.

        Args:
            input_grid: Input tensor (B, H, W) or (H, W) with values 0-9 or -1 (padding)
            target_grid: Target tensor (H, W) with values 0-9 or -1 (padding)

        Returns:
            IterationHistory with iteration_losses populated

        Raises:
            ValueError: If batch size > 1 (current implementation supports single examples)
        """
        # First, capture history normally
        history = self.capture(input_grid)

        # Compute losses for each iteration
        losses = compute_iteration_losses(history, target_grid)

        # Update history with losses
        history.iteration_losses = losses

        return history


def compute_iteration_losses(
    history: IterationHistory, target_grid: torch.Tensor
) -> List[float]:
    """
    Compute exact-match accuracy for each iteration vs target.

    This function computes the accuracy (0.0 to 1.0) of each iteration's prediction
    compared to the target grid. Accuracy of 1.0 means ALL non-padded pixels match
    perfectly, 0.0 means at least one mismatch exists.

    Args:
        history: IterationHistory with iteration_grids
        target_grid: Target grid (H, W) with values 0-9 or -1 (padding)

    Returns:
        List of accuracy values (0.0 to 1.0), one per iteration

    Example:
        >>> history = capture.capture(input_grid)
        >>> target = torch.zeros(5, 5)
        >>> losses = compute_iteration_losses(history, target)
        >>> losses
        [0.0, 0.0, 1.0]  # Got it right on 3rd iteration
    """
    accuracies = []

    # Convert target to numpy for comparison
    target_np = target_grid.cpu().numpy()

    for iteration_grid in history.iteration_grids:
        # Convert prediction to numpy
        pred_np = iteration_grid.cpu().numpy()

        # Create mask for valid positions (ignore padding)
        mask = (target_np != -1) & (pred_np != -1)

        # Handle empty mask (no valid positions) - vacuously true
        if mask.sum() == 0:
            accuracies.append(1.0)
            continue

        # Extract valid positions
        valid_preds = pred_np[mask]
        valid_targets = target_np[mask]

        # Check if ALL predictions match targets
        is_perfect = (valid_preds == valid_targets).all()
        accuracies.append(float(is_perfect))

    return accuracies
