"""Checkpoint save/load and best model tracking for TRM training.

This module provides utilities for:
- Saving complete training state (model + EMA + optimizer)
- Loading checkpoints for training resumption
- Tracking and auto-saving the best model based on validation accuracy

Key functions:
    - save_checkpoint: Save trainer state to disk
    - load_checkpoint: Load trainer state from checkpoint
    - save_periodic_checkpoint: Save a timestamped periodic checkpoint
    - BestModelTracker: Auto-save best model when accuracy improves

Design:
    - Checkpoints are PyTorch state dicts saved with torch.save
    - State includes all components needed for exact training resumption
    - BestModelTracker maintains best accuracy and calls save_checkpoint
"""
import torch
from pathlib import Path
from typing import Optional, Any


def _get_ema(trainer: Any):
    """Return the EMA model if it exists, regardless of attribute name."""
    if hasattr(trainer, 'ema') and trainer.ema is not None:
        return trainer.ema
    if hasattr(trainer, 'ema_model') and trainer.ema_model is not None:
        return trainer.ema_model
    return None


def save_checkpoint(
    trainer: Any,
    filepath: str | Path,
    epoch: int,
    step: int,
    best_accuracy: Optional[float] = None,
    total_epochs: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save complete training state to checkpoint file.

    Args:
        trainer: Trainer instance with model and optimizer attributes
        filepath: Path where checkpoint will be saved
        epoch: Current training epoch (within this run)
        step: Current training step
        best_accuracy: Best validation accuracy so far (optional)
        total_epochs: Cumulative epochs across all sessions (optional)
        metadata: Additional metadata to save (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "total_epochs": total_epochs if total_epochs is not None else (epoch + 1),
    }

    # Include EMA if available (supports both trainer.ema and trainer.ema_model)
    ema = _get_ema(trainer)
    if ema is not None:
        checkpoint["ema_state"] = ema.state_dict()

    if best_accuracy is not None:
        checkpoint["best_accuracy"] = best_accuracy

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, filepath)


def load_checkpoint(
    trainer: Any,
    filepath: str | Path,
    map_location: Optional[str | torch.device] = None,
) -> dict:
    """
    Load checkpoint and restore trainer state.

    Args:
        trainer: Trainer instance with model and optimizer attributes
        filepath: Path to checkpoint file
        map_location: Device mapping for checkpoint (e.g., 'cpu', 'cuda:0')

    Returns:
        Dictionary containing:
            - epoch: Epoch number from checkpoint
            - step: Step number from checkpoint
            - total_epochs: Cumulative epochs across all sessions
            - best_accuracy: Best validation accuracy (0.0 if not saved)
            - metadata: Additional metadata ({} if not saved)
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    # Restore trainer state
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore EMA if available (supports both trainer.ema and trainer.ema_model)
    ema = _get_ema(trainer)
    if "ema_state" in checkpoint and ema is not None:
        ema.load_state_dict(checkpoint["ema_state"])

    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "total_epochs": checkpoint.get("total_epochs", checkpoint["epoch"] + 1),
        "best_accuracy": checkpoint.get("best_accuracy", 0.0),
        "metadata": checkpoint.get("metadata", {}),
    }


def save_periodic_checkpoint(
    trainer: Any,
    save_dir: str | Path,
    epoch: int,
    total_epochs: int,
    best_accuracy: Optional[float] = None,
) -> Path:
    """Save a timestamped checkpoint for periodic saves (not best-only).

    Args:
        trainer: Trainer instance
        save_dir: Directory to save checkpoint in
        epoch: Current epoch index within this run
        total_epochs: Cumulative epoch count across all sessions
        best_accuracy: Best validation accuracy so far (optional)

    Returns:
        Path to the saved checkpoint file
    """
    filepath = Path(save_dir) / f"checkpoint_epoch_{total_epochs:06d}.pt"
    save_checkpoint(
        trainer, filepath,
        epoch=epoch, step=0,
        best_accuracy=best_accuracy,
        total_epochs=total_epochs,
    )
    return filepath


class BestModelTracker:
    """
    Track best model and auto-save when validation accuracy improves.

    Maintains the best validation accuracy seen so far and automatically
    saves a checkpoint whenever a new best is achieved. Prevents unnecessary
    disk I/O by only saving when accuracy actually improves.

    Args:
        save_dir: Directory where best model checkpoints are saved
        filename: Name for best model checkpoint file (default: "best_model.pt")
        min_delta: Minimum improvement to count as better (default: 0.0)

    Example:
        tracker = BestModelTracker(save_dir="checkpoints")

        for epoch in range(num_epochs):
            val_acc = validate(trainer)
            if tracker.update(trainer, val_acc, epoch, step):
                print(f"New best accuracy: {val_acc:.4f}")
    """

    def __init__(
        self,
        save_dir: str | Path,
        filename: str = "best_model.pt",
        min_delta: float = 0.0,
    ):
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.min_delta = min_delta
        self.best_accuracy = float("-inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        trainer: Any,
        accuracy: float,
        epoch: int,
        step: int,
        total_epochs: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Check if accuracy improved and save checkpoint if so.

        Args:
            trainer: Trainer instance to save
            accuracy: Current validation accuracy
            epoch: Current epoch number
            step: Current step number
            total_epochs: Cumulative epoch count (optional)
            metadata: Additional metadata to save (optional)

        Returns:
            True if new best accuracy achieved and checkpoint saved
            False if accuracy did not improve
        """
        if accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            filepath = self.save_dir / self.filename

            save_checkpoint(
                trainer=trainer,
                filepath=filepath,
                epoch=epoch,
                step=step,
                best_accuracy=accuracy,
                total_epochs=total_epochs,
                metadata=metadata,
            )
            return True

        return False

    def get_best_accuracy(self) -> float:
        """Get the best accuracy seen so far."""
        return self.best_accuracy if self.best_accuracy != float("-inf") else 0.0
