"""Checkpoint save/load and best model tracking for TRM training.

This module provides utilities for:
- Saving complete training state (model + EMA + optimizer)
- Loading checkpoints for training resumption
- Tracking and auto-saving the best model based on validation accuracy

Key functions:
    - save_checkpoint: Save trainer state to disk
    - load_checkpoint: Load trainer state from checkpoint
    - BestModelTracker: Auto-save best model when accuracy improves

Design:
    - Checkpoints are PyTorch state dicts saved with torch.save
    - State includes all components needed for exact training resumption
    - BestModelTracker maintains best accuracy and calls save_checkpoint
"""
import torch
from pathlib import Path
from typing import Optional, Any


def save_checkpoint(
    trainer: Any,
    filepath: str | Path,
    epoch: int,
    step: int,
    best_accuracy: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save complete training state to checkpoint file.

    Saves model, EMA, optimizer state from trainer.state_dict() along with
    training progress metadata. Checkpoint can be used to resume training
    from this exact point.

    Args:
        trainer: Trainer instance with state_dict() method
        filepath: Path where checkpoint will be saved
        epoch: Current training epoch
        step: Current training step
        best_accuracy: Best validation accuracy so far (optional)
        metadata: Additional metadata to save (optional)

    Saves:
        Dictionary containing:
            - trainer_state: Complete trainer state dict
            - epoch: Training epoch number
            - step: Training step number
            - best_accuracy: Best validation accuracy (if provided)
            - metadata: Additional metadata (if provided)

    Example:
        save_checkpoint(
            trainer=deep_trainer,
            filepath="checkpoints/model_epoch_10.pt",
            epoch=10,
            step=5000,
            best_accuracy=0.87,
            metadata={"config": config_dict}
        )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }

    # Include EMA if available
    if hasattr(trainer, 'ema_model') and trainer.ema_model is not None:
        checkpoint["ema_state"] = trainer.ema_model.state_dict()

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

    Loads checkpoint file and restores trainer state (model, EMA, optimizer).
    Returns checkpoint metadata for training loop to resume from correct position.

    Args:
        trainer: Trainer instance with load_state_dict() method
        filepath: Path to checkpoint file
        map_location: Device mapping for checkpoint (e.g., 'cpu', 'cuda:0')

    Returns:
        Dictionary containing:
            - epoch: Epoch number from checkpoint
            - step: Step number from checkpoint
            - best_accuracy: Best validation accuracy (if saved)
            - metadata: Additional metadata (if saved)

    Example:
        info = load_checkpoint(
            trainer=deep_trainer,
            filepath="checkpoints/model_epoch_10.pt",
            map_location="cpu"
        )
        start_epoch = info["epoch"] + 1
        best_acc = info.get("best_accuracy", 0.0)
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    # Restore trainer state
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore EMA if available
    if "ema_state" in checkpoint and hasattr(trainer, 'ema_model') and trainer.ema_model is not None:
        trainer.ema_model.load_state_dict(checkpoint["ema_state"])

    # Return metadata for training loop
    result = {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
    }

    if "best_accuracy" in checkpoint:
        result["best_accuracy"] = checkpoint["best_accuracy"]

    if "metadata" in checkpoint:
        result["metadata"] = checkpoint["metadata"]

    return result


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
        """
        Initialize best model tracker.

        Args:
            save_dir: Directory for saving best model
            filename: Checkpoint filename (default: "best_model.pt")
            min_delta: Minimum improvement threshold (default: 0.0)
        """
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
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Check if accuracy improved and save checkpoint if so.

        Args:
            trainer: Trainer instance to save
            accuracy: Current validation accuracy
            epoch: Current epoch number
            step: Current step number
            metadata: Additional metadata to save (optional)

        Returns:
            True if new best accuracy achieved and checkpoint saved
            False if accuracy did not improve

        Example:
            if tracker.update(trainer, val_acc, epoch, step):
                print(f"Saved new best model with accuracy {val_acc:.4f}")
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
                metadata=metadata,
            )
            return True

        return False

    def get_best_accuracy(self) -> float:
        """Get the best accuracy seen so far."""
        return self.best_accuracy if self.best_accuracy != float("-inf") else 0.0
