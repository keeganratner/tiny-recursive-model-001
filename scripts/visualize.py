#!/usr/bin/env python
"""Generate iteration timeline visualization for TRM model."""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trm.model import TRMNetwork, RecursiveRefinement
from trm.model.embedding import GridEmbedding
from trm.data.dataset import ARCDataset
from trm.visualization import IterationHistoryCapture, render_iteration_timeline
from trm.evaluation.checkpointing import load_checkpoint
from trm.training.deep_supervision import DeepSupervisionTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Generate iteration timeline visualization"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--task-id", "-t",
        type=str,
        required=True,
        help="ARC task ID to visualize (e.g., '007bbfb7')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="timeline.png",
        help="Output image path (default: timeline.png)"
    )
    parser.add_argument(
        "--pair-index", "-p",
        type=int,
        default=0,
        help="Train pair index to visualize (default: 0)"
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Don't show target grid in visualization"
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Don't show color-coded differences"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150)"
    )
    # Model config (must match checkpoint)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--inner-steps", type=int, default=6)

    args = parser.parse_args()

    # Check checkpoint file exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Load ARC dataset and find task
    print("Loading ARC dataset...")
    dataset = ARCDataset(data_dir="data/ARC-AGI", split="training")

    task = None
    task_index = None
    for i in range(len(dataset)):
        if dataset[i]["task_id"] == args.task_id:
            task = dataset[i]
            task_index = i
            break

    if task is None:
        print(f"Error: Task {args.task_id} not found in training split")
        print(f"Available task count: {len(dataset)}")
        sys.exit(1)

    print(f"Found task: {args.task_id} (index {task_index})")

    # Check pair index is valid
    num_train_pairs = len(task["train_inputs"])
    if args.pair_index >= num_train_pairs:
        print(f"Error: Pair index {args.pair_index} out of range")
        print(f"Task has {num_train_pairs} training pairs (indices 0-{num_train_pairs - 1})")
        sys.exit(1)

    # Get input/output pair
    input_grid = task["train_inputs"][args.pair_index]
    target_grid = task["train_outputs"][args.pair_index]

    print(f"Input grid shape: {input_grid.shape}")
    print(f"Target grid shape: {target_grid.shape}")

    # Build model and load checkpoint
    print("Building model...")
    network = TRMNetwork(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    embedding = GridEmbedding(args.hidden_dim)
    refinement = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
    )
    trainer = DeepSupervisionTrainer(refinement)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_data = load_checkpoint(args.checkpoint, trainer)
    print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")

    if "best_accuracy" in checkpoint_data:
        print(f"Best accuracy: {checkpoint_data['best_accuracy']:.4f}")

    # Set model to eval mode
    refinement.eval()

    # Capture iteration history
    print("Running inference and capturing history...")
    capture = IterationHistoryCapture(refinement)
    history = capture.capture(input_grid.unsqueeze(0))  # Add batch dimension

    print(f"Captured {len(history.iteration_grids)} iterations")
    print(f"Halted early: {history.halted_early}")

    # Render and save timeline
    print(f"Rendering timeline to {args.output}...")
    target = None if args.no_target else target_grid
    fig = render_iteration_timeline(
        history=history,
        target_grid=target,
        output_path=args.output,
        show_diff=not args.no_diff,
        dpi=args.dpi,
    )

    print(f"Successfully saved timeline to {args.output}")


if __name__ == "__main__":
    main()
