"""
Overfit diagnostic: train on a single synthetic task to verify the model can converge.

Uses a simple copy task (output = input) on a fixed 4x4 grid. If the model can
memorize this in ~200 steps, gradient flow is working correctly.

Run with:
    python scripts/overfit_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.training import DeepSupervisionTrainer


def make_copy_task(grid_size: int = 4, num_colors: int = 6, seed: int = 42):
    """Create a fixed input/output pair where output = input (copy task)."""
    torch.manual_seed(seed)
    grid = torch.randint(0, num_colors, (1, grid_size, grid_size))
    mask = torch.ones(1, grid_size, grid_size, dtype=torch.bool)
    return grid, grid.clone(), mask


def build_model(hidden_dim: int = 128):
    network = TRMNetwork(
        num_colors=10,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
    )
    embedding = GridEmbedding(hidden_dim=hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=3,
        inner_steps=6,
        num_colors=10,
        halt_threshold=0.9,
        enable_halting=False,  # Disable halting for overfit test
    )
    return model


def run_overfit_test(steps: int = 300, report_every: int = 25):
    print("=" * 60)
    print("Overfit diagnostic: copy task (output = input)")
    print("=" * 60)

    input_grid, target_grid, mask = make_copy_task()
    print(f"Task: copy a {input_grid.shape[1]}x{input_grid.shape[2]} grid")
    print(f"Input:  {input_grid[0].tolist()}")
    print(f"Target: {target_grid[0].tolist()}")
    print()

    model = build_model(hidden_dim=128)
    trainer = DeepSupervisionTrainer(
        model,
        learning_rate=3e-4,
        max_sup_steps=3,
        grad_clip_norm=1.0,
        ema_decay=0.999,
    )

    initial_loss = None
    final_loss = None
    final_acc = None

    for step in range(1, steps + 1):
        result = trainer.train_step_deep_supervision(input_grid, target_grid, mask)

        if step == 1:
            initial_loss = result["total_loss"]

        if step % report_every == 0 or step == 1:
            acc_pct = result["accuracy"] * 100
            print(
                f"Step {step:4d} | loss: {result['total_loss']:.4f} | "
                f"ce: {result['ce_loss']:.4f} | acc: {acc_pct:.1f}%"
            )

    final_loss = result["total_loss"]
    final_acc = result["accuracy"]

    print()
    print("=" * 60)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Final acc:    {final_acc * 100:.1f}%")

    loss_reduced = final_loss < initial_loss * 0.5
    print()
    if loss_reduced:
        print("PASS: Loss reduced by >50% — gradient flow is working.")
    else:
        print(f"FAIL: Loss only went from {initial_loss:.4f} -> {final_loss:.4f}.")
        print("      Gradient flow may still be broken.")

    return loss_reduced


if __name__ == "__main__":
    success = run_overfit_test(steps=300, report_every=25)
    sys.exit(0 if success else 1)
