"""Test-time augmentation ensemble evaluation for TRM on ARC-AGI.

Implements the 1000-sample ensemble from Samsung's "Less is More" paper
(arXiv:2510.04871).  For each task, N random D8 + color-permutation
augmentations are applied, predictions are inverse-transformed back to
original space, and per-cell majority voting gives the final answer.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.data import ARCDataset
from src.trm.data.augmentation import D8Transform, ColorPermutation
from src.trm.evaluation import load_checkpoint

# Inverse D8 transform IDs: D8_INVERSE[i] is the inverse of transform i
#   0 (identity)  <-> 0
#   1 (rot90 CW)  <-> 3 (rot270 CW)
#   2 (rot180)    <-> 2
#   3 (rot270 CW) <-> 1 (rot90 CW)
#   4 (flipH)     <-> 4
#   5 (flipV)     <-> 5
#   6 (transpose) <-> 6
#   7 (antidiag)  <-> 7
D8_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def pad_grid(grid: torch.Tensor, target_h: int, target_w: int, pad_val: int = -1) -> torch.Tensor:
    """Pad a 2-D grid to (target_h, target_w) with pad_val."""
    h, w = grid.shape
    padded = torch.full((target_h, target_w), pad_val, dtype=grid.dtype, device=grid.device)
    padded[:h, :w] = grid
    return padded


def build_demo_tensor(
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    device: torch.device,
):
    """Stack variable-size demo grids into padded tensors for forward_in_context.

    Returns:
        demo_in:   (1, max_demos, H, W) long
        demo_out:  (1, max_demos, H, W) long
        num_demos: (1,) long
    """
    n = len(inputs)
    max_h = max(max(g.shape[0] for g in inputs), max(g.shape[0] for g in outputs))
    max_w = max(max(g.shape[1] for g in inputs), max(g.shape[1] for g in outputs))

    di = torch.stack([pad_grid(g, max_h, max_w) for g in inputs])   # (n, H, W)
    do = torch.stack([pad_grid(g, max_h, max_w) for g in outputs])  # (n, H, W)

    return (
        di.unsqueeze(0).to(device),               # (1, n, H, W)
        do.unsqueeze(0).to(device),
        torch.tensor([n], dtype=torch.long, device=device),  # (1,)
    )


def build_test_tensor(grid: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return (1, H, W) test input tensor."""
    return grid.unsqueeze(0).to(device)   # (1, H, W)


@torch.no_grad()
def ensemble_predict(
    model: RecursiveRefinement,
    train_inputs: list[torch.Tensor],
    train_outputs: list[torch.Tensor],
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    Run N augmented forward passes and return the majority-voted prediction.

    Returns:
        pred:  (H_t, W_t) long tensor — majority-voted prediction in original space
        exact: float — 1.0 if pred matches test_output exactly, else 0.0
    """
    H_t, W_t = test_output.shape
    # Accumulate per-cell, per-color vote counts
    vote_counts = torch.zeros(H_t, W_t, 10, dtype=torch.float32)

    for _ in range(n_samples):
        # 1. Sample random D8 + color augmentation
        d8_id = torch.randint(0, 8, ()).item()
        color_perm = ColorPermutation.random()
        d8 = D8Transform(int(d8_id))
        inv_d8 = D8Transform(D8_INVERSE[int(d8_id)])

        # 2. Augment all demo pairs and test input (consistent transform)
        aug_ti = color_perm.apply(d8.apply(test_input))  # (H_t', W_t')
        aug_demo_in  = [color_perm.apply(d8.apply(g)) for g in train_inputs]
        aug_demo_out = [color_perm.apply(d8.apply(g)) for g in train_outputs]

        # 3. Build model-ready tensors
        demo_in_t, demo_out_t, num_demos = build_demo_tensor(aug_demo_in, aug_demo_out, device)
        test_in_t = build_test_tensor(aug_ti, device)   # (1, H_t', W_t')

        # 4. Run model
        out = model.forward_in_context(demo_in_t, demo_out_t, num_demos, test_in_t)
        logits = out["logits"]  # (1, H_t', W_t', 10)
        pred_aug = logits[0].argmax(dim=-1)             # (H_t', W_t')

        # 5. Inverse-transform prediction back to original space
        #    a. Inverse D8 (restores original spatial layout)
        pred_d8 = inv_d8.apply(pred_aug)               # should be (H_t, W_t)
        #    b. Verify shape matches target (D8 shape-swap transforms on non-square grids
        #       may produce an irreconcilable mismatch — skip rather than corrupt votes)
        if pred_d8.shape != (H_t, W_t):
            continue
        #    c. Inverse color permutation
        inv_color = torch.argsort(color_perm.permutation).to(device)
        pred_orig = inv_color[pred_d8.clamp(min=0)]    # (H_t, W_t)

        # 6. Accumulate one-hot votes
        pred_oh = F.one_hot(pred_orig, num_classes=10).float()  # (H_t, W_t, 10)
        vote_counts += pred_oh.cpu()

    # Majority vote
    final_pred = vote_counts.argmax(dim=-1)  # (H_t, W_t)

    # Exact match
    target = test_output.cpu()
    exact = 1.0 if (final_pred == target).all().item() else 0.0

    return final_pred, exact


def load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract model config from checkpoint if saved, else return defaults."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("config", {})


def create_model(cfg: dict, device: torch.device) -> RecursiveRefinement:
    """Create model matching the saved checkpoint config."""
    hidden_dim  = cfg.get("hidden_dim", 256)
    num_heads   = cfg.get("num_heads", 4)
    num_layers  = cfg.get("num_layers", 2)
    outer_steps = cfg.get("outer_steps", 2)
    inner_steps = cfg.get("inner_steps", 3)

    network   = TRMNetwork(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
    embedding = GridEmbedding(hidden_dim=hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        enable_halting=True,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="TRM ensemble evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--n-samples", type=int, default=1000, help="Augmentation samples per task")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks evaluated (for quick testing)")
    parser.add_argument("--medium", action="store_true", help="Use medium model config (h=256, T=2, n=3)")
    parser.add_argument("--fast", action="store_true", help="Use fast model config (h=64, T=1, n=1)")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Ensemble samples: {args.n_samples}")

    # Build model with correct config
    if args.fast:
        cfg = dict(hidden_dim=64, num_heads=2, num_layers=1, outer_steps=1, inner_steps=1)
    elif args.medium:
        cfg = dict(hidden_dim=256, num_heads=4, num_layers=2, outer_steps=2, inner_steps=3)
    else:
        cfg = load_config_from_checkpoint(args.checkpoint)

    model = create_model(cfg, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint weights — BestModelTracker saves under key "model_state"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset
    data_dir = args.data_dir or str(project_root / "data")
    dataset = ARCDataset(data_dir=data_dir, split=args.split)
    n_tasks = len(dataset) if args.max_tasks is None else min(args.max_tasks, len(dataset))
    print(f"Tasks to evaluate: {n_tasks} / {len(dataset)}")
    print("-" * 60)

    total_exact = 0
    t0 = time.time()

    for task_idx in range(n_tasks):
        item = dataset[task_idx]
        task_id = item["task_id"]

        train_in  = item["train_inputs"]    # List[Tensor(H,W)]
        train_out = item["train_outputs"]
        test_in_list  = item["test_inputs"]
        test_out_list = item["test_outputs"]

        task_correct = 0
        for test_pair_idx in range(len(test_in_list)):
            test_input  = test_in_list[test_pair_idx]
            test_output = test_out_list[test_pair_idx]

            pred, exact = ensemble_predict(
                model=model,
                train_inputs=train_in,
                train_outputs=train_out,
                test_input=test_input,
                test_output=test_output,
                n_samples=args.n_samples,
                device=device,
            )
            task_correct += exact

        task_solved = 1.0 if task_correct == len(test_in_list) else 0.0
        total_exact += task_solved

        elapsed = time.time() - t0
        running_acc = total_exact / (task_idx + 1)
        print(
            f"[{task_idx+1:4d}/{n_tasks}] {task_id} | "
            f"solved={task_solved:.0f} | "
            f"running_acc={running_acc:.4f} | "
            f"elapsed={elapsed:.0f}s"
        )

    final_acc = total_exact / n_tasks
    print("=" * 60)
    print(f"FINAL exact-match accuracy: {final_acc:.4f}  ({total_exact:.0f}/{n_tasks} tasks)")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
