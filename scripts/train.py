"""Training script for TRM on ARC-AGI dataset."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader

from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.data import ARCDataset, AugmentedARCDataset, arc_collate_fn, PAD_VALUE
from src.trm.training import TRMTrainer, DeepSupervisionTrainer
from src.trm.evaluation import compute_exact_match_accuracy, save_checkpoint, load_checkpoint, BestModelTracker


def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """Set up logger that writes to both terminal and a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    logger = logging.getLogger("trm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Terminal handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return logger


def load_config() -> DictConfig:
    """Load config from configs/config.yaml (Hydra-compatible format)."""
    config_path = project_root / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg


def create_model(cfg: DictConfig) -> RecursiveRefinement:
    """Create TRM model from config."""
    network = TRMNetwork(
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
    )
    embedding = GridEmbedding(hidden_dim=cfg.model.hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=cfg.recursion.outer_steps,
        inner_steps=cfg.recursion.inner_steps,
        halt_threshold=cfg.recursion.halt_threshold,
        enable_halting=True,
    )
    return model


def validate_epoch(model, dataloader, device):
    """Run validation on evaluation split, return exact-match accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            test_inputs = batch["test_inputs"]
            test_outputs = batch["test_outputs"]
            test_masks = batch["test_output_masks"]
            num_pairs = batch["num_test_pairs"]

            B, max_pairs, H, W = test_inputs.shape

            for pair_idx in range(max_pairs):
                valid_tasks = num_pairs > pair_idx
                if not valid_tasks.any():
                    continue

                input_grid = test_inputs[:, pair_idx, :, :][valid_tasks].to(device)
                target_grid = test_outputs[:, pair_idx, :, :][valid_tasks].to(device)
                mask = test_masks[:, pair_idx, :, :][valid_tasks].to(device)

                if not mask.any():
                    continue

                output = model(input_grid)
                logits = output["logits"]

                accuracy = compute_exact_match_accuracy(logits, target_grid, mask)
                total_correct += accuracy.sum().item()
                total_samples += accuracy.numel()

    return total_correct / max(total_samples, 1)


class EarlyStopping:
    """Early stopping with patience tracking."""
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(trainer, dataloader, epoch, device, use_deep_supervision=False):
    """Run one training epoch."""
    total_loss = 0.0
    total_ce = 0.0
    total_bce = 0.0
    total_acc = 0.0
    total_steps = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        train_inputs = batch["train_inputs"]
        train_outputs = batch["train_outputs"]
        train_masks = batch["train_output_masks"]
        num_pairs = batch["num_train_pairs"]

        B, max_pairs, H, W = train_inputs.shape

        for pair_idx in range(max_pairs):
            input_grid = train_inputs[:, pair_idx, :, :]
            target_grid = train_outputs[:, pair_idx, :, :]
            mask = train_masks[:, pair_idx, :, :]

            valid_tasks = num_pairs > pair_idx
            if not valid_tasks.any():
                continue

            input_grid = input_grid[valid_tasks].to(device)
            target_grid = target_grid[valid_tasks].to(device)
            mask = mask[valid_tasks].to(device)

            if not mask.any():
                continue

            if use_deep_supervision:
                result = trainer.train_step_deep_supervision(input_grid, target_grid, mask)
            else:
                result = trainer.train_step(input_grid, target_grid, mask)

            total_loss += result["total_loss"]
            total_ce += result["ce_loss"]
            total_bce += result["bce_loss"]
            total_acc += result["accuracy"]
            if use_deep_supervision:
                total_steps += result.get("steps", 0)
            num_batches += 1

    if num_batches == 0:
        return {"loss": 0, "ce": 0, "bce": 0, "acc": 0, "steps": 0}

    metrics = {
        "loss": total_loss / num_batches,
        "ce": total_ce / num_batches,
        "bce": total_bce / num_batches,
        "acc": total_acc / num_batches,
    }
    if use_deep_supervision:
        metrics["steps"] = total_steps / num_batches
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="TRM Training Script")
    parser.add_argument("--fast", action="store_true", help="Use smaller model for quick testing")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--deep-supervision", action="store_true", help="Use deep supervision training mode")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (D8 + color permutation)")
    parser.add_argument("--no-d8", action="store_true", help="Disable D8 geometric augmentation (only with --augment)")
    parser.add_argument("--no-color", action="store_true", help="Disable color permutation augmentation (only with --augment)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation (train only)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs/)")
    args = parser.parse_args()

    # Build run name and set up logging
    mode = "deep_sup" if args.deep_supervision else "basic"
    aug = "_aug" if args.augment else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trm_{mode}{aug}_{timestamp}"
    log = setup_logging(Path(args.log_dir), run_name)

    # Load config
    cfg = load_config()

    if args.fast:
        cfg.model.hidden_dim = 64
        cfg.model.num_layers = 1
        cfg.model.num_heads = 2
        cfg.recursion.outer_steps = 1
        cfg.recursion.inner_steps = 1
        cfg.data.batch_size = 4

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    log.info("=" * 60)
    log.info(f"TRM Training - {'Deep Supervision' if args.deep_supervision else 'Basic (Terminal Supervision)'}")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Mode: {'fast' if args.fast else 'full'}")
    log.info(f"Augmentation: {args.augment}")
    if args.augment:
        log.info(f"  D8: {not args.no_d8}, Color: {not args.no_color}")
    log.info(f"Deep supervision: {args.deep_supervision}")
    if args.deep_supervision:
        log.info(f"  Max sup steps: {cfg.training.get('max_sup_steps', 16)}")
        log.info(f"  Grad clip norm: {cfg.training.get('grad_clip_norm', 1.0)}")

    log.info("Creating model...")
    model = create_model(cfg)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {param_count:,}")
    log.info(f"Hidden dim: {cfg.model.hidden_dim}")
    log.info(f"Recursion: T={cfg.recursion.outer_steps}, n={cfg.recursion.inner_steps}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.deep_supervision:
        trainer = DeepSupervisionTrainer(
            model=model,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            beta1=cfg.training.beta1,
            beta2=cfg.training.beta2,
            max_sup_steps=cfg.training.get("max_sup_steps", 16),
            grad_clip_norm=cfg.training.get("grad_clip_norm", 1.0),
            ema_decay=cfg.training.ema_decay,
        )
    else:
        trainer = TRMTrainer(
            model=model,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            beta1=cfg.training.beta1,
            beta2=cfg.training.beta2,
        )

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(trainer, args.resume, map_location=device)
        start_epoch = checkpoint_info["epoch"] + 1
        best_val_acc = checkpoint_info.get("best_accuracy", 0.0)
        log.info(f"Resuming from epoch {start_epoch}, best accuracy: {best_val_acc:.4f}")

    log.info("Loading ARC-AGI dataset...")
    data_dir = project_root / "data"

    if args.augment:
        enable_d8 = not args.no_d8
        enable_color = not args.no_color
        train_dataset = AugmentedARCDataset(
            data_dir=str(data_dir),
            split="training",
            enable_d8=enable_d8,
            enable_color=enable_color,
        )
        aug_multiplier = train_dataset.pipeline.get_effective_multiplier()
        log.info(f"Augmentation: D8={enable_d8}, Color={enable_color} ({aug_multiplier}x effective)")
    else:
        train_dataset = ARCDataset(data_dir=str(data_dir), split="training")
    log.info(f"Training tasks: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=arc_collate_fn,
        num_workers=0,
    )

    val_dataloader = None
    if not args.no_validate:
        val_dataset = ARCDataset(data_dir=str(data_dir), split="evaluation")
        log.info(f"Validation tasks: {len(val_dataset)}")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            collate_fn=arc_collate_fn,
            num_workers=0,
        )

    best_model_tracker = None
    early_stopping = None
    if val_dataloader is not None:
        best_model_tracker = BestModelTracker(
            save_dir=checkpoint_dir,
            filename="best_model.pt",
            min_delta=0.0,
        )
        if args.resume:
            best_model_tracker.best_accuracy = best_val_acc
        early_stopping = EarlyStopping(patience=args.patience)

    num_epochs = args.epochs
    log.info(f"Training for {num_epochs} epochs (batch_size={cfg.data.batch_size})...")
    log.info(f"Validation: {'enabled (patience=' + str(args.patience) + ')' if val_dataloader else 'disabled'}")
    log.info("-" * 60)

    losses = []
    for epoch in range(start_epoch, start_epoch + num_epochs):
        metrics = train_epoch(trainer, train_dataloader, epoch, device, use_deep_supervision=args.deep_supervision)
        losses.append(metrics["loss"])

        if args.deep_supervision:
            msg = (
                f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                f"loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                f"bce={metrics['bce']:.4f} acc={metrics['acc']:.2%} "
                f"steps={metrics.get('steps', 0):.1f}"
            )
        else:
            msg = (
                f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                f"loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                f"bce={metrics['bce']:.4f} acc={metrics['acc']:.2%}"
            )

        if val_dataloader is not None:
            val_model = trainer.get_ema_model() if hasattr(trainer, 'get_ema_model') else trainer.model
            val_acc = validate_epoch(val_model, val_dataloader, device)

            is_best = best_model_tracker.update(
                trainer=trainer,
                accuracy=val_acc,
                epoch=epoch,
                step=0,
            )

            msg += f" | val_acc={val_acc:.4f}"
            if is_best:
                msg += " *"

            log.info(msg)

            if early_stopping(val_acc):
                log.info(f"Early stopping after {epoch+1} epochs (best val_acc={best_model_tracker.get_best_accuracy():.4f})")
                break
        else:
            log.info(msg)

    log.info("-" * 60)

    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            log.info(f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
        else:
            log.info(f"WARNING: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    if best_model_tracker is not None:
        log.info(f"Best val_acc={best_model_tracker.get_best_accuracy():.4f} saved to {checkpoint_dir / 'best_model.pt'}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
