"""Comprehensive tests that must pass before starting the long training run.

These tests verify:
1. Halting head correctly masks padded tokens
2. EMA is saved and restored via checkpoints
3. total_epochs accumulates across resumed sessions
4. Periodic checkpoint saving at --save-every intervals
5. KeyboardInterrupt causes checkpoint save
6. Separate embedding LR creates two optimizer param groups
7. forward_in_context passes mask to halting head
8. Gradient flows end-to-end (overfit single task)
9. Checkpoint resume produces consistent results
10. Full config runs without OOM (smoke test with real config)

Run with: pytest tests/test_long_run.py -v
All 10 tests must pass before starting the long training run.
"""
import json
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.model.heads import HaltingHead
from src.trm.training.trainer import TRMTrainer
from src.trm.training.deep_supervision import DeepSupervisionTrainer
from src.trm.evaluation.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_periodic_checkpoint,
    _get_ema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model(outer_steps=1, inner_steps=1, hidden_dim=64, num_heads=2, num_layers=1):
    """Create a minimal model for fast tests."""
    network = TRMNetwork(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
    embedding = GridEmbedding(hidden_dim=hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        enable_halting=True,
    )
    return model


def make_tiny_trainer(embed_lr=None, **kwargs):
    """Create a TRMTrainer with tiny model."""
    model = make_tiny_model()
    return TRMTrainer(model=model, embed_lr=embed_lr, **kwargs)


def make_tiny_deep_trainer(embed_lr=None, **kwargs):
    """Create a DeepSupervisionTrainer with tiny model."""
    model = make_tiny_model()
    return DeepSupervisionTrainer(model=model, embed_lr=embed_lr, **kwargs)


def random_grid_batch(B=2, H=4, W=4, device="cpu"):
    return torch.randint(0, 10, (B, H, W), device=device)


# ---------------------------------------------------------------------------
# Test 1: Halting head mask correctness
# ---------------------------------------------------------------------------

class TestHaltingHeadMask:
    """Test 1: HaltingHead correctly ignores padded tokens when mask is given."""

    def test_mask_excludes_padded_positions(self):
        """With mask, only valid tokens contribute to halt confidence."""
        torch.manual_seed(0)
        hidden_dim = 32
        head = HaltingHead(hidden_dim=hidden_dim)

        # Set projection weight to all-positive so large padding pushes confidence high
        head.proj.weight.data.fill_(1.0 / hidden_dim)

        B, seq_len = 2, 10
        # Valid tokens: first 5 (small values ~0); padded: last 5 (large values = 100)
        x = torch.zeros(B, seq_len, hidden_dim)
        x[:, 5:, :] = 100.0  # large padded values

        mask = torch.zeros(B, seq_len, dtype=torch.bool)
        mask[:, :5] = True  # first 5 tokens are valid

        conf_no_mask = head(x, mask=None)
        conf_with_mask = head(x, mask=mask)

        # Without mask: pooled mean includes large padding → large positive proj → conf near 1.0
        assert conf_no_mask.min().item() > 0.99, (
            f"Without mask, large padded values should push confidence near 1.0, got {conf_no_mask}"
        )

        # With mask: pooled mean is only over zeros → proj output = 0 → conf = sigmoid(0) = 0.5
        assert (conf_with_mask - 0.5).abs().max().item() < 0.01, (
            f"With mask over zero tokens, confidence should be ~0.5, got {conf_with_mask}"
        )

        # The two should differ significantly
        assert not torch.allclose(conf_no_mask, conf_with_mask, atol=0.01), (
            "Masked and unmasked confidences must differ when padding has extreme values"
        )

    def test_mask_matches_manual_subset_average(self):
        """Masked pooling must equal manually computing mean over valid positions."""
        torch.manual_seed(1)
        hidden_dim = 16
        head = HaltingHead(hidden_dim=hidden_dim)

        B, seq_len = 3, 8
        x = torch.randn(B, seq_len, hidden_dim)

        # Each batch item has a different number of valid tokens
        valid_counts = [3, 5, 8]
        mask = torch.zeros(B, seq_len, dtype=torch.bool)
        for i, n in enumerate(valid_counts):
            mask[i, :n] = True

        conf_masked = head(x, mask=mask)

        # Compute expected result manually
        expected_pooled = torch.zeros(B, hidden_dim)
        for i, n in enumerate(valid_counts):
            expected_pooled[i] = x[i, :n].mean(dim=0)
        expected_conf = torch.sigmoid(head.proj(expected_pooled)).squeeze(-1)

        assert torch.allclose(conf_masked, expected_conf, atol=1e-5), (
            "Masked confidence must match manually computed masked mean"
        )

    def test_no_mask_unchanged(self):
        """Without mask, behavior is unchanged (global average pooling)."""
        torch.manual_seed(2)
        hidden_dim = 16
        head = HaltingHead(hidden_dim=hidden_dim)

        x = torch.randn(2, 6, hidden_dim)
        conf_none = head(x, mask=None)
        conf_old = torch.sigmoid(head.proj(x.mean(dim=1))).squeeze(-1)

        assert torch.allclose(conf_none, conf_old, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: EMA saved and restored correctly
# ---------------------------------------------------------------------------

class TestEMASaveRestore:
    """Test 2: EMA model is correctly saved and restored via checkpoints."""

    def test_ema_found_by_get_ema(self):
        """_get_ema correctly finds trainer.ema (DeepSupervisionTrainer)."""
        trainer = make_tiny_deep_trainer()
        ema = _get_ema(trainer)
        assert ema is not None, "_get_ema should find trainer.ema"
        assert ema is trainer.ema

    def test_ema_not_found_for_base_trainer(self):
        """_get_ema returns None for TRMTrainer (no EMA)."""
        trainer = make_tiny_trainer()
        ema = _get_ema(trainer)
        assert ema is None, "_get_ema should return None for TRMTrainer"

    def test_ema_saved_and_restored(self, tmp_path):
        """EMA weights survive a checkpoint roundtrip."""
        torch.manual_seed(42)
        trainer = make_tiny_deep_trainer()

        # Run a few steps to diverge EMA from initial weights
        for _ in range(5):
            x = random_grid_batch()
            trainer.train_step_deep_supervision(x, x, torch.ones_like(x, dtype=torch.bool))

        # Record EMA module weights
        ema_before = {
            k: v.clone() for k, v in trainer.ema.module.state_dict().items()
        }

        # Save checkpoint with total_epochs=5
        ckpt_path = tmp_path / "ema_test.pt"
        save_checkpoint(trainer, ckpt_path, epoch=4, step=0, total_epochs=5)

        # Create fresh trainer and load
        trainer2 = make_tiny_deep_trainer()
        info = load_checkpoint(trainer2, ckpt_path)

        # EMA weights must match
        ema_after = trainer2.ema.module.state_dict()
        for k in ema_before:
            assert torch.allclose(ema_before[k].float(), ema_after[k].float(), atol=1e-6), (
                f"EMA param '{k}' mismatch after checkpoint restore"
            )

        assert info["total_epochs"] == 5
        assert info["epoch"] == 4

    def test_ema_state_in_checkpoint_file(self, tmp_path):
        """Checkpoint file contains 'ema_state' key for DeepSupervisionTrainer."""
        trainer = make_tiny_deep_trainer()
        ckpt_path = tmp_path / "ema_key_test.pt"
        save_checkpoint(trainer, ckpt_path, epoch=0, step=0)

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "ema_state" in ckpt, "Checkpoint must contain ema_state for EMA trainer"


# ---------------------------------------------------------------------------
# Test 3: total_epochs accumulates across resumed sessions
# ---------------------------------------------------------------------------

class TestTotalEpochsAccumulation:
    """Test 3: total_epochs counter accumulates correctly across resumes."""

    def test_total_epochs_saved_in_checkpoint(self, tmp_path):
        """total_epochs is written to checkpoint."""
        trainer = make_tiny_deep_trainer()
        ckpt_path = tmp_path / "epoch_test.pt"
        save_checkpoint(trainer, ckpt_path, epoch=2, step=0, total_epochs=3)

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["total_epochs"] == 3

    def test_total_epochs_loaded_from_checkpoint(self, tmp_path):
        """load_checkpoint returns correct total_epochs."""
        trainer = make_tiny_deep_trainer()
        ckpt_path = tmp_path / "epoch_load_test.pt"
        save_checkpoint(trainer, ckpt_path, epoch=2, step=0, total_epochs=3)

        info = load_checkpoint(trainer, ckpt_path)
        assert info["total_epochs"] == 3

    def test_total_epochs_accumulates_across_resumes(self, tmp_path):
        """Simulated two-session run: total_epochs = session1 + session2."""
        trainer = make_tiny_deep_trainer()

        # Session 1: train 3 epochs
        ckpt1 = tmp_path / "session1.pt"
        save_checkpoint(trainer, ckpt1, epoch=2, step=0, total_epochs=3)

        # Simulate resume: load, then run 3 more epochs
        trainer2 = make_tiny_deep_trainer()
        info = load_checkpoint(trainer2, ckpt1)
        assert info["total_epochs"] == 3

        # Session 2 adds 3 more → total = 6
        total_after_session2 = info["total_epochs"] + 3
        ckpt2 = tmp_path / "session2.pt"
        save_checkpoint(trainer2, ckpt2, epoch=5, step=0, total_epochs=total_after_session2)

        info2 = load_checkpoint(trainer2, ckpt2)
        assert info2["total_epochs"] == 6, (
            f"Expected total_epochs=6 after two 3-epoch sessions, got {info2['total_epochs']}"
        )

    def test_total_epochs_defaults_to_epoch_plus_1(self, tmp_path):
        """When total_epochs not given, it defaults to epoch+1."""
        trainer = make_tiny_deep_trainer()
        ckpt_path = tmp_path / "default_epoch.pt"
        save_checkpoint(trainer, ckpt_path, epoch=4, step=0)  # no total_epochs

        info = load_checkpoint(trainer, ckpt_path)
        assert info["total_epochs"] == 5  # epoch+1


# ---------------------------------------------------------------------------
# Test 4: Periodic checkpoint saving
# ---------------------------------------------------------------------------

class TestPeriodicCheckpointing:
    """Test 4: save_periodic_checkpoint creates correctly named files."""

    def test_periodic_checkpoint_filename(self, tmp_path):
        """Checkpoint filename contains zero-padded total_epochs."""
        trainer = make_tiny_deep_trainer()
        path = save_periodic_checkpoint(trainer, tmp_path, epoch=1, total_epochs=2)

        assert path.name == "checkpoint_epoch_000002.pt", (
            f"Expected checkpoint_epoch_000002.pt, got {path.name}"
        )
        assert path.exists()

    def test_periodic_checkpoint_sequence(self, tmp_path):
        """Saving at epochs 100, 200, 300 creates three files."""
        trainer = make_tiny_deep_trainer()
        for te in [100, 200, 300]:
            save_periodic_checkpoint(trainer, tmp_path, epoch=te - 1, total_epochs=te)

        for te in [100, 200, 300]:
            expected = tmp_path / f"checkpoint_epoch_{te:06d}.pt"
            assert expected.exists(), f"Missing {expected.name}"

    def test_periodic_checkpoint_loadable(self, tmp_path):
        """A periodic checkpoint can be loaded back."""
        trainer = make_tiny_deep_trainer()
        path = save_periodic_checkpoint(trainer, tmp_path, epoch=9, total_epochs=10)

        trainer2 = make_tiny_deep_trainer()
        info = load_checkpoint(trainer2, path)
        assert info["total_epochs"] == 10
        assert info["epoch"] == 9

    def test_periodic_checkpoint_stores_best_accuracy(self, tmp_path):
        """Periodic checkpoint records best_accuracy if provided."""
        trainer = make_tiny_deep_trainer()
        path = save_periodic_checkpoint(
            trainer, tmp_path, epoch=4, total_epochs=5, best_accuracy=0.42
        )
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["best_accuracy"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Test 5: KeyboardInterrupt saves checkpoint
# ---------------------------------------------------------------------------

class TestKeyboardInterruptSave:
    """Test 5: save_periodic_checkpoint is callable in KeyboardInterrupt handler."""

    def test_interrupt_handler_saves_checkpoint(self, tmp_path):
        """Simulating the interrupt handler: checkpoint is saved before exit."""
        trainer = make_tiny_deep_trainer()
        total_epochs_run = 2

        saved_path = None
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            saved_path = save_periodic_checkpoint(
                trainer, tmp_path,
                epoch=total_epochs_run - 1,
                total_epochs=total_epochs_run,
            )

        assert saved_path is not None
        assert saved_path.exists(), "Interrupt handler must save a checkpoint"
        info = load_checkpoint(trainer, saved_path)
        assert info["total_epochs"] == total_epochs_run

    def test_interrupt_checkpoint_preserves_model(self, tmp_path):
        """Model weights are preserved through interrupt-save-reload cycle."""
        torch.manual_seed(99)
        trainer = make_tiny_deep_trainer()

        # Train so weights are non-random
        x = random_grid_batch()
        for _ in range(3):
            trainer.train_step_deep_supervision(x, x, torch.ones_like(x, dtype=torch.bool))

        original_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            saved_path = save_periodic_checkpoint(trainer, tmp_path, epoch=2, total_epochs=3)

        trainer2 = make_tiny_deep_trainer()
        load_checkpoint(trainer2, saved_path)

        for k in original_weights:
            assert torch.allclose(original_weights[k].float(),
                                  trainer2.model.state_dict()[k].float(), atol=1e-6), (
                f"Model weight '{k}' not preserved through interrupt save"
            )


# ---------------------------------------------------------------------------
# Test 6: Separate embedding LR
# ---------------------------------------------------------------------------

class TestSeparateEmbedLR:
    """Test 6: Optimizer has two param groups with different LRs for embeddings."""

    def test_two_param_groups(self):
        """TRMTrainer creates exactly 2 optimizer param groups."""
        trainer = make_tiny_trainer(learning_rate=1e-4, embed_lr=1e-2)
        assert len(trainer.optimizer.param_groups) == 2, (
            "Optimizer must have exactly 2 param groups (embed + other)"
        )

    def test_embed_group_has_higher_lr(self):
        """Embedding param group has embed_lr; other group has learning_rate."""
        trainer = make_tiny_trainer(learning_rate=1e-4, embed_lr=1e-2)
        lrs = {pg["lr"] for pg in trainer.optimizer.param_groups}
        assert 1e-2 in lrs, "embed_lr=1e-2 must appear in param groups"
        assert 1e-4 in lrs, "learning_rate=1e-4 must appear in param groups"

    def test_embed_params_in_high_lr_group(self):
        """Embedding parameters are assigned to the high-LR group."""
        model = make_tiny_model()
        trainer = TRMTrainer(model=model, learning_rate=1e-4, embed_lr=1e-2)

        # Collect embedding parameter ids
        embed_param_ids = set()
        for m in [model.embedding, model.role_embedding]:
            for p in m.parameters():
                embed_param_ids.add(id(p))

        # Find the high-LR group
        high_lr_group = max(trainer.optimizer.param_groups, key=lambda g: g["lr"])
        high_lr_param_ids = {id(p) for p in high_lr_group["params"]}

        assert embed_param_ids == high_lr_param_ids, (
            "All embedding params must be in the high-LR group"
        )

    def test_all_params_covered(self):
        """Every model parameter appears in exactly one optimizer param group."""
        model = make_tiny_model()
        trainer = TRMTrainer(model=model, learning_rate=1e-4, embed_lr=1e-2)

        all_model_ids = {id(p) for p in model.parameters()}
        optimizer_ids = set()
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                optimizer_ids.add(id(p))

        assert all_model_ids == optimizer_ids, (
            "Every model parameter must appear in an optimizer param group"
        )

    def test_default_embed_lr_equals_learning_rate(self):
        """When embed_lr=None, both groups get the same LR."""
        trainer = make_tiny_trainer(learning_rate=1e-4, embed_lr=None)
        for group in trainer.optimizer.param_groups:
            assert group["lr"] == pytest.approx(1e-4)

    def test_deep_supervision_trainer_embed_lr(self):
        """DeepSupervisionTrainer forwards embed_lr to optimizer."""
        trainer = make_tiny_deep_trainer(learning_rate=1e-4, embed_lr=1e-2)
        lrs = {pg["lr"] for pg in trainer.optimizer.param_groups}
        assert 1e-2 in lrs
        assert 1e-4 in lrs


# ---------------------------------------------------------------------------
# Test 7: forward_in_context halt uses attn_mask
# ---------------------------------------------------------------------------

class TestForwardInContextHaltMask:
    """Test 7: Halting head in forward_in_context uses attn_mask to exclude padding."""

    def test_padding_content_does_not_affect_halt_confidence(self):
        """Same task with different padding values → same halt_confidence."""
        torch.manual_seed(7)
        model = make_tiny_model(outer_steps=1, inner_steps=1)
        model.eval()

        B = 1
        max_demos = 4   # 3 real + 1 padded slot
        H_d, W_d = 3, 3
        H_t, W_t = 3, 3

        demo_inputs  = torch.randint(0, 10, (B, max_demos, H_d, W_d))
        demo_outputs = torch.randint(0, 10, (B, max_demos, H_d, W_d))
        test_input   = torch.randint(0, 10, (B, H_t, W_t))
        num_demos    = torch.tensor([3])  # only 3 real demos

        # Run 1: padded slot has zeros
        demo_inputs_v1 = demo_inputs.clone()
        demo_inputs_v1[:, 3, :, :] = 0
        demo_outputs_v1 = demo_outputs.clone()
        demo_outputs_v1[:, 3, :, :] = 0

        # Run 2: padded slot has value 9 (different from zeros)
        demo_inputs_v2 = demo_inputs.clone()
        demo_inputs_v2[:, 3, :, :] = 9
        demo_outputs_v2 = demo_outputs.clone()
        demo_outputs_v2[:, 3, :, :] = 9

        with torch.no_grad():
            out1 = model.forward_in_context(demo_inputs_v1, demo_outputs_v1, num_demos, test_input)
            out2 = model.forward_in_context(demo_inputs_v2, demo_outputs_v2, num_demos, test_input)

        hc1 = out1["halt_confidence"]
        hc2 = out2["halt_confidence"]

        assert torch.allclose(hc1, hc2, atol=1e-4), (
            f"Halt confidence should be identical regardless of padding content. "
            f"Got {hc1.item():.6f} vs {hc2.item():.6f} (diff={abs(hc1 - hc2).item():.2e})"
        )

    def test_halt_confidence_shape(self):
        """forward_in_context returns halt_confidence of shape (B,)."""
        model = make_tiny_model(outer_steps=1, inner_steps=1)
        model.eval()
        B = 3
        demo_inputs  = torch.randint(0, 10, (B, 2, 4, 4))
        demo_outputs = torch.randint(0, 10, (B, 2, 4, 4))
        test_input   = torch.randint(0, 10, (B, 4, 4))
        num_demos    = torch.tensor([2, 2, 2])

        with torch.no_grad():
            out = model.forward_in_context(demo_inputs, demo_outputs, num_demos, test_input)

        assert out["halt_confidence"].shape == (B,), (
            f"halt_confidence shape should be ({B},), got {out['halt_confidence'].shape}"
        )


# ---------------------------------------------------------------------------
# Test 8: Gradient flow — overfit single task
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Test 8: Model can overfit a single synthetic task (identity transform)."""

    def test_overfit_identity_task(self):
        """Train on identity transform (output=input) for 100 steps, loss must drop."""
        torch.manual_seed(8)
        model = make_tiny_model(outer_steps=1, inner_steps=1, hidden_dim=64)
        trainer = TRMTrainer(model=model, learning_rate=1e-3, embed_lr=1e-2)

        B, H, W = 2, 3, 3
        x = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        initial_loss = None
        for step in range(100):
            result = trainer.train_step(x, x, mask)
            if initial_loss is None:
                initial_loss = result["total_loss"]

        final_loss = result["total_loss"]

        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
        assert final_loss < 2.0, (
            f"Final loss should be well below random baseline (~2.3): {final_loss:.4f}"
        )

    def test_in_context_grad_flow(self):
        """In-context training step produces finite, decreasing loss."""
        torch.manual_seed(88)
        model = make_tiny_model(outer_steps=1, inner_steps=1, hidden_dim=64)
        trainer = DeepSupervisionTrainer(model=model, learning_rate=1e-3, embed_lr=1e-2)

        B = 1
        demo_inputs  = torch.randint(0, 10, (B, 2, 3, 3))
        demo_outputs = demo_inputs.clone()  # identity demos
        test_input   = torch.randint(0, 10, (B, 3, 3))
        target       = test_input.clone()
        mask         = torch.ones(B, 3, 3, dtype=torch.bool)
        num_demos    = torch.tensor([2])

        losses = []
        for _ in range(30):
            result = trainer.train_step_in_context(
                demo_inputs, demo_outputs, num_demos, test_input, target, mask
            )
            losses.append(result["total_loss"])
            assert torch.isfinite(torch.tensor(result["total_loss"])), "Loss must be finite"

        assert losses[-1] < losses[0], (
            f"In-context loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 9: Checkpoint resume produces consistent results
# ---------------------------------------------------------------------------

class TestResumeConsistency:
    """Test 9: Resuming from checkpoint produces the same loss trajectory."""

    def test_resume_model_matches(self, tmp_path):
        """After save + load, next training step produces same loss."""
        torch.manual_seed(9)
        trainer_a = make_tiny_deep_trainer()

        x = random_grid_batch(B=2, H=4, W=4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)

        # Run 3 steps
        for _ in range(3):
            trainer_a.train_step_deep_supervision(x, x, mask)

        # Save
        ckpt = tmp_path / "resume_test.pt"
        save_checkpoint(trainer_a, ckpt, epoch=2, step=0, total_epochs=3)

        # Fresh trainer, load, run one more step
        torch.manual_seed(9)
        trainer_b = make_tiny_deep_trainer()
        load_checkpoint(trainer_b, ckpt)

        result_a = trainer_a.train_step_deep_supervision(x, x, mask)
        result_b = trainer_b.train_step_deep_supervision(x, x, mask)

        assert abs(result_a["total_loss"] - result_b["total_loss"]) < 1e-4, (
            f"Resumed trainer must produce same loss: "
            f"{result_a['total_loss']:.6f} vs {result_b['total_loss']:.6f}"
        )

    def test_total_epochs_preserved_on_resume(self, tmp_path):
        """total_epochs from checkpoint is retrieved and can be incremented."""
        trainer = make_tiny_deep_trainer()
        ckpt = tmp_path / "resume_epochs.pt"
        save_checkpoint(trainer, ckpt, epoch=49, step=0, total_epochs=50)

        trainer2 = make_tiny_deep_trainer()
        info = load_checkpoint(trainer2, ckpt)

        # Simulate: resume starts from total_epochs_run = 50
        total_epochs_run = info["total_epochs"]
        assert total_epochs_run == 50

        # After 10 more epochs
        total_epochs_run += 10
        save_checkpoint(trainer2, ckpt, epoch=59, step=0, total_epochs=total_epochs_run)

        info2 = load_checkpoint(trainer2, ckpt)
        assert info2["total_epochs"] == 60


# ---------------------------------------------------------------------------
# Test 10: Full config smoke test (no OOM, correct shapes, ~7M params)
# ---------------------------------------------------------------------------

class TestFullConfigSmoke:
    """Test 10: Full paper config (h=512, T=3, n=6) runs without OOM."""

    def test_full_config_forward_pass(self):
        """Full model (7M params) runs a forward pass with small grids."""
        from omegaconf import OmegaConf

        config_path = project_root / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)

        device = torch.device("cpu")  # always CPU for test portability

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
        ).to(device)

        # Check parameter count is in the expected range for full config (~7-12M)
        param_count = sum(p.numel() for p in model.parameters())
        assert 5_000_000 < param_count < 15_000_000, (
            f"Expected ~7-12M parameters for full config, got {param_count:,}"
        )

        # Forward pass with tiny grids (not full ARC size) to keep test fast
        B, H, W = 1, 5, 5
        x = torch.randint(0, 10, (B, H, W), device=device)
        with torch.no_grad():
            out = model(x)

        logits = out["logits"]
        assert logits.shape == (B, H, W, 10), f"Logits shape {logits.shape}"
        assert torch.isfinite(logits).all(), "Logits must be finite"
        assert out["halt_confidence"].shape == (B,)

    def test_full_config_training_step(self):
        """Full model can perform a single training step with batch_size=1."""
        from omegaconf import OmegaConf

        config_path = project_root / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)

        device = torch.device("cpu")

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
        ).to(device)

        trainer = TRMTrainer(
            model=model,
            learning_rate=cfg.training.learning_rate,
            embed_lr=1e-2,
            weight_decay=cfg.training.weight_decay,
        )

        B, H, W = 1, 5, 5
        x = torch.randint(0, 10, (B, H, W), device=device)
        mask = torch.ones(B, H, W, dtype=torch.bool, device=device)

        result = trainer.train_step(x, x, mask)

        assert torch.isfinite(torch.tensor(result["total_loss"])), "Loss must be finite"
        assert result["total_loss"] > 0, "Loss must be positive"

    def test_real_arc_task_forward(self):
        """Load a real ARC task and run a forward pass (validates data pipeline)."""
        from omegaconf import OmegaConf
        from src.trm.data import ARCDataset, arc_collate_fn
        from torch.utils.data import DataLoader

        config_path = project_root / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)

        data_dir = project_root / "data"
        if not (data_dir / "training").exists():
            pytest.skip("data/training/ not found; skipping real ARC task test")

        dataset = ARCDataset(data_dir=str(data_dir), split="training")
        loader = DataLoader(dataset, batch_size=1, collate_fn=arc_collate_fn)
        batch = next(iter(loader))

        device = torch.device("cpu")

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
        ).to(device)
        model.eval()

        # Use train pairs as demo context, predict test output
        demo_in  = batch["train_inputs"].to(device)
        demo_out = batch["train_outputs"].to(device)
        n_demos  = batch["num_train_pairs"].to(device)

        test_in  = batch["test_inputs"][:, 0].to(device)  # first test pair
        target   = batch["test_outputs"][:, 0].to(device)

        with torch.no_grad():
            out = model.forward_in_context(demo_in, demo_out, n_demos, test_in)

        logits = out["logits"]
        assert logits.shape[-1] == 10, "Output must have 10 color classes"
        assert torch.isfinite(logits).all(), "Logits must be finite"
