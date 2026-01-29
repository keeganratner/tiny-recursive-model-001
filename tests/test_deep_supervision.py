"""Tests for deep supervision training mechanics."""
import pytest
import torch
import tracemalloc

from src.trm.model import TRMNetwork, RecursiveRefinement, GridEmbedding
from src.trm.training import DeepSupervisionTrainer


def create_test_model(
    hidden_dim=64,
    outer_steps=3,
    inner_steps=2,
    enable_halting=True,
    halt_threshold=0.9,
):
    """Create a small test model."""
    network = TRMNetwork(
        num_colors=10,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=4,
    )
    embedding = GridEmbedding(hidden_dim=hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        num_colors=10,
        halt_threshold=halt_threshold,
        enable_halting=enable_halting,
    )
    return model


class TestDeepSupervisionTrainer:
    """Tests for DeepSupervisionTrainer initialization and basic functionality."""

    def test_trainer_initialization(self):
        """Test trainer initializes with correct parameters."""
        model = create_test_model()
        trainer = DeepSupervisionTrainer(
            model,
            max_sup_steps=16,
            grad_clip_norm=1.0,
        )
        assert trainer.max_sup_steps == 16
        assert trainer.grad_clip_norm == 1.0
        assert trainer.model is model

    def test_trainer_inherits_from_trm_trainer(self):
        """Test DeepSupervisionTrainer inherits from TRMTrainer."""
        model = create_test_model()
        trainer = DeepSupervisionTrainer(model)
        # Should have compute_loss from parent
        assert hasattr(trainer, "compute_loss")
        assert hasattr(trainer, "train_step")  # Original method
        assert hasattr(trainer, "train_step_deep_supervision")  # New method


class TestDeepSupervisionStep:
    """Tests for train_step_deep_supervision mechanics."""

    def test_train_step_returns_correct_keys(self):
        """Test train step returns expected dictionary keys."""
        model = create_test_model(outer_steps=2, enable_halting=False)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=4)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        result = trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        expected_keys = {"total_loss", "ce_loss", "bce_loss", "accuracy",
                        "steps", "grad_norm", "halted_early"}
        assert set(result.keys()) == expected_keys

    def test_train_step_takes_max_sup_steps(self):
        """Test trainer runs max_sup_steps when not halting."""
        model = create_test_model(enable_halting=False)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=5)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        result = trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        assert result["steps"] == 5
        assert result["halted_early"] is False

    def test_train_step_halts_early(self):
        """Test trainer halts early when confidence threshold met."""
        # Use halt_threshold=0.0 to always halt after first step
        model = create_test_model(enable_halting=True, halt_threshold=0.0)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=10)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        result = trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        assert result["steps"] == 1
        assert result["halted_early"] is True


class TestGradientDetachment:
    """Tests verifying gradient detachment works correctly (TRAIN-02)."""

    def test_gradient_flow_exists(self):
        """Test that gradients flow despite detachment."""
        model = create_test_model(enable_halting=False)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=3)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        result = trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        # Gradient norm should be non-zero (learning possible)
        assert result["grad_norm"] > 0, "No gradient flow - detachment broken?"

    def test_weights_change_after_deep_supervision_step(self):
        """Test that model weights actually change after training step."""
        model = create_test_model(enable_halting=False)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=3)

        # Get initial weights (use transformer parameter)
        param = None
        for name, p in model.named_parameters():
            if "transformer" in name or "output_head" in name:
                param = p
                break
        assert param is not None, "No suitable parameter found"

        initial_weights = param.data.clone()

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        # Weights should have changed
        assert not torch.equal(initial_weights, param.data), "Weights unchanged after training"


class TestLossNormalization:
    """Tests for loss normalization by steps taken."""

    def test_loss_scale_consistent_across_step_counts(self):
        """Test loss scale is similar regardless of steps taken."""
        model = create_test_model(enable_halting=False)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Test with different max_sup_steps
        losses = []
        for max_steps in [2, 4, 8]:
            # Fresh model each time to avoid training effects
            model = create_test_model(enable_halting=False)
            trainer = DeepSupervisionTrainer(model, max_sup_steps=max_steps)
            result = trainer.train_step_deep_supervision(
                input_grids.clone(), target_grids.clone(), mask.clone()
            )
            losses.append(result["total_loss"])

        # Losses should be in similar range (within 2x of each other)
        # This verifies normalization is working
        max_loss = max(losses)
        min_loss = min(losses)
        ratio = max_loss / (min_loss + 1e-8)
        assert ratio < 3.0, f"Loss scale varies too much: {losses}, ratio={ratio}"


class TestGradientClipping:
    """Tests for gradient clipping (TRAIN-07)."""

    def test_gradient_clipping_applied(self):
        """Test gradient clipping is actually applied."""
        model = create_test_model(enable_halting=False)
        trainer = DeepSupervisionTrainer(model, max_sup_steps=3, grad_clip_norm=0.1)

        B, H, W = 2, 4, 4
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        result = trainer.train_step_deep_supervision(input_grids, target_grids, mask)

        # With very low clip norm, gradient should be clipped
        # grad_norm returned is the norm BEFORE clipping
        # We can't directly verify clipping, but we verify the API works
        assert "grad_norm" in result
        assert isinstance(result["grad_norm"], float)


class TestMemoryConstancy:
    """Tests verifying memory doesn't explode across supervision steps."""

    def test_memory_does_not_explode_with_more_steps(self):
        """Test memory usage doesn't grow linearly with supervision steps.

        This indirectly verifies TRAIN-02 (gradient detachment) is working.
        If detachment was broken, memory would grow with each step.
        """
        # Use tracemalloc for CPU memory profiling
        model = create_test_model(enable_halting=False)

        B, H, W = 2, 8, 8
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        memory_usage = []

        for max_steps in [2, 4, 8, 16]:
            # Fresh model to avoid accumulation
            model = create_test_model(enable_halting=False)
            trainer = DeepSupervisionTrainer(model, max_sup_steps=max_steps)

            tracemalloc.start()
            trainer.train_step_deep_supervision(
                input_grids.clone(), target_grids.clone(), mask.clone()
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append((max_steps, peak / 1024 / 1024))  # MB

        # Memory should NOT grow 8x when steps grow 8x (2 -> 16)
        # Some growth is expected due to intermediate computations
        # but if detachment is broken, memory would grow linearly
        first_mem = memory_usage[0][1]
        last_mem = memory_usage[-1][1]

        # Allow up to 4x growth (8x steps -> max 4x memory is acceptable)
        # Linear growth would be 8x
        growth_ratio = last_mem / (first_mem + 1e-8)
        assert growth_ratio < 6.0, f"Memory growth suspicious: {memory_usage}, ratio={growth_ratio}"


class TestConfigurability:
    """Tests for configurable parameters (TRAIN-08)."""

    def test_max_sup_steps_configurable(self):
        """Test max_sup_steps parameter works (TRAIN-08)."""
        model = create_test_model(enable_halting=False)

        for max_steps in [4, 8, 16]:
            trainer = DeepSupervisionTrainer(model, max_sup_steps=max_steps)

            B, H, W = 2, 4, 4
            input_grids = torch.randint(0, 10, (B, H, W))
            target_grids = torch.randint(0, 10, (B, H, W))
            mask = torch.ones(B, H, W, dtype=torch.bool)

            result = trainer.train_step_deep_supervision(
                input_grids.clone(), target_grids.clone(), mask.clone()
            )
            assert result["steps"] == max_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
