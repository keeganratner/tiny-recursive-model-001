"""Tests for visualization modules."""
import pytest
import tempfile
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.trm.visualization import (
    IterationHistory,
    IterationHistoryCapture,
    IterationTimelineRenderer,
    render_iteration_timeline,
    ARC_COLORMAP,
)
from src.trm.model.recursive import RecursiveRefinement
from src.trm.model.network import TRMNetwork
from src.trm.model.embedding import GridEmbedding


@pytest.fixture
def minimal_model():
    """Create minimal RecursiveRefinement for testing."""
    # Small dimensions for fast tests
    hidden_dim = 64
    num_layers = 1
    num_heads = 2
    num_colors = 10
    outer_steps = 2
    inner_steps = 2

    # Create components
    embedding = GridEmbedding(hidden_dim=hidden_dim)
    network = TRMNetwork(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_colors=num_colors,
    )

    # Create refinement module
    refinement = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        num_colors=num_colors,
        halt_threshold=0.9,
        enable_halting=False,  # Disable halting for predictable test behavior
    )

    return refinement


@pytest.fixture
def sample_grids():
    """Create sample grids for testing."""
    # Create simple 5x5 grids with random ARC colors
    torch.manual_seed(42)
    input_grid = torch.randint(0, 10, (5, 5))
    target_grid = torch.randint(0, 10, (5, 5))

    return input_grid, target_grid


class TestIterationHistoryCapture:
    """Tests for IterationHistoryCapture class."""

    def test_capture_returns_history(self, minimal_model, sample_grids):
        """Verify capture() returns IterationHistory."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        assert isinstance(history, IterationHistory)

    def test_history_contains_all_iterations(self, minimal_model, sample_grids):
        """Check iteration_grids length matches outer_steps."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        # Should have one grid per outer iteration
        assert len(history.iteration_grids) == minimal_model.outer_steps

    def test_input_preserved(self, minimal_model, sample_grids):
        """Verify input_grid matches original input."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        # Input should be preserved exactly
        assert torch.equal(history.input_grid, input_grid)

    def test_predictions_discrete(self, minimal_model, sample_grids):
        """Confirm iteration_grids contain values 0-9 (argmax applied)."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        # All predictions should be discrete ARC colors 0-9
        for grid in history.iteration_grids:
            assert grid.min() >= 0
            assert grid.max() <= 9

    def test_confidences_are_floats(self, minimal_model, sample_grids):
        """Verify halt_confidences are Python floats."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        # All confidences should be Python floats
        for conf in history.halt_confidences:
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0

    def test_final_prediction_matches_last_iteration(self, minimal_model, sample_grids):
        """Verify final_prediction equals last iteration_grid."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture(input_grid)

        # Final prediction should match last iteration
        assert torch.equal(history.final_prediction, history.iteration_grids[-1])

    def test_batch_size_one_accepted(self, minimal_model, sample_grids):
        """Test that batch_size=1 input works."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        # Add batch dimension
        batched_input = input_grid.unsqueeze(0)

        history = capture.capture(batched_input)

        assert isinstance(history, IterationHistory)

    def test_batch_size_greater_than_one_raises(self, minimal_model, sample_grids):
        """Test that batch_size>1 raises ValueError."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        # Create batch of size 2
        batched_input = input_grid.unsqueeze(0).repeat(2, 1, 1)

        with pytest.raises(ValueError, match="batch_size=1"):
            capture.capture(batched_input)


class TestIterationTimelineRenderer:
    """Tests for IterationTimelineRenderer class."""

    def test_render_returns_figure(self, minimal_model, sample_grids):
        """Verify render_timeline returns matplotlib Figure."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_without_target(self, minimal_model, sample_grids):
        """Timeline renders with just input and iterations."""
        input_grid, _ = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid=None)

        # Should have input + iteration rows
        expected_rows = 1 + len(history.iteration_grids)
        assert len(fig.axes) == expected_rows
        plt.close(fig)

    def test_render_with_target(self, minimal_model, sample_grids):
        """Timeline includes target row when provided."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid=target_grid)

        # Should have input + iterations + target rows
        expected_rows = 1 + len(history.iteration_grids) + 1
        assert len(fig.axes) == expected_rows
        plt.close(fig)

    def test_colormap_has_10_colors(self):
        """Verify ARC_COLORMAP covers 0-9."""
        # ARC_COLORMAP should have exactly 10 colors
        assert ARC_COLORMAP.N == 10

    def test_figure_dimensions(self, minimal_model, sample_grids):
        """Check figure height scales with number of rows."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()

        # Render without target
        fig1 = renderer.render_timeline(history, target_grid=None)
        height1 = fig1.get_figheight()

        # Render with target
        fig2 = renderer.render_timeline(history, target_grid=target_grid)
        height2 = fig2.get_figheight()

        # Figure with target should be taller
        assert height2 > height1

        plt.close(fig1)
        plt.close(fig2)

    def test_render_with_diff_overlay(self, minimal_model, sample_grids):
        """Test that diff overlay doesn't crash."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid=target_grid, show_diff=True)

        # Should render without errors
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestRenderIterationTimeline:
    """Tests for render_iteration_timeline convenience function."""

    def test_returns_figure(self, minimal_model, sample_grids):
        """Verify function returns Figure."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        fig = render_iteration_timeline(history, target_grid)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, minimal_model, sample_grids):
        """Verify output_path creates image file."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.png"

            fig = render_iteration_timeline(
                history, target_grid, output_path=str(output_path)
            )

            # File should exist
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            plt.close(fig)

    def test_returns_figure_even_with_save(self, minimal_model, sample_grids):
        """Even with output_path, should return Figure."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.png"

            fig = render_iteration_timeline(
                history, target_grid, output_path=str(output_path)
            )

            # Should return figure even when saving
            assert isinstance(fig, plt.Figure)

            plt.close(fig)
