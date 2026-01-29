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
    compute_iteration_losses,
    ARC_COLORMAP,
    InteractiveTimelineRenderer,
    render_interactive_timeline,
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


class TestHaltingIndicators:
    """Tests for halting indicators in renderer."""

    def test_render_shows_halted_label(self, minimal_model, sample_grids):
        """When history.halted_early=True, verify figure contains 'HALTED' text."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        # Manually set halted_early to True for testing
        history.halted_early = True

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid)

        # Check that the last iteration title contains "HALTED"
        last_iteration_ax = fig.axes[-2]  # Second to last (before target)
        title_text = last_iteration_ax.get_title()

        assert "[HALTED]" in title_text

        plt.close(fig)

    def test_render_shows_max_iter_label(self, minimal_model, sample_grids):
        """When history.halted_early=False, verify figure contains 'MAX ITER' text."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        # halted_early should be False by default (model has halting disabled)
        assert history.halted_early is False

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid)

        # Check that the last iteration title contains "MAX ITER"
        last_iteration_ax = fig.axes[-2]  # Second to last (before target)
        title_text = last_iteration_ax.get_title()

        assert "[MAX ITER]" in title_text

        plt.close(fig)

    def test_halting_iteration_emphasized(self, minimal_model, sample_grids):
        """Verify halting iteration has bold font weight."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        renderer = IterationTimelineRenderer()
        fig = renderer.render_timeline(history, target_grid)

        # Check that the last iteration has bold title
        last_iteration_ax = fig.axes[-2]  # Second to last (before target)
        title_obj = last_iteration_ax.title

        # Check fontweight property
        assert title_obj.get_fontweight() in ['bold', 700, 'heavy']  # Various bold representations

        plt.close(fig)


class TestLossComputation:
    """Tests for loss computation functionality."""

    def test_compute_iteration_losses_returns_list(self, minimal_model, sample_grids):
        """Verify compute_iteration_losses returns list of floats."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        losses = compute_iteration_losses(history, target_grid)

        assert isinstance(losses, list)
        assert all(isinstance(loss, float) for loss in losses)

    def test_compute_iteration_losses_length(self, minimal_model, sample_grids):
        """Length matches number of iterations."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        losses = compute_iteration_losses(history, target_grid)

        assert len(losses) == len(history.iteration_grids)

    def test_compute_iteration_losses_range(self, minimal_model, sample_grids):
        """Values are between 0.0 and 1.0."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        losses = compute_iteration_losses(history, target_grid)

        for loss in losses:
            assert 0.0 <= loss <= 1.0

    def test_compute_iteration_losses_handles_padding(self, minimal_model):
        """Padding cells (value -1) are ignored."""
        # Create grids with padding (must be long tensors)
        input_grid = torch.zeros(5, 5, dtype=torch.long)
        target_grid = torch.zeros(5, 5, dtype=torch.long)
        target_grid[3:, :] = -1  # Add padding to bottom rows

        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)

        # Manually set one iteration to match target (except padding)
        history.iteration_grids[0][:3, :] = 0  # Match non-padded region
        history.iteration_grids[0][3:, :] = 5  # Different values in padded region

        losses = compute_iteration_losses(history, target_grid)

        # Should return 1.0 because padded region is ignored
        assert losses[0] == 1.0

    def test_capture_with_losses(self, minimal_model, sample_grids):
        """Verify capture_with_losses returns history with losses populated."""
        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)

        history = capture.capture_with_losses(input_grid, target_grid)

        # Check that iteration_losses is populated
        assert history.iteration_losses is not None
        assert isinstance(history.iteration_losses, list)
        assert len(history.iteration_losses) == len(history.iteration_grids)


class TestInteractiveTimelineRenderer:
    """Tests for InteractiveTimelineRenderer class."""

    def test_interactive_renderer_returns_figure(self, minimal_model, sample_grids):
        """Verify render_interactive returns plotly Figure."""
        plotly = pytest.importorskip("plotly")
        import plotly.graph_objects as go

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(
            history, target_grid, history.iteration_losses
        )

        assert isinstance(fig, go.Figure)

    def test_interactive_has_slider(self, minimal_model, sample_grids):
        """Verify figure has slider control for iterations."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(
            history, target_grid, history.iteration_losses
        )

        # Check that figure has sliders
        assert hasattr(fig, "layout")
        assert hasattr(fig.layout, "sliders")
        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) > 0

    def test_interactive_loss_plot_present(self, minimal_model, sample_grids):
        """Verify loss subplot exists when losses provided."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(
            history, target_grid, history.iteration_losses
        )

        # Figure should have traces (grid heatmap + loss plot)
        assert len(fig.data) > 0

        # Check for loss plot trace (Scatter type)
        has_scatter = any(trace.type == "scatter" for trace in fig.data)
        assert has_scatter, "Loss plot should include scatter trace"

    def test_interactive_html_export(self, minimal_model, sample_grids):
        """Verify write_html creates valid file."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(
            history, target_grid, history.iteration_losses
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.html"

            # Export to HTML
            fig.write_html(str(output_path), include_plotlyjs=True)

            # File should exist and have content
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_interactive_standalone_html(self, minimal_model, sample_grids):
        """Verify HTML contains embedded plotly.js."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(
            history, target_grid, history.iteration_losses
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.html"

            # Export to HTML with embedded plotly
            fig.write_html(str(output_path), include_plotlyjs=True)

            # Read file and check for plotly.js (use UTF-8 encoding)
            html_content = output_path.read_text(encoding='utf-8')

            # Should contain plotly code (not just CDN link)
            assert "plotly" in html_content.lower()
            # Check for embedded JS (characteristic of include_plotlyjs=True)
            assert len(html_content) > 100000  # Embedded plotly.js is large

    def test_interactive_without_losses(self, minimal_model, sample_grids):
        """Verify renderer works without loss plot."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture(input_grid)  # No losses

        renderer = InteractiveTimelineRenderer()
        fig = renderer.render_interactive(history, target_grid, losses=None)

        # Should still return valid figure
        assert hasattr(fig, "data")
        assert len(fig.data) > 0


class TestRenderInteractiveTimeline:
    """Tests for render_interactive_timeline convenience function."""

    def test_returns_figure(self, minimal_model, sample_grids):
        """Verify function returns plotly Figure."""
        plotly = pytest.importorskip("plotly")
        import plotly.graph_objects as go

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        fig = render_interactive_timeline(
            history, target_grid, history.iteration_losses
        )

        assert isinstance(fig, go.Figure)

    def test_saves_html_file(self, minimal_model, sample_grids):
        """Verify output_path creates HTML file."""
        plotly = pytest.importorskip("plotly")

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.html"

            fig = render_interactive_timeline(
                history,
                target_grid,
                history.iteration_losses,
                output_path=str(output_path),
            )

            # File should exist
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_returns_figure_even_with_save(self, minimal_model, sample_grids):
        """Even with output_path, should return Figure."""
        plotly = pytest.importorskip("plotly")
        import plotly.graph_objects as go

        input_grid, target_grid = sample_grids
        capture = IterationHistoryCapture(minimal_model)
        history = capture.capture_with_losses(input_grid, target_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timeline.html"

            fig = render_interactive_timeline(
                history,
                target_grid,
                history.iteration_losses,
                output_path=str(output_path),
            )

            # Should return figure even when saving
            assert isinstance(fig, go.Figure)
