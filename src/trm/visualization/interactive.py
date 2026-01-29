"""Interactive HTML timeline rendering with Plotly."""
from typing import Optional, List

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .capture import IterationHistory


# Official ARC-AGI 10-color palette for Plotly colorscale
# Format: [position, color_hex] where position is 0.0 to 1.0
ARC_COLORS_PLOTLY = [
    [0.0, "#000000"],   # 0: Black
    [0.1, "#0074D9"],   # 1: Blue
    [0.2, "#FF4136"],   # 2: Red
    [0.3, "#2ECC40"],   # 3: Green
    [0.4, "#FFDC00"],   # 4: Yellow
    [0.5, "#AAAAAA"],   # 5: Gray
    [0.6, "#F012BE"],   # 6: Magenta
    [0.7, "#FF851B"],   # 7: Orange
    [0.8, "#7FDBFF"],   # 8: Cyan
    [0.9, "#870C25"],   # 9: Brown
    [1.0, "#870C25"],   # Cap at brown
]


class InteractiveTimelineRenderer:
    """
    Renders iteration history as interactive HTML with Plotly.

    Creates an interactive visualization with:
    - Grid visualization with slider to step through iterations
    - Loss/accuracy plot showing refinement quality over time
    - Halting indicator showing model confidence and status

    The output is a standalone HTML file that works without internet connection
    (includes embedded plotly.js).

    Args:
        cell_size: Size of each grid cell in pixels for HTML (default 30)
    """

    def __init__(self, cell_size: int = 30):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each grid cell in pixels for HTML display
        """
        self.cell_size = cell_size

    def render_interactive(
        self,
        history: IterationHistory,
        target_grid: Optional[torch.Tensor] = None,
        losses: Optional[List[float]] = None,
    ) -> go.Figure:
        """
        Render iteration timeline as interactive Plotly figure.

        Creates a two-row layout:
        - Top row: Grid heatmap with slider to step through iterations
        - Bottom row: Loss/accuracy line plot synchronized with slider

        Args:
            history: IterationHistory from capture
            target_grid: Optional target solution (H, W) for comparison
            losses: Optional list of accuracy values per iteration

        Returns:
            plotly.graph_objects.Figure with interactive controls
        """
        # Determine if we have a bottom plot
        has_loss_plot = losses is not None and len(losses) > 0

        # Create subplots: 2 rows if loss plot, else 1 row
        if has_loss_plot:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Grid Visualization", "Accuracy Over Iterations"),
                vertical_spacing=0.15,
            )
        else:
            fig = make_subplots(rows=1, cols=1)

        # Build grid frames for slider
        frames = self._build_grid_frames(history, target_grid)

        # Add initial grid (Input)
        initial_frame = frames[0]
        fig.add_trace(
            go.Heatmap(
                z=initial_frame["z"],
                colorscale=ARC_COLORS_PLOTLY,
                zmin=0,
                zmax=9,
                showscale=False,
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Color: %{z}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add loss plot if available
        if has_loss_plot:
            iteration_nums = list(range(len(losses)))
            fig.add_trace(
                go.Scatter(
                    x=iteration_nums,
                    y=losses,
                    mode="lines+markers",
                    name="Accuracy",
                    line=dict(color="blue", width=2),
                    marker=dict(size=8),
                    hovertemplate="Iteration: %{x}<br>Accuracy: %{y:.3f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add vertical line for current iteration (will be updated by slider)
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[0, 1],
                    mode="lines",
                    name="Current",
                    line=dict(color="red", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

            # Add halting line if model halted early
            if history.halted_early:
                halting_iter = len(history.iteration_grids) - 1
                fig.add_trace(
                    go.Scatter(
                        x=[halting_iter, halting_iter],
                        y=[0, 1],
                        mode="lines",
                        name="Halted",
                        line=dict(color="green", width=2, dash="dot"),
                        showlegend=True,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )

        # Create slider steps and frames
        slider_steps = []
        plotly_frames = []

        for i, frame_data in enumerate(frames):
            # Create frame for animation
            frame_traces = [
                go.Heatmap(
                    z=frame_data["z"],
                    colorscale=ARC_COLORS_PLOTLY,
                    zmin=0,
                    zmax=9,
                    showscale=False,
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Color: %{z}<extra></extra>",
                )
            ]

            # Update loss plot marker if present
            if has_loss_plot:
                # Get current iteration index (frames include Input + iterations + optionally Target)
                if frame_data["label"] == "Input":
                    current_iter = None
                elif frame_data["label"] == "Target":
                    current_iter = None
                else:
                    # Extract iteration number from label
                    current_iter = frame_data["iteration_idx"]

                # Add vertical line at current iteration
                if current_iter is not None and current_iter < len(losses):
                    frame_traces.append(
                        go.Scatter(
                            x=[current_iter, current_iter],
                            y=[0, 1],
                            mode="lines",
                            line=dict(color="red", width=2, dash="dash"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                else:
                    # No vertical line for Input/Target
                    frame_traces.append(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="lines",
                            showlegend=False,
                        )
                    )

                # Add halting line (static across frames)
                if history.halted_early:
                    halting_iter = len(history.iteration_grids) - 1
                    frame_traces.append(
                        go.Scatter(
                            x=[halting_iter, halting_iter],
                            y=[0, 1],
                            mode="lines",
                            line=dict(color="green", width=2, dash="dot"),
                            showlegend=True if i == 0 else False,
                            hoverinfo="skip",
                        )
                    )

            plotly_frames.append(
                go.Frame(
                    data=frame_traces,
                    name=str(i),
                    layout=go.Layout(title_text=frame_data["title"]),
                )
            )

            # Create slider step
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": frame_data["label"],
                    "method": "animate",
                }
            )

        # Add frames to figure
        fig.frames = plotly_frames

        # Add slider
        sliders = [
            {
                "active": 0,
                "yanchor": "top",
                "y": -0.05 if not has_loss_plot else -0.1,
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Step: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": slider_steps,
            }
        ]

        # Update layout
        fig.update_layout(
            sliders=sliders,
            title={
                "text": frames[0]["title"],
                "x": 0.5,
                "xanchor": "center",
            },
            height=800 if has_loss_plot else 600,
            showlegend=has_loss_plot,
        )

        # Configure grid plot axes
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)

        # Configure loss plot axes if present
        if has_loss_plot:
            fig.update_xaxes(title_text="Iteration", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=2, col=1)

        return fig

    def _build_grid_frames(
        self, history: IterationHistory, target_grid: Optional[torch.Tensor]
    ) -> List[dict]:
        """
        Build frame data for slider animation.

        Creates a list of frame dictionaries, each containing:
        - z: Grid data for heatmap
        - label: Short label for slider
        - title: Full title for display
        - iteration_idx: Iteration index (for loss plot sync)

        Args:
            history: IterationHistory with grids
            target_grid: Optional target grid

        Returns:
            List of frame dictionaries
        """
        frames = []

        # Frame 0: Input
        input_data = self._prepare_grid_data(history.input_grid)
        frames.append(
            {
                "z": input_data,
                "label": "Input",
                "title": "Input Grid",
                "iteration_idx": None,
            }
        )

        # Frames 1-N: Iterations
        for i, (grid, confidence) in enumerate(
            zip(history.iteration_grids, history.halt_confidences)
        ):
            grid_data = self._prepare_grid_data(grid)

            # Check if this is the last iteration
            is_last_iteration = i == len(history.iteration_grids) - 1

            if is_last_iteration:
                if history.halted_early:
                    status_label = "<b>[HALTED]</b>"
                else:
                    status_label = "<b>[MAX ITER]</b>"
                title = f"<b>Iteration {i} | conf: {confidence:.3f} {status_label}</b>"
            else:
                title = f"Iteration {i} | conf: {confidence:.3f}"

            frames.append(
                {
                    "z": grid_data,
                    "label": f"Iter {i}",
                    "title": title,
                    "iteration_idx": i,
                }
            )

        # Optional: Target
        if target_grid is not None:
            target_data = self._prepare_grid_data(target_grid)
            frames.append(
                {
                    "z": target_data,
                    "label": "Target",
                    "title": "Target Grid",
                    "iteration_idx": None,
                }
            )

        return frames

    def _prepare_grid_data(self, grid: torch.Tensor) -> list:
        """
        Prepare grid tensor for Plotly heatmap.

        Converts tensor to list format and handles padding.

        Args:
            grid: Grid tensor (H, W) with values 0-9 or -1 (padding)

        Returns:
            List of lists suitable for Plotly heatmap z parameter
        """
        # Convert to numpy
        grid_np = grid.cpu().numpy()

        # Handle padding: set -1 to 0 (black) for display
        grid_display = grid_np.copy()
        grid_display[grid_display == -1] = 0

        # Convert to list for Plotly (y-axis flipped for proper display)
        return grid_display.tolist()


def render_interactive_timeline(
    history: IterationHistory,
    target_grid: Optional[torch.Tensor] = None,
    losses: Optional[List[float]] = None,
    output_path: Optional[str] = None,
) -> go.Figure:
    """
    Convenience function to render interactive timeline and optionally save as HTML.

    Args:
        history: IterationHistory from capture
        target_grid: Optional target solution (H, W) for comparison
        losses: Optional list of accuracy values per iteration
        output_path: Optional path to save HTML file (e.g., "timeline.html")

    Returns:
        plotly.graph_objects.Figure with interactive controls

    Example:
        >>> capture = IterationHistoryCapture(refinement)
        >>> history = capture.capture_with_losses(input_grid, target_grid)
        >>> fig = render_interactive_timeline(
        ...     history, target_grid, history.iteration_losses,
        ...     output_path="timeline.html"
        ... )
    """
    renderer = InteractiveTimelineRenderer()
    fig = renderer.render_interactive(history, target_grid, losses)

    if output_path is not None:
        # Save as standalone HTML with embedded plotly.js
        fig.write_html(output_path, include_plotlyjs=True)

    return fig
