"""Grid timeline rendering with ARC color palette."""
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .capture import IterationHistory


# Official ARC-AGI 10-color palette
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Gray
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Cyan
    "#870C25",  # 9: Brown
]

# Create matplotlib colormap for ARC colors
ARC_COLORMAP = ListedColormap(ARC_COLORS)


class IterationTimelineRenderer:
    """
    Renders iteration history as visual timeline.

    Creates a vertical layout showing the evolution of model predictions:
    - Row 0: Input grid (static reference)
    - Rows 1-N: Iteration grids (model predictions at each outer iteration)
    - Row N+1: Target grid (optional, for comparison)

    When show_diff=True and target provided, adds color-coded overlays:
    - Green border: Cell matches target
    - Red border: Cell doesn't match target
    - Yellow border: Cell changed from previous iteration

    Args:
        cell_size: Size of each grid cell in inches (default 0.3)
    """

    def __init__(self, cell_size: float = 0.3):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each grid cell in inches for figure sizing
        """
        self.cell_size = cell_size
        self.colormap = ARC_COLORMAP

    def render_timeline(
        self,
        history: IterationHistory,
        target_grid: Optional[torch.Tensor] = None,
        show_diff: bool = True,
    ) -> Figure:
        """
        Render iteration timeline as matplotlib figure.

        Creates a vertical layout with:
        - Input grid at top
        - Each iteration's prediction in sequence
        - Optional target grid at bottom

        Args:
            history: IterationHistory from capture
            target_grid: Optional target solution (H, W) for comparison
            show_diff: Whether to show difference overlay (requires target_grid)

        Returns:
            matplotlib Figure object
        """
        # Extract grids
        input_grid = history.input_grid
        iteration_grids = history.iteration_grids
        confidences = history.halt_confidences

        # Determine number of rows
        num_rows = 1 + len(iteration_grids)  # Input + iterations
        if target_grid is not None:
            num_rows += 1  # Add target row

        # Get grid dimensions for figure sizing
        H, W = input_grid.shape

        # Create figure with vertical layout
        fig_width = W * self.cell_size + 3  # Extra space for labels
        fig_height = num_rows * H * self.cell_size + 1
        fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, fig_height))

        # Ensure axes is array even for single row
        if num_rows == 1:
            axes = [axes]

        row_idx = 0

        # Row 0: Input grid
        self._render_grid(axes[row_idx], input_grid, "Input")
        row_idx += 1

        # Rows 1-N: Iteration grids
        for iter_idx, (grid, confidence) in enumerate(zip(iteration_grids, confidences)):
            # Check if this is the last iteration (potential halting point)
            is_last_iteration = (iter_idx == len(iteration_grids) - 1)

            # Build title with confidence and halting indicator
            if is_last_iteration:
                if history.halted_early:
                    status_label = "[HALTED]"
                    title_fontweight = 'bold'
                else:
                    status_label = "[MAX ITER]"
                    title_fontweight = 'bold'
                title = f"Iteration {iter_idx} | conf: {confidence:.3f} {status_label}"
            else:
                title = f"Iteration {iter_idx} | conf: {confidence:.3f}"
                title_fontweight = 'normal'

            self._render_grid(axes[row_idx], grid, title, fontweight=title_fontweight)

            # Add diff overlay if requested
            if show_diff and target_grid is not None:
                # Get previous grid (or input for first iteration)
                prev_grid = iteration_grids[iter_idx - 1] if iter_idx > 0 else input_grid
                self._render_diff_overlay(axes[row_idx], grid, prev_grid, target_grid)

            row_idx += 1

        # Optional: Target grid
        if target_grid is not None:
            self._render_grid(axes[row_idx], target_grid, "Target")

        # Adjust layout
        plt.tight_layout()

        return fig

    def _render_grid(self, ax: plt.Axes, grid: torch.Tensor, title: str, fontweight: str = 'normal') -> None:
        """
        Render single grid with imshow.

        Args:
            ax: Matplotlib axes to render on
            grid: Grid tensor (H, W) with values 0-9 or -1 (padding)
            title: Title for this grid
            fontweight: Font weight for title ('normal' or 'bold')
        """
        # Convert to numpy for matplotlib
        grid_np = grid.cpu().numpy()

        # Handle padding: set -1 to 0 (black) for display
        grid_display = grid_np.copy()
        grid_display[grid_display == -1] = 0

        # Render with ARC colormap
        ax.imshow(grid_display, cmap=self.colormap, vmin=0, vmax=9, interpolation="nearest")

        # Set title with optional font weight
        ax.set_title(title, fontsize=10, pad=5, fontweight=fontweight)

        # Remove axis ticks but keep grid
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid lines between cells
        H, W = grid_display.shape
        ax.set_xticks([x - 0.5 for x in range(W + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(H + 1)], minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    def _render_diff_overlay(
        self,
        ax: plt.Axes,
        current: torch.Tensor,
        previous: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Add colored borders to show differences.

        Color coding:
        - Green: Correct cell (matches target)
        - Red: Incorrect cell (doesn't match target)
        - Yellow: Changed from previous iteration

        Args:
            ax: Matplotlib axes to add overlay to
            current: Current iteration grid (H, W)
            previous: Previous iteration grid (H, W)
            target: Target solution grid (H, W)
        """
        H, W = current.shape

        # Convert to numpy for comparison
        current_np = current.cpu().numpy()
        previous_np = previous.cpu().numpy()
        target_np = target.cpu().numpy()

        # Iterate over cells and add colored rectangles
        for i in range(H):
            for j in range(W):
                # Skip padding cells
                if current_np[i, j] == -1:
                    continue

                # Check correctness (green/red)
                is_correct = current_np[i, j] == target_np[i, j]
                # Check if changed from previous (yellow)
                has_changed = current_np[i, j] != previous_np[i, j]

                # Determine border color
                if has_changed and not is_correct:
                    # Changed but still wrong: red with thick border
                    color = "red"
                    linewidth = 2
                elif has_changed and is_correct:
                    # Changed and now correct: green with thick border
                    color = "green"
                    linewidth = 2
                elif is_correct:
                    # Correct (didn't change): thin green border
                    color = "green"
                    linewidth = 1
                else:
                    # Incorrect (didn't change): thin red border
                    color = "red"
                    linewidth = 1

                # Add rectangle border
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)


def render_iteration_timeline(
    history: IterationHistory,
    target_grid: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
    show_diff: bool = True,
    dpi: int = 150,
) -> Figure:
    """
    Convenience function to render and optionally save timeline.

    Args:
        history: IterationHistory from capture
        target_grid: Optional target solution (H, W) for comparison
        output_path: Optional path to save figure (e.g., "timeline.png")
        show_diff: Whether to show difference overlay (requires target_grid)
        dpi: DPI for saved figure (default 150)

    Returns:
        matplotlib Figure object

    Example:
        >>> capture = IterationHistoryCapture(refinement)
        >>> history = capture.capture(input_grid)
        >>> fig = render_iteration_timeline(history, target, output_path="out.png")
    """
    renderer = IterationTimelineRenderer()
    fig = renderer.render_timeline(history, target_grid, show_diff)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
