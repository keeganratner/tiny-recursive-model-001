"""Data augmentation for ARC-AGI grids.

This module provides geometric (D8 dihedral group), color permutation, and
translational augmentations to increase effective training data size.

D8 dihedral group: The 8 symmetries of a square (4 rotations + 4 reflections)
Color permutations: Shuffling ARC colors 1-9 while keeping 0 (black) fixed
Translational augmentation: Random shift of grid content within 30x30 canvas

Matches the augmentation strategy in the Samsung TRM paper (arXiv:2510.04871).
"""
import torch
from typing import List, Tuple, Optional
import random


def _pad_grid(grid: torch.Tensor, pad_top: int, pad_left: int) -> torch.Tensor:
    """Pad grid with black (color 0) on top and left to shift content.

    Args:
        grid: Tensor of shape (H, W)
        pad_top: Number of rows to add at the top
        pad_left: Number of columns to add at the left

    Returns:
        Padded tensor of shape (H + pad_top, W + pad_left), new cells = 0
    """
    if pad_top == 0 and pad_left == 0:
        return grid
    H, W = grid.shape
    new_grid = torch.zeros(H + pad_top, W + pad_left, dtype=grid.dtype)
    new_grid[pad_top:, pad_left:] = grid
    return new_grid


class D8Transform:
    """D8 dihedral group transformations (8 symmetries of a square).

    The D8 group consists of:
    - 4 rotations: 0°, 90°, 180°, 270°
    - 4 reflections: horizontal, vertical, main diagonal, anti-diagonal

    All transformations preserve the grid structure and can be applied
    consistently to input-output pairs for training.
    """

    def __init__(self, transform_id: int):
        """Initialize a D8 transform.

        Args:
            transform_id: Integer in [0-7] identifying the transformation:
                0: Identity (no change)
                1: Rotate 90° clockwise
                2: Rotate 180°
                3: Rotate 270° clockwise (90° counter-clockwise)
                4: Flip horizontal (left-right)
                5: Flip vertical (top-bottom)
                6: Flip along main diagonal (transpose)
                7: Flip along anti-diagonal (transpose + 180° rotation)
        """
        if not 0 <= transform_id <= 7:
            raise ValueError(f"transform_id must be in [0, 7], got {transform_id}")
        self.transform_id = transform_id

    @staticmethod
    def get_all_transforms() -> List['D8Transform']:
        """Get all 8 D8 group transformations.

        Returns:
            List of 8 D8Transform instances (one for each group element)
        """
        return [D8Transform(i) for i in range(8)]

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply the D8 transformation to a grid.

        Args:
            grid: Tensor of shape (H, W) or (B, H, W) containing grid values

        Returns:
            Transformed grid with same dtype and device as input
        """
        if self.transform_id == 0:
            # Identity
            return grid.clone()
        elif self.transform_id == 1:
            # Rotate 90° clockwise
            return torch.rot90(grid, k=1, dims=(-2, -1))
        elif self.transform_id == 2:
            # Rotate 180°
            return torch.rot90(grid, k=2, dims=(-2, -1))
        elif self.transform_id == 3:
            # Rotate 270° clockwise
            return torch.rot90(grid, k=3, dims=(-2, -1))
        elif self.transform_id == 4:
            # Flip horizontal (left-right)
            return torch.flip(grid, dims=[-1])
        elif self.transform_id == 5:
            # Flip vertical (top-bottom)
            return torch.flip(grid, dims=[-2])
        elif self.transform_id == 6:
            # Transpose (main diagonal flip)
            return grid.transpose(-2, -1)
        elif self.transform_id == 7:
            # Anti-diagonal flip (transpose + 180° rotation)
            return torch.rot90(grid.transpose(-2, -1), k=2, dims=(-2, -1))
        else:
            raise ValueError(f"Invalid transform_id: {self.transform_id}")

    def apply_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the SAME transformation to both input and output grids.

        This ensures geometric consistency required for training - the input-output
        relationship is preserved under the transformation.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (transformed_input, transformed_output)
        """
        return self.apply(input_grid), self.apply(output_grid)

    def __repr__(self) -> str:
        names = ["Identity", "Rot90", "Rot180", "Rot270",
                 "FlipH", "FlipV", "Transpose", "AntiDiag"]
        return f"D8Transform({self.transform_id}: {names[self.transform_id]})"


class ColorPermutation:
    """Color permutation augmentation for ARC grids.

    Shuffles ARC colors 1-9 while preserving color 0 (black), matching the
    paper's approach. Color 0 (black) is preserved because it serves as the
    background in most ARC tasks.

    This creates up to 9! = 362,880 possible augmentations per grid.

    Important: Handles PAD_VALUE (-1) correctly by leaving it unchanged.
    """

    def __init__(self, permutation: Optional[torch.Tensor] = None):
        """Initialize a color permutation.

        Args:
            permutation: Tensor of shape (10,) mapping old colors to new colors.
                        Must preserve index 0 (i.e. permutation[0] == 0).
                        If None, generates a random permutation of colors 1-9.
        """
        if permutation is None:
            permutation = self._make_random_perm()

        if permutation.shape != (10,):
            raise ValueError(f"Permutation must have shape (10,), got {permutation.shape}")

        self.permutation = permutation

    @staticmethod
    def _make_random_perm() -> torch.Tensor:
        """Create a permutation that fixes color 0 and shuffles colors 1-9."""
        perm = torch.zeros(10, dtype=torch.long)
        perm[0] = 0  # black (color 0) is always preserved
        perm[1:] = torch.randperm(9) + 1  # permute colors 1-9
        return perm

    @staticmethod
    def random() -> 'ColorPermutation':
        """Create a random color permutation preserving color 0 (black).

        Matches the paper's approach: permute only colors 1-9, keep 0 fixed.

        Returns:
            ColorPermutation with randomly shuffled colors 1-9
        """
        return ColorPermutation(ColorPermutation._make_random_perm())

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply color permutation to a grid.

        Args:
            grid: Tensor containing color values (0-9) and possibly PAD_VALUE (-1)

        Returns:
            Grid with permuted colors, PAD_VALUE preserved, color 0 preserved
        """
        # Handle padding: -1 should remain -1
        pad_mask = (grid == -1)

        # Clamp to valid range [0, 9] for indexing
        clamped = grid.clamp(min=0)

        # Apply permutation
        result = self.permutation[clamped]

        # Restore padding
        result[pad_mask] = -1

        return result

    def apply_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the SAME color permutation to both input and output grids.

        This ensures color consistency - if color A maps to color B in the input,
        it also maps to color B in the output.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (permuted_input, permuted_output)
        """
        return self.apply(input_grid), self.apply(output_grid)

    @staticmethod
    def compose(perm1: 'ColorPermutation', perm2: 'ColorPermutation') -> 'ColorPermutation':
        """Compose two color permutations.

        The result applies perm1 first, then perm2.

        Args:
            perm1: First permutation to apply
            perm2: Second permutation to apply

        Returns:
            Composed permutation
        """
        composed = perm2.permutation[perm1.permutation]
        return ColorPermutation(composed)

    def __repr__(self) -> str:
        return f"ColorPermutation({self.permutation.tolist()})"


class AugmentationPipeline:
    """Combined augmentation pipeline for D8, color permutation, and translation.

    Applies geometric and/or color and/or translational augmentations to
    input-output grid pairs, ensuring consistency between input and output.

    Key design: use sample() + apply_with() for task-level consistent augmentation
    (all pairs in a task get the same transforms). This is required for in-context
    learning where consistent color/geometry mappings must hold across demo pairs.
    """

    ARC_MAX = 30  # Maximum ARC grid dimension (30x30 canvas)

    def __init__(
        self,
        enable_d8: bool = True,
        enable_color: bool = True,
        enable_translation: bool = False,
    ):
        """Initialize augmentation pipeline.

        Args:
            enable_d8: Whether to apply D8 geometric transforms
            enable_color: Whether to apply color permutations (colors 1-9, 0 fixed)
            enable_translation: Whether to apply translational augmentation
                                (random padding to shift grid within 30x30 canvas)
        """
        self.enable_d8 = enable_d8
        self.enable_color = enable_color
        self.enable_translation = enable_translation

    def sample(self) -> Tuple[Optional[D8Transform], Optional[ColorPermutation]]:
        """Sample a consistent set of transforms for one task.

        Returns transforms that should be applied to ALL pairs in the task.
        For translational augmentation, see AugmentedARCDataset which handles
        the task-wide max-dimension computation before sampling the offset.

        Returns:
            Tuple of (d8_transform or None, color_permutation or None)
        """
        d8 = D8Transform(random.randint(0, 7)) if self.enable_d8 else None
        color = ColorPermutation.random() if self.enable_color else None
        return d8, color

    def apply_with(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        d8: Optional[D8Transform],
        color: Optional[ColorPermutation],
        pad_top: int = 0,
        pad_left: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply specific transforms to an input-output pair.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor
            d8: D8 transform to apply, or None
            color: Color permutation to apply, or None
            pad_top: Top padding for translational augmentation (0 = no shift)
            pad_left: Left padding for translational augmentation (0 = no shift)

        Returns:
            Tuple of (augmented_input, augmented_output)
        """
        aug_input, aug_output = input_grid, output_grid

        if d8 is not None:
            aug_input, aug_output = d8.apply_pair(aug_input, aug_output)

        if color is not None:
            aug_input, aug_output = color.apply_pair(aug_input, aug_output)

        if pad_top > 0 or pad_left > 0:
            aug_input = _pad_grid(aug_input, pad_top, pad_left)
            aug_output = _pad_grid(aug_output, pad_top, pad_left)

        return aug_input, aug_output

    def augment_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a fresh random augmentation to a single pair.

        Note: For consistent task-level augmentation (required for in-context
        learning), use sample() + apply_with() instead. This method is kept for
        backward compatibility with non-in-context training.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (augmented_input, augmented_output) with consistent transforms
        """
        d8, color = self.sample()
        return self.apply_with(input_grid, output_grid, d8, color)

    def get_effective_multiplier(self) -> int:
        """Get the effective data multiplier from enabled augmentations.

        Returns:
            Integer multiplier:
                - 1 if nothing enabled
                - 8 if only D8 enabled
                - 362880 if only color enabled (9!, preserving color 0)
                - 2903040 if D8 + color (8 * 9!)
                - Higher with translation (translation adds ~900x from 30x30 canvas)
        """
        multiplier = 1

        if self.enable_d8:
            multiplier *= 8

        if self.enable_color:
            multiplier *= 362880  # 9! (colors 1-9 permuted, 0 fixed)

        return multiplier

    def __repr__(self) -> str:
        return (f"AugmentationPipeline(d8={self.enable_d8}, color={self.enable_color}, "
                f"translation={self.enable_translation}, multiplier={self.get_effective_multiplier()})")


if __name__ == "__main__":
    print("Testing D8 transforms...")

    # Create a test grid with distinct values
    test_grid = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print(f"Original grid:\n{test_grid}\n")

    # Apply all 8 transforms
    transforms = D8Transform.get_all_transforms()
    results = []

    for t in transforms:
        result = t.apply(test_grid)
        results.append(result)
        print(f"{t}:\n{result}\n")

    # Verify we get 8 distinct results
    unique_results = []
    for r in results:
        is_unique = True
        for u in unique_results:
            if torch.equal(r, u):
                is_unique = False
                break
        if is_unique:
            unique_results.append(r)

    assert len(unique_results) == 8, f"Expected 8 distinct transforms, got {len(unique_results)}"

    # Verify identity transform
    identity = D8Transform(0).apply(test_grid)
    assert torch.equal(identity, test_grid), "Identity transform should return unchanged grid"

    print("[PASS] D8 transforms verified: 8 distinct orientations\n")

    # Test color permutation
    print("Testing color permutation...")

    color_grid = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [-1, -1, -1]  # Padding row
    ])

    print(f"Original grid:\n{color_grid}\n")

    perm = ColorPermutation.random()
    permuted = perm.apply(color_grid)

    print(f"Permuted grid:\n{permuted}\n")
    print(f"Permutation: {perm.permutation.tolist()}\n")

    # Verify padding is preserved
    assert torch.all(permuted[2, :] == -1), "PAD_VALUE (-1) should be preserved"

    # Verify color 0 (black) is preserved
    assert permuted[0, 0] == 0, "Color 0 (black) should be preserved"

    # Verify structure is preserved (same positions are non-padding)
    assert (color_grid != -1).sum() == (permuted != -1).sum(), "Structure should be preserved"

    # Verify only colors 1-9 are permuted (0 stays 0)
    assert perm.permutation[0] == 0, "Color 0 must map to 0"
    assert set(perm.permutation[1:].tolist()) == set(range(1, 10)), "Colors 1-9 must be a permutation of 1-9"

    # Test apply_pair consistency
    input_grid = torch.tensor([[0, 1], [2, 3]])
    output_grid = torch.tensor([[4, 5], [6, 7]])

    perm2 = ColorPermutation(torch.tensor([0, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
    perm_in, perm_out = perm2.apply_pair(input_grid, output_grid)

    # Verify same permutation applied to both
    assert torch.equal(perm_in, perm2.apply(input_grid)), "apply_pair should use same permutation"
    assert torch.equal(perm_out, perm2.apply(output_grid)), "apply_pair should use same permutation"

    print("[PASS] Color permutation verified: structure preserved, PAD_VALUE handled, color 0 fixed\n")

    # Test translational augmentation
    print("Testing translational augmentation (_pad_grid)...")
    small_grid = torch.tensor([[1, 2], [3, 4]])
    padded = _pad_grid(small_grid, pad_top=2, pad_left=3)
    assert padded.shape == (4, 5), f"Expected (4, 5), got {padded.shape}"
    assert padded[2, 3] == 1, "Content should start at (pad_top, pad_left)"
    assert padded[0, 0] == 0, "Padded area should be 0 (black)"
    print("[PASS] Translational augmentation verified\n")

    # Test combined pipeline
    print("Testing augmentation pipeline...")

    pipeline = AugmentationPipeline(enable_d8=True, enable_color=True, enable_translation=True)
    print(f"{pipeline}\n")

    # Test sample + apply_with (task-level consistent augmentation)
    test_in = torch.tensor([[0, 1], [2, 3]])
    test_out = torch.tensor([[1, 2], [3, 4]])

    d8, color = pipeline.sample()
    aug_in, aug_out = pipeline.apply_with(test_in, test_out, d8, color, pad_top=1, pad_left=2)
    print(f"Input: {test_in.tolist()} -> {aug_in.tolist()}")
    print(f"Output: {test_out.tolist()} -> {aug_out.tolist()}")

    # Verify effective multiplier calculation
    assert pipeline.get_effective_multiplier() == 8 * 362880, "Wrong multiplier calculation"

    pipeline_d8_only = AugmentationPipeline(enable_d8=True, enable_color=False)
    assert pipeline_d8_only.get_effective_multiplier() == 8, "D8-only should be 8x"

    pipeline_color_only = AugmentationPipeline(enable_d8=False, enable_color=True)
    assert pipeline_color_only.get_effective_multiplier() == 362880, "Color-only should be 9!"

    pipeline_disabled = AugmentationPipeline(enable_d8=False, enable_color=False)
    assert pipeline_disabled.get_effective_multiplier() == 1, "Disabled should be 1x"

    print("\n[PASS] Augmentation pipeline verified")
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
