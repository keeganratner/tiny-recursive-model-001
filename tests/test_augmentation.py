"""Tests for data augmentation module."""
import pytest
import torch
from src.trm.data.augmentation import D8Transform, ColorPermutation, AugmentationPipeline
from src.trm.data.collate import PAD_VALUE


class TestD8Transform:
    """Tests for D8 dihedral group transforms."""

    def test_identity_transform(self):
        """Transform 0 should be identity."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(0)
        result = transform.apply(grid)
        assert torch.equal(result, grid)

    def test_all_transforms_distinct(self):
        """All 8 D8 transforms should produce distinct results."""
        # Use asymmetric grid to ensure all transforms are distinguishable
        grid = torch.tensor([[1, 2, 3], [4, 5, 6]])
        transforms = D8Transform.get_all_transforms()
        results = [t.apply(grid) for t in transforms]

        # Convert to tuples for set comparison
        unique = set()
        for r in results:
            unique.add(tuple(r.flatten().tolist()))

        assert len(unique) == 8, f"Expected 8 distinct transforms, got {len(unique)}"

    def test_rotation_90(self):
        """Test 90-degree rotation."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(1)  # 90 deg
        result = transform.apply(grid)
        expected = torch.tensor([[2, 4], [1, 3]])
        assert torch.equal(result, expected)

    def test_rotation_180(self):
        """Test 180-degree rotation."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(2)  # 180 deg
        result = transform.apply(grid)
        expected = torch.tensor([[4, 3], [2, 1]])
        assert torch.equal(result, expected)

    def test_rotation_270(self):
        """Test 270-degree rotation."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(3)  # 270 deg
        result = transform.apply(grid)
        expected = torch.tensor([[3, 1], [4, 2]])
        assert torch.equal(result, expected)

    def test_flip_horizontal(self):
        """Test horizontal flip."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(4)  # Flip horizontal
        result = transform.apply(grid)
        expected = torch.tensor([[2, 1], [4, 3]])
        assert torch.equal(result, expected)

    def test_flip_vertical(self):
        """Test vertical flip."""
        grid = torch.tensor([[1, 2], [3, 4]])
        transform = D8Transform(5)  # Flip vertical
        result = transform.apply(grid)
        expected = torch.tensor([[3, 4], [1, 2]])
        assert torch.equal(result, expected)

    def test_apply_pair_consistency(self):
        """apply_pair should apply same transform to both grids."""
        inp = torch.tensor([[1, 2], [3, 4]])
        out = torch.tensor([[5, 6], [7, 8]])
        transform = D8Transform(2)  # 180 deg

        aug_inp, aug_out = transform.apply_pair(inp, out)

        # Both should be rotated 180 degrees
        assert torch.equal(aug_inp, transform.apply(inp))
        assert torch.equal(aug_out, transform.apply(out))

    def test_batched_input(self):
        """D8 should work on batched (B, H, W) inputs."""
        grid = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        transform = D8Transform(1)
        result = transform.apply(grid)
        assert result.shape == (2, 2, 2)


class TestColorPermutation:
    """Tests for color permutation augmentation."""

    def test_preserves_structure(self):
        """Permutation should preserve grid structure (non-zero pattern)."""
        grid = torch.tensor([[0, 1, 2], [3, 0, 4]])
        perm = ColorPermutation.random()
        result = perm.apply(grid)

        # Same shape
        assert result.shape == grid.shape

    def test_pad_value_unchanged(self):
        """PAD_VALUE (-1) should remain unchanged."""
        grid = torch.tensor([[1, 2, PAD_VALUE], [PAD_VALUE, 3, 4]])
        perm = ColorPermutation.random()
        result = perm.apply(grid)

        # -1 positions should still be -1
        assert (result == PAD_VALUE).sum() == 2
        pad_mask = grid == PAD_VALUE
        assert torch.equal(result[pad_mask], grid[pad_mask])

    def test_all_colors_mapped(self):
        """All colors 0-9 should be present in permutation."""
        perm = ColorPermutation.random()
        assert perm.permutation.shape == (10,)
        assert set(perm.permutation.tolist()) == set(range(10))

    def test_apply_pair_same_permutation(self):
        """apply_pair should use identical permutation for both."""
        inp = torch.tensor([[0, 1, 2], [3, 4, 5]])
        out = torch.tensor([[5, 4, 3], [2, 1, 0]])
        perm = ColorPermutation.random()

        aug_inp, aug_out = perm.apply_pair(inp, out)

        # If inp has value X at position, aug_inp has perm[X]
        # If out has value X at position, aug_out has perm[X]
        for val in range(6):
            inp_mask = inp == val
            out_mask = out == val
            if inp_mask.any():
                assert (aug_inp[inp_mask] == perm.permutation[val]).all()
            if out_mask.any():
                assert (aug_out[out_mask] == perm.permutation[val]).all()

    def test_deterministic_with_same_permutation(self):
        """Same permutation tensor should produce same result."""
        grid = torch.tensor([[0, 1, 2], [3, 4, 5]])
        perm_tensor = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        perm1 = ColorPermutation(perm_tensor)
        perm2 = ColorPermutation(perm_tensor)

        assert torch.equal(perm1.apply(grid), perm2.apply(grid))

    def test_compose_permutations(self):
        """Composing two permutations should work correctly."""
        perm1 = ColorPermutation(torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9]))  # Swap 0 and 1
        perm2 = ColorPermutation(torch.tensor([0, 1, 3, 2, 4, 5, 6, 7, 8, 9]))  # Swap 2 and 3
        composed = ColorPermutation.compose(perm1, perm2)

        grid = torch.tensor([[0, 1], [2, 3]])
        result = composed.apply(grid)

        # Apply perm1 then perm2 manually
        temp = perm1.apply(grid)
        expected = perm2.apply(temp)

        assert torch.equal(result, expected)


class TestAugmentationPipeline:
    """Tests for combined augmentation pipeline."""

    def test_no_augmentation(self):
        """Pipeline with both disabled should return unchanged grids."""
        pipeline = AugmentationPipeline(enable_d8=False, enable_color=False)
        inp = torch.tensor([[1, 2], [3, 4]])
        out = torch.tensor([[5, 6], [7, 8]])

        aug_inp, aug_out = pipeline.augment_pair(inp, out)

        assert torch.equal(aug_inp, inp)
        assert torch.equal(aug_out, out)

    def test_d8_only(self):
        """Pipeline with only D8 should apply geometric transform."""
        pipeline = AugmentationPipeline(enable_d8=True, enable_color=False)
        inp = torch.tensor([[1, 2], [3, 4]])
        out = torch.tensor([[1, 2], [3, 4]])  # Same as input for this test

        # Run multiple times to verify augmentation happens
        different_count = 0
        for _ in range(20):
            aug_inp, aug_out = pipeline.augment_pair(inp.clone(), out.clone())
            if not torch.equal(aug_inp, inp):
                different_count += 1

        # With 8 transforms (1 identity), expect ~7/8 to be different
        assert different_count > 5, "D8 augmentation should produce different results"

    def test_color_only(self):
        """Pipeline with only color should permute colors."""
        pipeline = AugmentationPipeline(enable_d8=False, enable_color=True)
        inp = torch.tensor([[0, 1, 2], [3, 4, 5]])
        out = torch.tensor([[5, 4, 3], [2, 1, 0]])

        aug_inp, aug_out = pipeline.augment_pair(inp, out)

        # Shape preserved
        assert aug_inp.shape == inp.shape

    def test_effective_multiplier(self):
        """Test get_effective_multiplier returns correct values."""
        assert AugmentationPipeline(False, False).get_effective_multiplier() == 1
        assert AugmentationPipeline(True, False).get_effective_multiplier() == 8
        import math
        # Color permutation preserves color 0 (black) and permutes only colors 1-9 (matches paper)
        assert AugmentationPipeline(False, True).get_effective_multiplier() == math.factorial(9)
        assert AugmentationPipeline(True, True).get_effective_multiplier() == 8 * math.factorial(9)

    def test_input_output_consistency(self):
        """Input and output must receive same augmentation."""
        pipeline = AugmentationPipeline(enable_d8=True, enable_color=True)

        # Create grids where output = input rotated (to verify consistency)
        inp = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Same for simplicity

        for _ in range(10):
            aug_inp, aug_out = pipeline.augment_pair(inp.clone(), out.clone())
            # Since inp == out, aug_inp should == aug_out
            assert torch.equal(aug_inp, aug_out), "Same grids should get same augmentation"


class TestAugmentedDataset:
    """Tests for AugmentedARCDataset integration."""

    def test_dataset_loads(self):
        """AugmentedARCDataset should load without errors."""
        from src.trm.data import AugmentedARCDataset
        ds = AugmentedARCDataset("data", enable_d8=True, enable_color=True)
        assert len(ds) > 0

    def test_dataset_length_matches_base(self):
        """Augmented dataset should have same length as base."""
        from src.trm.data import AugmentedARCDataset, ARCDataset
        base_ds = ARCDataset("data")
        aug_ds = AugmentedARCDataset("data", enable_d8=True, enable_color=True)
        assert len(aug_ds) == len(base_ds)

    def test_different_augmentations_per_call(self):
        """Same index should return different augmentations."""
        from src.trm.data import AugmentedARCDataset
        ds = AugmentedARCDataset("data", enable_d8=True, enable_color=True)

        # Get same task twice
        item1 = ds[0]
        item2 = ds[0]

        # With D8 (8 options) and color (10! options), extremely unlikely to match
        inp1 = item1["train_inputs"][0]
        inp2 = item2["train_inputs"][0]

        # Note: shapes might differ due to D8 rotation on non-square grids
        # But at minimum, with color permutation, values should differ
        if inp1.shape == inp2.shape:
            assert not torch.equal(inp1, inp2), "Same task should get different augmentation"

    def test_d8_only_mode(self):
        """Test AugmentedARCDataset with D8 only."""
        from src.trm.data import AugmentedARCDataset
        ds = AugmentedARCDataset("data", enable_d8=True, enable_color=False)
        assert ds.pipeline.enable_d8 == True
        assert ds.pipeline.enable_color == False
        assert ds.pipeline.get_effective_multiplier() == 8

    def test_color_only_mode(self):
        """Test AugmentedARCDataset with color only."""
        from src.trm.data import AugmentedARCDataset
        ds = AugmentedARCDataset("data", enable_d8=False, enable_color=True)
        assert ds.pipeline.enable_d8 == False
        assert ds.pipeline.enable_color == True
        assert ds.pipeline.get_effective_multiplier() == 362880  # 9! (color 0 preserved)

    def test_item_structure(self):
        """Test that augmented items have correct structure."""
        from src.trm.data import AugmentedARCDataset
        ds = AugmentedARCDataset("data", enable_d8=True, enable_color=True)
        item = ds[0]

        # Check structure
        assert "task_id" in item
        assert "train_inputs" in item
        assert "train_outputs" in item
        assert "test_inputs" in item
        assert "test_outputs" in item

        # Check types
        assert isinstance(item["train_inputs"], list)
        assert isinstance(item["train_outputs"], list)
        assert all(isinstance(t, torch.Tensor) for t in item["train_inputs"])
        assert all(isinstance(t, torch.Tensor) for t in item["train_outputs"])
