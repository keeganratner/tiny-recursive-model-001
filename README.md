# Tiny Recursive Model (TRM) for ARC-AGI

## Overview

This is a PyTorch implementation of the Tiny Recursive Model (TRM) from Samsung's "Less is More: Recursive Reasoning with Tiny Networks" paper ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)), applied to ARC-AGI puzzles. The core value of this project is visualizing recursive reasoning - seeing how the model iteratively refines its answer through multiple reasoning steps, making the internal problem-solving process transparent and interpretable.

The implementation features a 2-layer transformer with recursive refinement that achieves effective depth of 42 layers through weight sharing while maintaining only ~10.5M parameters. By reusing the same small network repeatedly in a nested loop structure, the TRM demonstrates that parameter efficiency and reasoning depth can be decoupled.

## Installation

### Requirements

- Python 3.8 or higher
- CPU-only PyTorch (no CUDA required for development)

### Setup

```bash
# Clone repository
git clone https://github.com/keeganr-dot/trm001.git
cd trm001

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project uses minimal dependencies:
- `torch>=2.0.0` - PyTorch for neural network implementation
- `einops>=0.7.0` - Tensor operations
- `hydra-core>=1.3.0` & `omegaconf>=2.3.0` - Configuration management
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization
- `pytest>=7.0.0` - Testing framework

## Quick Start

### Download ARC-AGI Dataset

```bash
# Clone the official ARC-AGI repository into data directory
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI
```

This provides access to 800 tasks (400 training, 400 evaluation) from the ARC-AGI benchmark.

### Training Examples

```bash
# Quick test run (smaller model, limited data)
python scripts/train.py --fast --epochs 5

# Basic training without deep supervision
python scripts/train.py --epochs 50

# Deep supervision training (recommended)
python scripts/train.py --deep-supervision --epochs 50 --patience 5

# Full training with augmentation
python scripts/train.py --deep-supervision --augment --epochs 50 --patience 5

# Resume from checkpoint
python scripts/train.py --deep-supervision --resume checkpoints/best_model.pt
```

**Training modes:**
- `--fast`: Uses smaller batch size (4) and limited data (10 tasks) for quick testing
- `--deep-supervision`: Enables intermediate supervision at each outer iteration (recommended for better training)
- `--augment`: Enables data augmentation (D8 dihedral transforms + color permutation = 29M× multiplier)
- `--patience N`: Early stopping if validation accuracy doesn't improve for N epochs
- `--resume PATH`: Resume training from saved checkpoint

**Checkpoints:** Best model is automatically saved to `checkpoints/best_model.pt` during training.

## Project Structure

```
trm001/
├── src/trm/
│   ├── model/              # TRM architecture components
│   │   ├── embedding.py    # GridEmbedding (color → vector)
│   │   ├── layers.py       # RMSNorm, SwiGLU, RotaryEmbedding
│   │   ├── transformer.py  # TRMStack (2-layer transformer)
│   │   ├── heads.py        # OutputHead, HaltingHead
│   │   ├── network.py      # TRMNetwork (complete model)
│   │   └── recursive.py    # RecursiveRefinement (nested loops)
│   ├── data/               # Dataset and augmentation
│   │   ├── dataset.py      # ARCDataset loader
│   │   ├── collate.py      # Batch padding and masking
│   │   └── augmentation.py # D8 transforms + color permutation
│   ├── training/           # Training loops and deep supervision
│   │   └── trainer.py      # TRMTrainer with deep supervision
│   └── evaluation/         # Checkpointing and metrics
│       ├── metrics.py      # Exact-match accuracy
│       ├── checkpoint.py   # Save/load and best model tracking
│       └── validation.py   # Validation loop and early stopping
├── scripts/
│   └── train.py            # Main training script with CLI
├── configs/
│   └── config.yaml         # Hyperparameters (T=3, n=6, dim=512)
└── tests/                  # Comprehensive test suite (100+ tests)
    ├── test_model.py       # Architecture component tests
    ├── test_recursive.py   # Recursion and halting tests
    ├── test_training.py    # Training mechanics tests
    └── test_*.py           # Additional test modules
```

## Architecture

### Overview

The TRM achieves remarkable parameter efficiency through weight sharing: a single 2-layer transformer is applied recursively in a nested loop structure. With default hyperparameters (T=3 outer iterations, n=6 inner iterations), the network makes 21 forward passes per inference, equivalent to an effective depth of 42 transformer layers while maintaining only 10.5M parameters.

This recursive refinement approach demonstrates that reasoning depth and parameter count can be decoupled - the model can perform deep reasoning without requiring a physically deep network.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    TRM Architecture Flow                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Grid (x)     ──┐                                       │
│  [B, H, W]            │                                       │
│                       │                                       │
│  ┌────────────────────▼─────────────────────┐                │
│  │        Outer Loop (T=3 iterations)       │                │
│  │                                           │                │
│  │  ┌─────────────────────────────────────┐ │                │
│  │  │   Inner Loop (n=6 iterations)       │ │                │
│  │  │                                     │ │                │
│  │  │   Combine: x + y + z                │ │                │
│  │  │       ↓                             │ │                │
│  │  │   z = TRMNetwork(combined)          │ │                │
│  │  │       ↓                             │ │                │
│  │  │   (repeat n=6 times)                │ │                │
│  │  └─────────────────────────────────────┘ │                │
│  │                                           │                │
│  │  Combine: y + z (after inner loop)        │                │
│  │       ↓                                   │                │
│  │  y = TRMNetwork(combined)                 │                │
│  │       ↓                                   │                │
│  │  Check halting confidence ≥ threshold?   │                │
│  │       ↓                                   │                │
│  └───────────────────────────────────────────┘                │
│                       ↓                                       │
│  Output Grid (y)                                              │
│  [B, H, W, 10]                                                │
└──────────────────────────────────────────────────────────────┘
```

**Weight Sharing:** The same `TRMNetwork` instance is reused for all 21 forward passes (6 inner × 3 outer + 3 outer updates). This is the key innovation enabling parameter efficiency.

### TRMNetwork Components

| Component | Input Shape | Output Shape | Description |
|-----------|-------------|--------------|-------------|
| **GridEmbedding** | (B, H, W) | (B, H, W, 512) | Converts 10-color grid cells to 512-dimensional vectors. Uses 11 embeddings: 0-9 for ARC colors, 10 for padding. |
| **TRMStack** | (B, H×W, 512) | (B, H×W, 512) | 2-layer transformer with RMSNorm, multi-head self-attention (8 heads), SwiGLU activation, and rotary positional embeddings. |
| **OutputHead** | (B, H×W, 512) | (B, H×W, 10) | Linear projection from hidden states to 10-class color logits. No softmax (applied in loss). |
| **HaltingHead** | (B, H×W, 512) | (B,) | Global average pooling over sequence, then linear projection to scalar confidence score with sigmoid activation. |

### Recursive State Flow

The TRM maintains two evolving states during recursion:

- **z (latent state)**: Updated n=6 times per outer iteration. Accumulates intermediate reasoning. Reset to zeros at the start of each outer iteration.
- **y (answer state)**: Updated once per outer iteration. Represents the current answer prediction. Initialized to zeros and refined across T=3 outer iterations.

**State combination:** States are combined by converting logits to hard predictions (argmax), embedding them, and summing the embeddings element-wise:
- Inner loop: `combined = embed(x) + embed(argmax(y)) + embed(argmax(z))`
- Outer loop: `combined = embed(argmax(y)) + embed(argmax(z))`

### Paper Alignment

| Paper Specification | Implementation | Notes |
|---------------------|----------------|-------|
| ~7M parameters | 10.5M parameters | Higher due to hidden_dim=512; paper doesn't specify exact hyperparameters |
| 2-layer transformer | 2-layer TRMStack | Matches specification |
| RMSNorm | RMSNorm | No mean centering, weight-only normalization |
| SwiGLU activation | SwiGLU with 8/3 expansion | Gated activation in FFN layers |
| Rotary embeddings | RotaryEmbedding (2D) | Positional encoding for 2D grid structure |
| No bias | bias=False everywhere | All nn.Linear and nn.Embedding layers |
| T=3, n=6 | Default T=3, n=6 | Configurable via configs/config.yaml |
| Halting threshold | 0.9 (configurable) | Early stopping when confidence ≥ threshold |
| Deep supervision | Up to 16 steps | Intermediate losses at each outer iteration |
| EMA smoothing | 0.999 decay | Exponential moving average of weights |
| Data augmentation | D8 + color permutation | 8 geometric transforms × 3.6M color perms = 29M× |

### Key Implementation Details

**Grid-to-Sequence Transformation:** ARC grids are 2D, but transformers operate on sequences. The TRMNetwork flattens spatial dimensions (B, H, W) → (B, H×W) before the transformer and reshapes back (B, H×W, 10) → (B, H, W, 10) after the output head.

**Adaptive Halting:** After each outer iteration, the halting head produces a confidence score. If ALL batch items exceed the threshold (default 0.9), recursion stops early. This enables the model to use fewer iterations for easier problems.

**Deep Supervision:** During training, losses are computed at intermediate outer iterations (not just the final output). This provides more training signal and helps gradients flow through the recursive structure. Gradients are detached between iterations to prevent backpropagation through the entire recursion chain.

**Parameter Efficiency:** With 10.5M parameters and 21 forward passes, the TRM achieves effective depth comparable to a 42-layer transformer (which would require ~221M parameters with the same architecture).

## Training

### Basic Training

The TRM uses CrossEntropy loss for grid prediction and Binary CrossEntropy (BCE) for halting confidence. Training employs the AdamW optimizer with beta1=0.9, beta2=0.95, and weight_decay=0.01. In basic training mode, supervision is applied only to the final output (terminal supervision).

```bash
python scripts/train.py --epochs 50
```

### Deep Supervision (Recommended)

Deep supervision computes loss at each of up to 16 supervision steps during recursion, providing richer training signal throughout the recursive refinement process. This approach includes several key mechanisms for training stability:

- **Gradient detachment:** After each supervision step, `z.detach()` and `y.detach()` prevent backpropagation through previous iterations, keeping memory constant regardless of supervision depth
- **Loss normalization:** Total loss is divided by `steps_taken` to maintain consistent scale whether the model halts at step 1 or step 16
- **EMA weight smoothing:** Exponential moving average with decay=0.999 provides stable model weights for validation
- **Gradient clipping:** Max norm of 1.0 prevents gradient explosion during training

```bash
python scripts/train.py --deep-supervision --epochs 50 --patience 5
```

### Why Deep Supervision Works

Each supervision step provides a learning signal to the recursive refinement process. The critical innovation is gradient detachment: by applying `z.detach()` and `y.detach()` after computing each step's loss, we prevent gradients from flowing through the entire recursion chain. This means:

- Memory usage stays constant regardless of supervision depth
- Each step learns independently while still benefiting from the recursive structure
- Loss normalization by `steps_taken` ensures consistent loss scale across variable halting
- The model learns both to refine answers AND when to halt confidently

## Data Augmentation

The TRM employs two complementary augmentation strategies that can dramatically expand the effective training set size:

### Augmentation Strategies

| Transform | Multiplier | Description |
|-----------|------------|-------------|
| None | 1× | Original data only |
| D8 Dihedral | 8× | 4 rotations (0°, 90°, 180°, 270°) + 4 reflections (horizontal, vertical, diagonal, anti-diagonal) |
| Color Permutation | 3,628,800× | All permutations of 10 ARC colors (10! = 3,628,800) |
| Combined | 29,030,400× | D8 × Color = 8 × 3,628,800 = 29,030,400 variants per task |

### Usage Examples

```bash
# Enable all augmentation (recommended for full training)
python scripts/train.py --deep-supervision --augment --epochs 50

# D8 geometric transforms only (no color permutation)
python scripts/train.py --deep-supervision --augment --no-color

# Color permutation only (no D8 transforms)
python scripts/train.py --deep-supervision --augment --no-d8
```

### On-the-fly Augmentation

Augmentation is applied on-the-fly during training:

- Fresh random transformation generated for each `__getitem__` call
- Same task index returns different augmented versions across epochs
- Memory efficient: no need to store precomputed augmentation variants
- High diversity: with 29M possible variants, the model rarely sees identical examples

## Evaluation

### Exact-Match Accuracy

The TRM uses strict exact-match evaluation: each prediction receives 1.0 (100% correct) only if ALL non-padded pixels match the ground truth exactly, or 0.0 for any mismatch. There is no partial credit. Evaluation is performed on the ARC-AGI evaluation split containing 400 tasks.

When deep supervision is enabled, evaluation uses EMA (exponential moving average) weights rather than raw model weights, as EMA weights typically provide better generalization.

### Checkpointing

The training pipeline automatically saves checkpoints:

- **Best model tracking:** Automatically saves when validation accuracy improves
- **Complete state:** Checkpoint includes model weights, EMA weights (if enabled), optimizer state, and current epoch
- **Resume capability:** Training can be resumed from any checkpoint with `--resume checkpoints/best_model.pt`
- **Minimum delta:** Only saves if improvement exceeds threshold (prevents saving for negligible gains)

### Early Stopping

Training includes patience-based early stopping to prevent overfitting on the small ARC-AGI dataset:

- **Default patience:** 5 epochs (configurable via `--patience N`)
- **Counter reset:** Resets to 0 whenever validation accuracy improves
- **Prevents overfitting:** Stops training when model stops improving, avoiding wasted compute
- **Preserves best model:** Best checkpoint remains saved even after early stopping

## Visualization

The iteration timeline visualization shows how the model's answer evolves across recursive refinement steps.

### Generating a Timeline

```bash
python scripts/visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --task-id 007bbfb7 \
    --output timeline.png
```

### Understanding the Timeline

The visualization displays:
- **Top row**: Input puzzle (static reference)
- **Middle rows**: Model's answer at each outer iteration (1, 2, 3)
- **Bottom row**: Target solution (ground truth)

Color coding shows:
- **Green border**: Correct cell (matches target)
- **Red border**: Incorrect cell (doesn't match target)
- **Yellow border**: Changed from previous iteration

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint, -c` | Path to model checkpoint | Required |
| `--task-id, -t` | ARC task ID to visualize | Required |
| `--output, -o` | Output image path | `timeline.png` |
| `--pair-index, -p` | Train pair index | `0` |
| `--no-target` | Hide target grid | False |
| `--no-diff` | Disable color coding | False |
| `--dpi` | Image resolution | `150` |

## Hyperparameters

All hyperparameters are configured in `configs/config.yaml` and can be overridden via command-line arguments.

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 512 | Transformer hidden dimension |
| num_layers | 2 | Number of transformer layers |
| num_heads | 8 | Number of attention heads |
| outer_steps (T) | 3 | Outer recursion iterations |
| inner_steps (n) | 6 | Inner recursion iterations |
| halt_threshold | 0.9 | Halting confidence threshold |

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 1e-4 | AdamW learning rate |
| weight_decay | 0.01 | L2 regularization strength |
| beta1 | 0.9 | AdamW first moment decay |
| beta2 | 0.95 | AdamW second moment decay |
| ema_decay | 0.999 | EMA weight smoothing decay |
| max_sup_steps | 16 | Maximum deep supervision steps |
| grad_clip_norm | 1.0 | Gradient clipping threshold |
| batch_size | 32 | Training batch size |

## Results

**Note:** Full training results are pending. The training infrastructure is complete and tested.

### Expected Performance

Based on the original paper:
- Paper reports ~45% accuracy on ARC-AGI-1 with TRM
- Our implementation: To be determined (requires full training run)

### Training Infrastructure Validated

The following components have been implemented and tested:
- ✓ Loss decreases during training (gradient flow confirmed)
- ✓ Deep supervision loss normalization working correctly
- ✓ EMA weights tracked and saved in checkpoints
- ✓ Early stopping triggers correctly based on patience
- ✓ Checkpoints save and load complete state successfully

### Test Coverage

- 100+ tests across all modules
- Unit tests for each architecture component
- Integration tests for training pipeline
- All tests passing

## Paper Reference

This implementation is based on:

**"Less is More: Recursive Reasoning with Tiny Networks"**
- Authors: Samsung AI Research
- arXiv: [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)
- Published: 2025

### Key Contributions from the Paper

- **Tiny recursive networks** achieve performance comparable to much larger models
- **Weight sharing** enables deep effective computation (42 layers) with minimal parameters (10.5M)
- **Deep supervision** with gradient detachment enables stable training of recursive structures
- **Adaptive halting** mechanism allows variable computation depth based on problem difficulty

### Implementation Notes

This implementation faithfully reproduces the paper's architecture with the following considerations:

- **Parameter count:** Our implementation has ~10.5M parameters vs. paper's reported ~7M. This difference is due to hidden_dim=512; the paper doesn't specify exact hyperparameters, leading to ambiguity in reproduction.
- **Recursive structure:** Exact match with T=3 outer iterations and n=6 inner iterations (21 forward passes per inference).
- **Deep supervision:** Implements gradient detachment pattern as described in the paper.
- **Augmentation:** Implements D8 dihedral group and color permutation as suggested for ARC-AGI tasks.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes following conventional commit style
6. Push to your fork and submit a pull request

Please ensure all tests pass and add new tests for any new functionality.

## Acknowledgments

- **Samsung AI Research** for the innovative TRM architecture and paper
- **François Chollet** for creating and maintaining the ARC-AGI dataset
- **PyTorch team** for the deep learning framework that powers this implementation

