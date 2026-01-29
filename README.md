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

