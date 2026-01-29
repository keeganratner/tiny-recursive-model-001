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
