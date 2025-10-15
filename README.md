# Transformer From Scratch

A flexible, modular implementation of transformer architectures in PyTorch, built from the ground up to support various transformer-based operations and applications. Long-term goal is to implement a code-generation model for CUDA kernel synthesis.

## Current Status

The repository currently features a Vision Transformer (ViT) implementation trained on CIFAR-100, achieving approximately **60% validation accuracy**. This performance is expected for several reasons:

- **Small image resolution**: CIFAR-100 images are only 32×32 pixels, which provides limited spatial information
- **Lack of inductive bias**: Unlike CNNs, transformers don't have built-in assumptions about spatial locality and translation invariance, which can be disadvantageous for small images
- **Limited training data**: CIFAR-100 has only 50,000 training images across 100 classes (500 per class)

Despite these challenges, the implementation demonstrates that the core transformer architecture is working correctly.

## Future Goals

This repository is being developed with a long-term vision to support general-purpose transformer operations. The ultimate goal is to:

1. **Expand beyond vision**: Add support for text, multimodal, and other transformer applications
2. **Build a code generation model**: Train a small transformer model specifically for code generation
3. **Specialize in CUDA kernel generation**: Fine-tune the model to generate optimized CUDA kernels
4. **Reinforcement learning**: Improve kernel generation through RL-based optimization, potentially using:
   - Execution time as reward signal
   - Correctness verification
   - Compiler feedback
5. **Modular architecture**: Maintain flexibility to experiment with different architectures, training techniques, and applications

## Architecture Overview

The codebase is designed with modularity and extensibility in mind:

### Core Components

- **`transformer.py`**: Core transformer implementation
  - Multi-head self-attention mechanism
  - Flexible feed-forward networks
  - Configurable normalization (RMSNorm, LayerNorm) and positioning (pre/post-norm)
  - Support for attention masking

- **`tokenizers.py`**: Input tokenization modules
  - `ImageTokenizer`: Patches images into tokens using convolutional projection
  - Learnable CLS tokens and positional embeddings

- **`heads.py`**: Task-specific output heads
  - `ViTHead`: Classification head for vision tasks

- **`training.py`**: Training infrastructure
  - Flexible trainer with validation support
  - Label smoothing
  - Integration with PyTorch optimizers and schedulers

- **`norms.py`**: Normalization layers (RMSNorm, LayerNorm)

- **`defaults.py`**: Pre-configured model setups (e.g., CIFAR-100 ViT)

## Features

- **Modular design**: Easy to swap components (tokenizers, heads, normalization)
- **Flexible configuration**: Control model depth, width, attention heads, and more
- **Multiple activation functions**: GELU, ReLU, or no activation
- **Training utilities**: Built-in trainer with progress tracking and evaluation
- **Data augmentation**: Configurable data loading with normalization and augmentation

## Model Architecture Details

The default CIFAR-100 configuration uses:

- **8 transformer layers**
- **384-dimensional embeddings**
- **12 attention heads** with 32-dimensional keys/queries and values
- **4× expansion ratio** in feed-forward layers (1536 dimensions)
- **RMSNorm** with pre-normalization
- **4×4 patch size** (64 total patches from 32×32 images)
- **GELU activation**
- **AdamW optimizer** with cosine annealing and linear warmup
- **Label smoothing** (0.1) for regularization


## Project Structure

```
.
├── transformer.py          # Core transformer architecture
├── tokenizers.py           # Input tokenization modules
├── heads.py               # Task-specific output heads
├── training.py            # Training and evaluation logic
├── norms.py               # Normalization layers
├── image_dataloader.py    # Data loading utilities
├── defaults.py            # Pre-configured model setups
└── main.py                # Example training script
```

## Requirements

- PyTorch
- torchvision
- tqdm
- CUDA (optional, for GPU acceleration)

## License

MIT