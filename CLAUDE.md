# Claude Context: LLM Basic Project

> This file provides context for Claude when working on this project.

## Project Overview

This is an **educational project** to build a miniature Large Language Model from scratch using PyTorch. The goal is to understand how modern LLMs (GPT, LLaMA, etc.) work by implementing them step-by-step.

- **Approach**: Progressive enhancement from classical (2017) to modern (2024) techniques
- **Dataset**: TinyStories (2.2B chars training, 22M validation, 230 unique characters)
- **Model Size**: ~1.2M parameters (tiny but functional)
- **Package Manager**: `uv` (modern Python package manager)

## Current Implementation Status

| Step | File | Status | Description |
|------|------|--------|-------------|
| 0 | `src/config.py` | ✅ Done | Centralized hyperparameters (ModelConfig dataclass) |
| 1 | `src/01_data_preparation.py` | ✅ Done | Load data, create vocab mappings (stoi/itos) |
| 2 | `src/02_tokenizer.py` | ✅ Done | Tokenizer class, get_batch(), TextDataset |
| 3 | `src/03_model.py` | ⏳ Pending | Classical Transformer (embeddings, attention, blocks) |
| 3b | `src/03_model_modern.py` | ⏳ Pending | Modern upgrades (RoPE, RMSNorm, SwiGLU, GQA) |
| 4 | `src/04_train.py` | ⏳ Pending | Training loop with AdamW, lr scheduling |
| 5 | `src/05_inference.py` | ⏳ Pending | Generation with sampling strategies |
| 6 | `src/06_visualize.py` | ⏳ Pending | Attention maps, embeddings, loss curves |
| 7 | `src/07_app.py` | ⏳ Pending | Gradio web UI |

## Key Configuration (from `src/config.py`)

```python
block_size: int = 64        # Context length (max 64 chars)
batch_size: int = 32        # Sequences per batch
vocab_size: int = 0         # Set from data (230 for TinyStories)
n_embd: int = 128           # Embedding dimension
n_head: int = 4             # Attention heads
n_layer: int = 6            # Transformer blocks
dropout: float = 0.1        # Regularization

# Training
max_iters: int = 5000
learning_rate: float = 3e-4

# Modern upgrade flags (all False by default)
use_rope: bool = False
use_rmsnorm: bool = False
use_swiglu: bool = False
use_kv_cache: bool = False
use_gqa: bool = False
```

## How to Run Code

This project uses `uv` (not pip/conda). Always use these commands:

```bash
# Run a Python file
uv run python src/02_tokenizer.py

# Install/sync dependencies
uv sync

# Add a new package
uv add <package-name>
```

**Note**: On Windows PowerShell, use `;` instead of `&&` for command chaining.

## Project Structure Conventions

- **Source files**: `src/XX_descriptive_name.py` (numbered by step)
- **Data**: `data/` (contains TinyStories files, not committed)
- **Models**: `models/` (checkpoints saved here)
- **Outputs**: `outputs/` (visualizations, samples)
- **Tests**: Each module has `if __name__ == "__main__"` test block

## Important Design Decisions

1. **Character-level tokenization** (not BPE) - simpler for learning
2. **Pre-norm architecture** (LayerNorm before attention/FFN) - more stable
3. **Causal (autoregressive) masking** - essential for text generation
4. **Device auto-detection** - config picks cuda/mps/cpu automatically
5. **Deterministic seed** (42) - reproducible results

## Next Steps (When Continuing)

### Step 3: Classical Transformer Model

Implement `src/03_model.py` with:

1. **Token + Position Embeddings** (`nn.Embedding` tables)
2. **Single Attention Head** (Q, K, V projections, causal mask)
3. **Multi-Head Attention** (4 heads, concat + projection)
4. **Feed-Forward Network** (expand 4x, ReLU, project back)
5. **Transformer Block** (pre-norm + residual connections)
6. **Full GPT Model** (stack of blocks + final linear)

Key formula: `Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V`

Refer to PLAN.md lines 1148-1263 for detailed architecture specifications.

## Testing Guidelines

- Each module should be runnable standalone: `uv run python src/XX_file.py`
- Use small synthetic data for tests (not full 2GB dataset)
- Include assertions for shape checks and round-trip verification
- Print clear OK/FAIL status

## Dependencies (from `pyproject.toml`)

```toml
dependencies = [
    "torch>=2.2",
    "numpy",
    "tqdm",
    "matplotlib",
    "seaborn",
]
```

## Dataset Information

- **Source**: TinyStoriesV2-GPT4 (HuggingFace)
- **Files**: 
  - `data/TinyStoriesV2-GPT4-train.txt` (2.2B chars)
  - `data/TinyStoriesV2-GPT4-valid.txt` (22M chars)
- **Vocab**: 230 unique characters (ASCII + some Unicode)

## References

- PLAN.md - Complete implementation guide with detailed explanations
- nanoGPT (Karpathy) - Minimal GPT reference implementation
- "Attention Is All You Need" (2017) - Original Transformer paper
