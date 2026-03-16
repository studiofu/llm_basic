# Cross-Platform PyTorch Setup Guide

This project **auto-detects** your platform and installs the correct PyTorch version:
- **macOS** → PyTorch with MPS (Apple Silicon GPU) support
- **Windows/Linux + NVIDIA GPU** → PyTorch with CUDA 12.4 support
- **Any platform without GPU** → PyTorch CPU version

## Quick Start (Zero-Config!)

On **any** platform, just run:

```bash
uv sync
```

That's it! The correct PyTorch will be installed automatically.

### Verify Installation

```bash
uv run python src/check_pytorch.py
```

## Platform-Specific Output

### macOS with Apple Silicon (M1/M2/M3/M4)
```
PyTorch version: 2.6.0
MPS (Apple Silicon): Available: True
Recommended device: mps
✓ Apple Silicon GPU will be used for training
```

### Windows/Linux with NVIDIA GPU
```
PyTorch version: 2.6.0+cu124
CUDA (NVIDIA GPU): Available: True
  Device: NVIDIA GeForce RTX 4090
Recommended device: cuda
✓ NVIDIA GPU will be used for training
```

### Any Platform (CPU Only)
```
PyTorch version: 2.6.0
Recommended device: cpu
⚠ CPU will be used (slower training)
```

## Device Detection in Code

The project automatically detects the best available device:

```python
from src.config import get_config

config = get_config()  # device="auto" by default
print(config.device)   # "cuda", "mps", or "cpu"

model = create_model(config, vocab_size)
model = model.to(config.device)  # Automatically uses best device
```

Or use the helper function directly:

```python
from src.check_pytorch import get_device

# Auto-detect best device
device = get_device("auto")  # Returns "cuda", "mps", or "cpu"

# Force specific device (falls back to CPU if unavailable)
device = get_device("cuda")
device = get_device("mps")
device = get_device("cpu")
```

## Configuration

Edit `src/config.py` to change device behavior:

```python
@dataclass
class ModelConfig:
    # ... other settings ...
    device: str = "auto"    # "auto", "cuda", "mps", or "cpu"
```

## Troubleshooting

### "CUDA not available" on Windows with NVIDIA GPU

The project should auto-install CUDA-enabled PyTorch. If not:

```bash
# Force reinstall with CUDA
uv add torch --index https://download.pytorch.org/whl/cu124 --force-reinstall
```

Check your NVIDIA driver:
```bash
nvidia-smi
```

### "MPS not available" on macOS

Requirements:
- macOS 12.3 or later
- Apple Silicon Mac (M1/M2/M3/M4)

Intel Macs only support CPU (this is a hardware limitation).

### Slow training

If training is slow on CPU:

1. Reduce batch size in `src/config.py`:
   ```python
   batch_size = 16  # or 8
   ```

2. Reduce model size:
   ```python
   n_embd = 64    # Smaller embeddings
   n_layer = 4    # Fewer transformer layers
   ```

## How It Works

The `pyproject.toml` uses platform markers:

```toml
[tool.uv.sources]
torch = [
    # Windows/Linux: Install CUDA version from PyTorch index
    { index = "pytorch-cu124", marker = "sys_platform != 'darwin'" },
    # macOS: Install from PyPI (MPS support built-in)
]
```

- `sys_platform != 'darwin'` matches Windows and Linux → CUDA version
- No match on macOS → Falls back to default PyPI → MPS version
