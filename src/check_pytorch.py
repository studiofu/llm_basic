"""
Check PyTorch installation and device support (CPU, CUDA, MPS).
Run: uv run python src/check_pytorch.py
"""

import sys


def get_device(prefer: str = "auto") -> str:
    """Get the best available device for the current platform.
    
    Priority order when auto:
        1. CUDA (NVIDIA GPU) - fastest on Windows/Linux
        2. MPS (Apple Silicon) - fastest on macOS
        3. CPU - works everywhere
    
    Args:
        prefer: Preferred device ("auto", "cuda", "mps", "cpu")
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    
    Examples:
        >>> device = get_device("auto")  # Auto-detect best device
        >>> device = get_device("cuda")  # Force CUDA or fallback to CPU
        >>> model = model.to(device)
    """
    import torch
    
    if prefer != "auto":
        # Validate user preference
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif prefer == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_device(device: str) -> bool:
    """Test if a device is working by running a simple tensor operation.
    
    Args:
        device: Device to test ("cuda", "mps", "cpu")
    
    Returns:
        True if device works, False otherwise
    """
    import torch
    
    try:
        x = torch.randn(2, 3).to(device)
        y = x + 1
        _ = y.sum().item()  # Force sync
        return True
    except Exception as e:
        print(f"  Device test failed: {e}")
        return False


def main() -> None:
    print("PyTorch Device Check\n" + "=" * 50)

    # Import torch
    try:
        import torch
    except ImportError as e:
        print("❌ PyTorch is not installed.")
        print(f"   Error: {e}")
        print("   Install: uv add torch")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}\n")

    # Check CPU (always available)
    print("CPU:")
    print("  Available: Yes (always)")
    if check_device("cpu"):
        print("  Test: ✓ OK")
    else:
        print("  Test: ❌ FAIL")

    # Check CUDA (NVIDIA GPU on Windows/Linux)
    print("\nCUDA (NVIDIA GPU):")
    cuda_available = torch.cuda.is_available()
    print(f"  Available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        if check_device("cuda"):
            print("  Test: ✓ OK")
        else:
            print("  Test: ❌ FAIL")
    else:
        print("  Note: Install torch with CUDA for NVIDIA GPU support on Windows/Linux")

    # Check MPS (Apple Silicon on macOS)
    print("\nMPS (Apple Silicon / Metal):")
    mps_available = torch.backends.mps.is_available()
    print(f"  Available: {mps_available}")
    if mps_available:
        print("  Backend: Metal Performance Shaders")
        if check_device("mps"):
            print("  Test: ✓ OK")
        else:
            print("  Test: ❌ FAIL")
    else:
        print("  Note: Only available on macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)")

    # Summary
    print("\n" + "=" * 50)
    device = get_device("auto")
    
    print(f"Recommended device: {device}")
    if device == "cuda":
        print("✓ NVIDIA GPU will be used for training (fastest)")
    elif device == "mps":
        print("✓ Apple Silicon GPU will be used for training (fast)")
    else:
        print("⚠ CPU will be used (slower training)")
        print("  Tip: Reduce batch_size in config.py if training is too slow")
    
    print("\n" + "-" * 50)
    print("Usage in your code:")
    print(f'  from check_pytorch import get_device')
    print(f'  device = get_device("auto")  # Returns "{device}"')
    print(f'  model = model.to(device)')
    print(f'  tensor = tensor.to(device)')
    print()


if __name__ == "__main__":
    main()
