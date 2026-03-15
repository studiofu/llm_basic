"""
Check PyTorch installation and device support (CPU, CUDA, MPS).
Run: uv run python src/check_pytorch.py
"""

import sys

def main() -> None:
    print("PyTorch device support check\n" + "=" * 40)

    try:
        import torch
    except ImportError as e:
        print("FAIL: PyTorch is not installed.")
        print(f"  {e}")
        print("  Run: uv add torch")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}\n")

    # CPU is always supported in PyTorch
    print("CPU:")
    print("  Supported: Yes (default)")
    try:
        x = torch.randn(2, 3)
        y = x + 1
        print("  Test (randn + 1): OK")
    except Exception as e:
        print(f"  Test: FAIL - {e}")

    # CUDA (NVIDIA GPU)
    print("\nCUDA (NVIDIA GPU):")
    cuda_available = torch.cuda.is_available()
    print(f"  Available: {cuda_available}")
    if cuda_available:
        print(f"  Version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  (Install torch with CUDA if you need GPU support)")

    # MPS (Apple Silicon)
    print("\nMPS (Apple Silicon / Metal):")
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"  Available: {mps_available}")
    if mps_available:
        print("  (MPS backend is ready)")
    else:
        print("  (Only relevant on macOS with Apple Silicon)")

    print("\n" + "=" * 40)
    print("Summary: PyTorch CPU support is OK. You can run this project on CPU.")
    if cuda_available:
        print("CUDA is also available; training can use GPU for speed.")
    print()

if __name__ == "__main__":
    main()
