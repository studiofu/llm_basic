"""Step 2: Tokenizer - Text encoding/decoding and batch generation."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, get_config


class Tokenizer:
    """Character-level tokenizer for text encoding and decoding.
    
    Wraps the string-to-int (stoi) and int-to-string (itos) mappings
    from data preparation into a clean, reusable interface.
    
    Attributes:
        stoi: Dictionary mapping characters to integer indices.
        itos: Dictionary mapping integer indices to characters.
        vocab_size: Number of unique characters in the vocabulary.
    
    Example:
        >>> tokenizer = Tokenizer(stoi, itos)
        >>> tokens = tokenizer.encode("Hello")
        >>> text = tokenizer.decode(tokens)
    """
    
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        """Initialize tokenizer with vocabulary mappings.
        
        Args:
            stoi: Character to index mapping.
            itos: Index to character mapping.
        """
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
    
    def encode(self, text: str) -> list[int]:
        """Encode text into list of integer tokens.
        
        Args:
            text: Input string to encode.
        
        Returns:
            List of integer token IDs.
        
        Raises:
            KeyError: If text contains characters not in vocabulary.
        """
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            bad = next(c for c in text if c not in self.stoi)
            raise KeyError(
                f"Character {repr(bad)} (ord={ord(bad)}) not in vocabulary (size={self.vocab_size})"
            ) from e
    
    def decode(self, tokens: list[int]) -> str:
        """Decode list of integer tokens into text.
        
        Args:
            tokens: List of integer token IDs.
        
        Returns:
            Decoded string.
        """
        return "".join([self.itos[i] for i in tokens])


def get_batch(
    split: str,
    config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of data for training or validation.
    
    Randomly samples batch_size starting positions from the dataset,
    extracts block_size-length sequences for inputs, and block_size-length
    sequences shifted by 1 for targets.
    
    Memory: Only the batch (batch_size * block_size elements) is read and
    moved to the configured device; the full train_data/val_data stay on
    their original device. For very large datasets, keep train_data/val_data
    on CPU so the full tensor does not consume GPU memory.
    
    Args:
        split: "train" or "val" - which dataset to sample from.
        config: ModelConfig containing batch_size, block_size, and device.
        train_data: Encoded training data as 1D torch.Tensor.
        val_data: Encoded validation data as 1D torch.Tensor.
    
    Returns:
        (x, y) tuple where:
        - x: Input tensor of shape (batch_size, block_size)
        - y: Target tensor of shape (batch_size, block_size), shifted by 1
        Both tensors are on the configured device.
    
    Example:
        >>> x, y = get_batch("train", config, train_data, val_data)
        >>> x.shape
        torch.Size([32, 64])
        >>> y.shape
        torch.Size([32, 64])
        >>> # y[i] is the next character after x[i] at each position
    """
    data = train_data if split == "train" else val_data
    batch_size = config.batch_size
    block_size = config.block_size
    device = config.device

    n = len(data)
    if n <= block_size:
        raise ValueError(
            f"Data length ({n}) must be greater than block_size ({block_size}) to form (input, target) pairs."
        )

    # Random starting positions in [0, n - block_size)
    ix = torch.randint(0, n - block_size, (batch_size,), device=data.device)
    # Vectorized indexing: (batch_size, block_size) in one go
    offsets = torch.arange(block_size, device=data.device)
    x = data[ix.unsqueeze(1) + offsets]  # (batch_size, block_size)
    y = data[ix.unsqueeze(1) + offsets + 1]  # targets shifted by 1

    x, y = x.to(device), y.to(device)
    return x, y


class TextDataset(Dataset):
    """PyTorch Dataset for text data with sliding window.
    
    Creates (input, target) pairs where target is input shifted by 1.
    Used with DataLoader for iterable-style data loading.
    
    Attributes:
        data: Encoded text as 1D torch.Tensor.
        block_size: Length of each sequence window.
    
    Example:
        >>> dataset = TextDataset(encoded_text, block_size=64)
        >>> x, y = dataset[0]  # First window
        >>> len(dataset)  # Number of possible windows
    """
    
    def __init__(self, data: torch.Tensor, block_size: int):
        """Initialize dataset.
        
        Args:
            data: Encoded text as 1D torch.Tensor.
            block_size: Length of each sequence window.
        """
        self.data = data
        self.block_size = block_size
    
    def __len__(self) -> int:
        """Return number of possible windows in the dataset."""
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single (input, target) pair at the given index.
        
        Args:
            idx: Starting position in the data.
        
        Returns:
            (x, y) tuple where x is input and y is target (shifted by 1).
        """
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


def create_dataloaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ModelConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation.
    
    Alternative to get_batch() for users who prefer DataLoader workflow.
    Note: get_batch() is typically used for training loops in this project.
    
    Args:
        train_data: Encoded training data as 1D torch.Tensor.
        val_data: Encoded validation data as 1D torch.Tensor.
        config: ModelConfig with batch_size and block_size.
    
    Returns:
        (train_loader, val_loader) tuple of PyTorch DataLoaders.
    
    Note:
        Both loaders use drop_last=True, so if len(data) < batch_size
        the loader will have 0 batches. Ensure data is long enough.
    """
    train_dataset = TextDataset(train_data, config.block_size)
    val_dataset = TextDataset(val_data, config.block_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility across platforms
        drop_last=True,  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    
    return train_loader, val_loader


def _create_test_data():
    """Create minimal test data for standalone testing."""
    # Simple story text for testing
    train_text = """Once upon a time, in a land far away, there lived a young princess named Aurora.
She loved to explore the enchanted forest near her castle. One day, while wandering
through the woods, she discovered a hidden cave behind a waterfall. Inside the cave
she found a magical book that could grant wishes.""" * 100  # Repeat to get more data
    
    val_text = """The little rabbit hopped through the meadow, searching for carrots.
He found a big orange carrot and was very happy.""" * 20
    
    # Create vocabulary from combined text
    chars = sorted(set(train_text + val_text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return train_text, val_text, stoi, itos


if __name__ == "__main__":
    # Test script - run to verify tokenizer and batch generation work correctly
    print("=" * 60)
    print("STEP 2: TOKENIZER - TESTING")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    
    # Create test data (avoid loading full 2GB dataset)
    print("\n1. Creating test data...")
    train_text, val_text, stoi, itos = _create_test_data()
    print(f"   Train text length: {len(train_text):,} chars")
    print(f"   Val text length:   {len(val_text):,} chars")
    print(f"   Vocabulary size:   {len(stoi)}")
    
    # Create tokenizer
    print("\n2. Creating tokenizer...")
    tokenizer = Tokenizer(stoi, itos)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encode/decode
    print("\n3. Testing encode/decode...")
    test_text = "Once upon a time"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   Original: '{test_text}'")
    print(f"   Tokens:   {tokens}")
    print(f"   Decoded:  '{decoded}'")
    assert test_text == decoded, "Encode/decode round-trip failed!"
    print("   OK Match: True")
    
    # Test with larger sample
    print("\n4. Testing with larger sample...")
    sample_text = train_text[100:300]  # 200 characters
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"   Sample length: {len(sample_text)} characters")
    print(f"   First 50 chars: {sample_text[:50]}...")
    assert sample_text == decoded, "Large sample round-trip failed!"
    print("   OK Round-trip match: True")
    
    # Prepare tensor data for batch testing
    print("\n5. Preparing tensor data...")
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    print(f"   Train data shape: {train_data.shape}")
    print(f"   Val data shape:   {val_data.shape}")
    
    # Test get_batch
    print("\n6. Testing get_batch()...")
    print(f"   Config: batch_size={config.batch_size}, block_size={config.block_size}")
    print(f"   Device: {config.device}")
    
    x, y = get_batch("train", config, train_data, val_data)
    print(f"   Input shape (x):  {x.shape}")
    print(f"   Target shape (y): {y.shape}")
    assert x.shape == (config.batch_size, config.block_size), "Input shape mismatch!"
    assert y.shape == (config.batch_size, config.block_size), "Target shape mismatch!"
    print("   OK Shapes correct: True")
    
    # Verify input-target relationship
    print("\n7. Verifying input-target relationship...")
    print("   First sequence of batch:")
    print(f"   Input (x[0]):  {tokenizer.decode(x[0].tolist()[:30])}...")
    print(f"   Target (y[0]): {tokenizer.decode(y[0].tolist()[:30])}...")
    
    # Verify targets are shifted by 1
    input_chars = tokenizer.decode(x[0].tolist()[:10])
    target_chars = tokenizer.decode(y[0].tolist()[:10])
    shift_correct = all(input_chars[i+1] == target_chars[i] for i in range(9))
    assert shift_correct, "Target is not shifted by 1!"
    print("   OK Target is input shifted by 1: True")
    
    # Test validation batch
    print("\n8. Testing validation batch...")
    x_val, y_val = get_batch("val", config, train_data, val_data)
    print(f"   Val batch shape: {x_val.shape}")
    assert x_val.shape == (config.batch_size, config.block_size), "Val shape mismatch!"
    print("   OK Validation batch works: True")
    
    # Test DataLoader (optional)
    print("\n9. Testing DataLoader...")
    train_loader, val_loader = create_dataloaders(train_data, val_data, config)
    x_dl, y_dl = next(iter(train_loader))
    print(f"   DataLoader batch shape: {x_dl.shape}")
    assert x_dl.shape[0] == config.batch_size, "DataLoader batch size mismatch!"
    print("   OK DataLoader works: True")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED OK")
    print("=" * 60)
