"""Step 1: Data Preparation - Load, analyze, and prepare the story dataset."""

from pathlib import Path

from config import get_config


def prepare_data(config=None):
    """Load data, create vocabulary mappings, and return processed data.

    Args:
        config: Optional ModelConfig. If None, uses default config.

    Returns:
        train_text: Raw training text.
        val_text: Raw validation text.
        stoi: Character to index mapping (dict).
        itos: Index to character mapping (dict).
        vocab_size: Number of unique characters.
        encode: Function text -> list[int].
        decode: Function list[int] -> text.
    """
    if config is None:
        config = get_config()
    
    # Load data files from config
    data_dir = Path(__file__).parent.parent / config.data_dir
    train_path = data_dir / config.train_file
    val_path = data_dir / config.val_file
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    print("Loading data...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()
    
    # Analyze vocabulary (combined so vocab covers both splits)
    print("Analyzing vocabulary...")
    chars = sorted(set(train_text + val_text))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
    itos = {i: ch for i, ch in enumerate(chars)}  # int to string
    
    # Encode/decode functions
    def encode(text):
        return [stoi[c] for c in text]
    
    def decode(tokens):
        return "".join([itos[i] for i in tokens])
    
    # Print statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Training characters:   {len(train_text):,}")
    print(f"Validation characters: {len(val_text):,}")
    print(f"Vocabulary size:       {vocab_size}")
    
    # Show some characters
    print(f"\nSome characters: {chars[:50]}")
    
    # Verify encode/decode
    sample = train_text[10000:10100]
    encoded = encode(sample)
    decoded = decode(encoded)
    
    print("\n" + "=" * 50)
    print("ENCODE/DECODE VERIFICATION")
    print("=" * 50)
    print(f"Original:  {sample[:50]}...")
    print(f"Decoded:   {decoded[:50]}...")
    print(f"Match:     {sample == decoded}")
    assert sample == decoded, "Encode/decode round-trip failed"

    return train_text, val_text, stoi, itos, vocab_size, encode, decode


if __name__ == "__main__":
    prepare_data()
