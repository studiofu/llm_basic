"""Step 1: Data Preparation - Load, analyze, and prepare the story dataset."""

from pathlib import Path


def prepare_data():
    """Load data, create vocabulary mappings, and return processed data."""
    
    # Load data files
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    val_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    
    print("Loading data...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()
    
    # Analyze vocabulary
    print("Analyzing vocabulary...")
    chars = sorted(list(set(train_text + val_text)))
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
    
    return train_text, val_text, stoi, itos, vocab_size


if __name__ == "__main__":
    prepare_data()
