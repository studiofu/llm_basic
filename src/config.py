"""Centralized hyperparameters and configuration for the LLM project."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the miniature LLM.
    
    This dataclass contains all hyperparameters and settings for the model,
    training, and generation. Modify values here to experiment with different
    configurations.
    """
    
    # Data files
    data_dir: str = "data"                                    # Directory containing data files
    train_file: str = "TinyStoriesV2-GPT4-train.txt"         # Training data filename
    val_file: str = "TinyStoriesV2-GPT4-valid.txt"           # Validation data filename
    
    # Data
    block_size: int = 64        # Maximum context length (how many characters the model sees)
    batch_size: int = 32        # Sequences per batch (parallel training examples)

    # Model architecture
    vocab_size: int = 0         # Set from data (number of unique characters)
    n_embd: int = 128           # Embedding dimension (richness of each character's representation)
    n_head: int = 4             # Number of attention heads (parallel attention patterns)
    n_layer: int = 6            # Number of transformer blocks (depth of processing)
    dropout: float = 0.1        # Dropout rate (prevents overfitting)

    # Training
    max_iters: int = 10000       # Total training iterations
    eval_interval: int = 250    # Evaluation frequency (check progress every N steps)
    eval_iters: int = 200       # Batches per evaluation (how thorough each check is)
    learning_rate: float = 3e-4 # Peak learning rate (step size for adjustments)
    min_lr: float = 3e-5        # Minimum learning rate (cosine decay end point)
    warmup_iters: int = 100     # Linear warmup steps (gentle start)
    weight_decay: float = 0.1   # AdamW weight decay (prevents large weights)

    # Modern upgrades (flags - enable in Step 6)
    use_rope: bool = False      # Rotary Position Embeddings (modern position encoding)
    use_rmsnorm: bool = False   # RMSNorm instead of LayerNorm (modern normalization)
    use_swiglu: bool = False    # SwiGLU instead of ReLU (modern activation)
    use_kv_cache: bool = False  # KV-Cache for inference (faster generation)
    use_gqa: bool = False       # Grouped Query Attention (modern attention optimization)

    # Generation
    temperature: float = 0.8    # Creativity dial (higher = more random)
    top_k: int = 50             # Only consider top K candidates
    top_p: float = 0.9          # Nucleus sampling (smart cutoff)
    repetition_penalty: float = 1.1  # Reduce probability of recent tokens
    max_new_tokens: int = 200   # Maximum tokens to generate

    # System
    device: str = "auto"        # "auto", "cuda", "mps", "cpu"
    compile_model: bool = False # torch.compile (PyTorch 2.0+ optimization)
    seed: int = 42              # Random seed for reproducibility

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


def get_config() -> ModelConfig:
    """Get the default configuration.
    
    Returns:
        ModelConfig: A configuration instance with default values.
    """
    return ModelConfig()


if __name__ == "__main__":
    # Print configuration when run directly
    config = get_config()
    print("Model Configuration:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
