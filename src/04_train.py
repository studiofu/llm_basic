"""Step 4: Training Loop - Teach the transformer model with backpropagation.

This script implements the complete training pipeline:
- Load data and create vocabulary mappings
- Initialize model and AdamW optimizer
- Training loop with forward/backward passes
- Learning rate warmup + cosine decay scheduling
- Periodic evaluation on validation set
- Save best checkpoint based on validation loss
- Generate sample text to monitor progress

Example:
    $ uv run python src/04_train.py
    
    Output:
    Iter    0 | Train loss: 4.174 | Val loss: 4.171 | LR: 0.000003
    Iter  250 | Train loss: 2.483 | Val loss: 2.511 | LR: 0.000300
    Sample @ iter 250: "Once upon a time there was a..."
    ...
    Best model saved to models/tinystories_best.pt
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ModelConfig, get_config

# Import modules with numeric prefixes using importlib
def _import_module(module_name: str, file_name: str):
    """Import a module by file path."""
    src_dir = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        module_name, src_dir / file_name
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import step modules
_data_prep = _import_module("data_preparation", "01_data_preparation.py")
_tokenizer = _import_module("tokenizer", "02_tokenizer.py")
_model = _import_module("model", "03_model.py")

# Expose needed classes/functions
prepare_data = _data_prep.prepare_data
Tokenizer = _tokenizer.Tokenizer
get_batch = _tokenizer.get_batch
GPTLanguageModel = _model.GPTLanguageModel
create_model = _model.create_model


def get_lr(iter_num: int, config: ModelConfig) -> float:
    """Calculate learning rate with warmup and cosine decay.
    
    Schedule:
    1. Warmup: Linear increase from near-zero to max_lr over warmup_iters
    2. Cosine decay: Decay from max_lr to min_lr over remaining iterations
    
    Args:
        iter_num: Current iteration number
        config: ModelConfig with learning_rate, min_lr, warmup_iters, max_iters
    
    Returns:
        Learning rate for current iteration
    """
    # Warmup phase: linear increase
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)
    
    # After max_iters: return minimum learning rate
    if iter_num > config.max_iters:
        return config.min_lr
    
    # Cosine decay phase
    decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    # Cosine decay from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Return learning rate between min_lr and max_lr
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict[str, float]:
    """Estimate loss on train and validation sets.
    
    Runs multiple evaluation batches and returns average loss for each split.
    This gives a more stable estimate than a single batch.
    
    Args:
        model: The GPT language model
        config: ModelConfig with eval_iters
        train_data: Encoded training data tensor
        val_data: Encoded validation data tensor
    
    Returns:
        Dictionary with 'train' and 'val' loss values
    """
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        split_losses = torch.zeros(config.eval_iters, device=config.device)
        
        for i in range(config.eval_iters):
            x, y = get_batch(split, config, train_data, val_data)
            _, loss = model(x, y)
            split_losses[i] = loss.item()
        
        losses[split] = split_losses.mean().item()
    
    model.train()
    return losses


def generate_sample(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    config: ModelConfig,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 100,
) -> str:
    """Generate a text sample from the model.
    
    Args:
        model: The GPT language model
        tokenizer: Tokenizer for encoding/decoding
        config: ModelConfig with device
        prompt: Starting text for generation
        max_new_tokens: Number of tokens to generate
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt and move to device
    start_tokens = tokenizer.encode(prompt)
    idx = torch.tensor([start_tokens], dtype=torch.long, device=config.device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
    
    # Decode and return
    generated_text = tokenizer.decode(generated[0].tolist())
    
    # model.train() is a method provided by PyTorch's nn.Module.
    # It sets the model in training mode, enabling certain layers like dropout and batch normalization to behave accordingly.
    # In contrast, model.eval() puts the model in evaluation mode, which disables dropout and uses running stats in batchnorm.
    model.train()
    return generated_text


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    iter_num: int,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: The GPT language model
        optimizer: The optimizer instance
        config: ModelConfig
        iter_num: Current iteration number
        val_loss: Validation loss at current iteration
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'iter_num': iter_num,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: The GPT language model to load into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Tuple of (iteration number, validation loss)
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iter_num = checkpoint.get('iter_num', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    return iter_num, val_loss


def train(config: ModelConfig | None = None) -> None:
    """Run the complete training loop.
    
    This function coordinates all training components:
    1. Load and prepare data
    2. Initialize model and optimizer
    3. Run training iterations with progress bar
    4. Evaluate periodically and save best checkpoint
    5. Generate samples to monitor quality
    
    Args:
        config: Optional ModelConfig. Uses default if not provided.
    """
    if config is None:
        config = get_config()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    print("=" * 60)
    print("STEP 4: TRAINING")
    print("=" * 60)
    
    # ========================================================================
    # 1. Load and Prepare Data
    # ========================================================================
    print("\n[1/5] Loading and preparing data...")
    
    train_text, val_text, stoi, itos, vocab_size, encode, decode = prepare_data(config)
    
    # Create tokenizer
    tokenizer = Tokenizer(stoi, itos)
    
    # Encode text to tensors (keep on CPU, move batches to GPU during training)
    print("  Encoding text to tensors...")
    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    val_data = torch.tensor(encode(val_text), dtype=torch.long)
    
    print(f"  Train data: {train_data.shape}")
    print(f"  Val data:   {val_data.shape}")
    
    # ========================================================================
    # 2. Initialize Model and Optimizer
    # ========================================================================
    print("\n[2/5] Initializing model...")
    
    # Create model with proper vocab size
    config.vocab_size = vocab_size
    model = create_model(config, vocab_size)
    model = model.to(config.device)
    
    # Print model info
    total_params = model.count_parameters()
    print(f"  Device: {config.device}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Block size: {config.block_size}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # ========================================================================
    # 3. Setup Checkpoint Directory
    # ========================================================================
    print("\n[3/5] Setting up checkpoint directory...")
    
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "models"
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_checkpoint_path = checkpoint_dir / "tinystories_best.pt"
    print(f"  Checkpoints will be saved to: {checkpoint_dir}")
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # ========================================================================
    # 4. Training Loop
    # ========================================================================
    print("\n[4/5] Starting training...")
    print(f"  Training for {config.max_iters} iterations")
    print(f"  Evaluating every {config.eval_interval} iterations")
    print(f"  Warmup iterations: {config.warmup_iters}")
    print("=" * 60)
    
    # Progress bar for training
    pbar = tqdm(range(config.max_iters), desc="Training", ncols=80)
    
    for iter_num in pbar:
        # Determine and set learning rate for this iteration
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample a batch of data
        x, y = get_batch('train', config, train_data, val_data)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
        
        # =====================================================================
        # Evaluation and Logging
        # =====================================================================
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            # Estimate loss on train and val
            losses = estimate_loss(model, config, train_data, val_data)
            
            # Print progress
            print(f"\nIter {iter_num:5d} | "
                  f"Train loss: {losses['train']:.3f} | "
                  f"Val loss: {losses['val']:.3f} | "
                  f"LR: {lr:.6f}")
            
            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(
                    model, optimizer, config, iter_num, best_val_loss, best_checkpoint_path
                )
                print(f"  ✓ New best val loss: {best_val_loss:.4f} - Checkpoint saved")
            
            # Generate sample text
            if iter_num > 0:  # Skip generation at iteration 0 (random model)
                sample = generate_sample(
                    model, tokenizer, config,
                    prompt="Once upon a time",
                    max_new_tokens=100,
                )
                # Show first 150 chars of generated sample
                print(f"  Sample: {sample[:150]}...")
    
    # ========================================================================
    # 5. Final Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("[5/5] Training Complete!")
    print("=" * 60)
    
    # Final evaluation
    final_losses = estimate_loss(model, config, train_data, val_data)
    print(f"\nFinal Results:")
    print(f"  Train loss: {final_losses['train']:.4f}")
    print(f"  Val loss:   {final_losses['val']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Generate final sample
    print(f"\nFinal sample generation:")
    final_sample = generate_sample(
        model, tokenizer, config,
        prompt="Once upon a time",
        max_new_tokens=200,
    )
    print(final_sample)
    
    print(f"\nBest model saved to: {best_checkpoint_path}")
    print("\nNext steps:")
    print("  - Run inference: uv run python src/05_inference.py")
    print("  - Visualize:     uv run python src/06_visualize.py")


def main():
    """Main entry point for training."""
    config = get_config()
    
    # Optional: Override config for testing (uncomment to use)
    # config.max_iters = 500  # Quick test run
    # config.eval_interval = 100
    # config.warmup_iters = 50
    
    train(config)


if __name__ == "__main__":
    main()
