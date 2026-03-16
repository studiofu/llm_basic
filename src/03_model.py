"""Step 3: Transformer Model Architecture (Classical).

This implements the original 2017 Transformer architecture as used in GPT:
- Token + Positional Embeddings
- Multi-Head Self-Attention with causal masking
- Feed-Forward Network with ReLU
- Pre-norm LayerNorm with residual connections



  Architecture Components

   Module               Purpose
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Head                 Single attention head with Q, K, V projections + causal mask
   MultiHeadAttention   4 parallel heads (32-dim each) with output projection
   FeedForward          2-layer MLP: 128 → 512 → 128 with ReLU
   TransformerBlock     Pre-norm: x + attn(LN(x)) then x + ff(LN(x))
   GPTLanguageModel     Complete model with embeddings + 6 blocks + LM head

  Key Features

  • Causal masking: Triangular mask prevents attending to future tokens
  • Pre-norm architecture: LayerNorm before attention/FFN for stable training
  • Residual connections: Enable gradient flow in deep networks
  • Generation method: Temperature scaling + top-k filtering for sampling

  Model Stats

  Total parameters: 1,212,416
  ├── Token embeddings:     8,320
  ├── Position embeddings:  8,192
  ├── Transformer blocks: 1,187,328  (98% of params!)
  ├── Final LayerNorm:        256
  └── LM head:              8,320

  Test Results

  ✓ Forward pass: (4, 32) → (4, 32, 65) logits
  ✓ Loss computation: 4.21 (random init, expected ~4.17 for ln(65))
  ✓ Generation: Successfully generates 20 new tokens autoregressively



"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class Head(nn.Module):
    """Single head of self-attention.
    
    Each head computes scaled dot-product attention with causal masking.
    The head dimension is n_embd // n_head (e.g., 128 // 4 = 32).
    
    Attributes:
        key: Linear projection for keys (n_embd -> head_size)
        query: Linear projection for queries (n_embd -> head_size)
        value: Linear projection for values (n_embd -> head_size)
        register_buffer: Causal mask to prevent attending to future tokens
    """
    
    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        
        # Causal mask: triangular matrix that prevents looking at future tokens
        # Shape: (1, block_size, block_size) - broadcasts over batch dimension
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, config.block_size, config.block_size
            )
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = head_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute single-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, head_size)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Compute Q, K, V projections
        # k, q, v: (batch_size, seq_len, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Attention scores: (batch_size, seq_len, head_size) @ (batch_size, head_size, seq_len)
        # -> (batch_size, seq_len, seq_len)
        # Scale by sqrt(head_size) to prevent softmax saturation
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        
        # Apply causal mask: set future positions to -inf
        # Only use the mask up to current seq_len
        # softmax is 0 when the score is -inf
        # scores[b, i, j] = “how much position i attends to position j” (before softmax).
        # if j > i, then scores[b, i, j] = -inf as j is future position
        scores = scores.masked_fill(
            self.mask[:, :seq_len, :seq_len] == 0, float("-inf")
        )
        
        # Softmax to get attention weights (sum to 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, head_size)
        # -> (batch_size, seq_len, head_size)
        out = attn_weights @ v
        
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel.
    
    Splits n_embd across n_head heads, each computing attention independently.
    Outputs are concatenated and projected back to n_embd.
    
    Attributes:
        heads: ModuleList of Head modules
        proj: Linear projection to combine heads (n_embd -> n_embd)
        dropout: Dropout after projection
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        head_size = config.n_embd // config.n_head
        
        # Create n_head attention heads
        self.heads = nn.ModuleList([
            Head(config, head_size) for _ in range(config.n_head)
        ])
        
        # Output projection: concatenated heads -> n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Run each head in parallel and concatenate results
        # Each head outputs (batch_size, seq_len, head_size)
        # Concatenate to (batch_size, seq_len, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project back to n_embd
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with ReLU activation.
    
    Standard 2-layer MLP that expands n_embd -> 4*n_embd -> n_embd.
    This is where the model stores factual knowledge.
    
    Attributes:
        net: Sequential layers (Linear -> ReLU -> Linear -> Dropout)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            # Expand: n_embd -> 4 * n_embd
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            # Compress: 4 * n_embd -> n_embd
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.
    
    Architecture (Pre-Norm):
        x = x + MultiHeadAttention(LayerNorm(x))   # Communication
        x = x + FeedForward(LayerNorm(x))           # Computation
    
    Attributes:
        ln1: LayerNorm before attention
        ln2: LayerNorm before feed-forward
        attn: Multi-head attention module
        ff: Feed-forward network module
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Communication step: tokens exchange information via attention
        # Pre-norm: LayerNorm before the sub-layer, then residual connection
        x = x + self.attn(self.ln1(x))
        
        # Computation step: each token processes its own information
        x = x + self.ff(self.ln2(x))
        
        return x


class GPTLanguageModel(nn.Module):
    """Complete GPT-style language model.
    
    Architecture:
        1. Token Embedding: converts token IDs to vectors
        2. Position Embedding: adds position information
        3. Transformer Blocks: n_layer blocks of attention + FFN
        4. Final LayerNorm: stabilize before output
        5. LM Head: projects to vocabulary size for prediction
    
    Attributes:
        token_embedding_table: Embedding for tokens (vocab_size -> n_embd)
        position_embedding_table: Embedding for positions (block_size -> n_embd)
        blocks: Sequential transformer blocks
        ln_final: Final layer normalization
        lm_head: Linear projection to vocabulary (n_embd -> vocab_size)
        config: ModelConfig instance
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        

        # if vocab_size = 65, n_embd = 128, block_size = 64        
        # because need to convert the token ID to a vector of size n_embd
        # this table is trained during training
        # self.token_embedding_table:      (65, 128)

        # because need to convert the position to a vector of size n_embd
        # self.position_embedding_table:   (64, 128)
        # token_emb:       (batch_size, seq_len, 128)
        # where seq_len is the length of the input sequence
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embeddings: each position (0 to block_size-1) gets an n_embd vector
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(config.n_embd)

        # A Linear layer is a fully connected neural network layer, also known as a dense layer.
    
        # it converts the embedding vector of size n_embd to a vector of logits for each vocabulary token (vocab_size).
        # the last dimension is the probability of the next token in the vocabulary
        # the highest value in the vector in the last dimension is the most likely next token        
        # example, input [1, 64, 384] (after transformer blocks) , output [1, 64, 65] (logits for each vocabulary token)
        #         
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small random values for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the model.
        
        Args:
            idx: Input token IDs of shape (batch_size, seq_len)
            targets: Target token IDs for training, shape (batch_size, seq_len)
        
        Returns:
            logits: Prediction scores of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = idx.shape
        
        # Token embedding: each ID -> learned vector of size n_embd
        # (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        # embeeding is a kind of lookup table
        # here you pass (batch_size, seq_len) , it will return (batch_size, seq_len, n_embd)
        # input 64 batch, 10 chars input (64, 10) , output (64, 10, 128)
        token_emb = self.token_embedding_table(idx)
        
        # Position embedding: each position 0..seq_len-1 -> n_embd vector
        # pos_emb shape: (seq_len, n_embd)
        pos = torch.arange(seq_len, device=idx.device)
        # pos is a tensor of shape (seq_len,)\
        # if seq_len = 6, pos = [0, 1, 2, 3, 4, 5]

        # when you pass (seq_len,) , it will return (seq_len, n_embd)
        pos_emb = self.position_embedding_table(pos)

        # when add to token_emb, it will add to each batch 
        # it is so called broadcasting, no matter how many dimensions you have
        #         
        # Combine: token_emb + pos_emb. Broadcast adds pos_emb to each batch.
        # x shape: (batch_size, seq_len, n_embd) — same as token_emb
        x = token_emb + pos_emb
        
        # Pass through transformer blocks (input and output both batch_size x seq_len x n_embd)
        x = self.blocks(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary: (batch_size, seq_len, vocab_size)
        # logits are pre‑softmax scores, not probabilities.
        # if you want probabilities, you need to apply softmax to logits
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (batch*seq, vocab_size) vs (batch*seq,)
            batch_seq_vocab = logits.view(-1, self.config.vocab_size)
            batch_seq = targets.view(-1)
            loss = F.cross_entropy(batch_seq_vocab, batch_seq)
        
        return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.
        
        Args:
            idx: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated token IDs including input, shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to block_size if needed
                idx_cond = idx[:, -self.config.block_size:]
                
                # Get predictions
                logits, _ = self(idx_cond)
                
                # Focus on last token's logits
                logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply temperature scaling
                # temperature < 1, more focused, less random, as the value of logits become bigger, after softmax, the value of probs become bigger, so the model is more likely to choose the token with higher probability
                # temperature > 1, value become smaller, so the model is more likely to choose the token with lower probability
                logits = logits / temperature
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#
def create_model(config: ModelConfig, vocab_size: int) -> GPTLanguageModel:
    """Create and initialize a GPT model.
    
    Args:
        config: Model configuration
        vocab_size: Size of vocabulary (set from data)
    
    Returns:
        Initialized GPTLanguageModel
    """
    config.vocab_size = vocab_size
    model = GPTLanguageModel(config)
    return model


def main():
    """Test the model architecture."""
    print("=" * 60)
    print("STEP 3: TRANSFORMER MODEL - TESTING")
    print("=" * 60)
    
    from config import get_config
    
    config = get_config()
    
    # Set a small vocab size for testing
    test_vocab_size = 65
    config.vocab_size = test_vocab_size
    
    print(f"\n1. Configuration:")
    print(f"   vocab_size: {config.vocab_size}")
    print(f"   n_embd: {config.n_embd}")
    print(f"   n_head: {config.n_head}")
    print(f"   n_layer: {config.n_layer}")
    print(f"   block_size: {config.block_size}")
    print(f"   head_size: {config.n_embd // config.n_head}")
    
    print(f"\n2. Creating model...")
    model = create_model(config, test_vocab_size)
    model = model.to(config.device)
    print(f"   Device: {config.device}")
    
    print(f"\n3. Parameter count:")
    total_params = model.count_parameters()
    print(f"   Total parameters: {total_params:,}")
    
    # Break down by component
    token_emb_params = model.token_embedding_table.weight.numel()
    pos_emb_params = model.position_embedding_table.weight.numel()
    lm_head_params = model.lm_head.weight.numel()
    block_params = sum(p.numel() for p in model.blocks.parameters())
    ln_final_params = sum(p.numel() for p in model.ln_final.parameters())
    
    print(f"\n   Breakdown:")
    print(f"   - Token embeddings: {token_emb_params:,}")
    print(f"   - Position embeddings: {pos_emb_params:,}")
    print(f"   - Transformer blocks: {block_params:,}")
    print(f"   - Final LayerNorm: {ln_final_params:,}")
    print(f"   - LM head: {lm_head_params:,}")
    
    print(f"\n4. Testing forward pass...")
    batch_size = 4
    seq_len = 32
    
    # Create random input
    x = torch.randint(0, test_vocab_size, (batch_size, seq_len), device=config.device)
    y = torch.randint(0, test_vocab_size, (batch_size, seq_len), device=config.device)
    
    print(f"   Input shape: {x.shape}")
    logits, loss = model(x, y)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    print(f"\n5. Testing generation...")
    # Start with single token
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(start_idx, max_new_tokens=20, temperature=0.8)
    print(f"   Start: {start_idx.shape}")
    print(f"   Generated: {generated.shape}")
    print(f"   Generated tokens: {generated[0].tolist()}")
    
    print(f"\n6. Model components:")
    print(f"   {model}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
