# Simple LLM Demonstration Project

> A step-by-step guide to building a miniature Large Language Model from scratch — using the same architecture and techniques behind GPT, LLaMA, Gemma, and Mistral.

## Project Goal

Build a fully functional mini-LLM that can:
1. Learn from a story dataset using transformer architecture
2. Generate coherent text with multiple sampling strategies
3. Visualize what the model learns (attention maps, embeddings, loss curves)
4. Progressively upgrade from classical to modern techniques

**Philosophy**: Each step is a self-contained lesson. You will start with the simplest possible version, verify it works, then upgrade components one at a time — just like how the field evolved from the original Transformer (2017) to today's frontier models.

**Note**: This is for educational purposes — we won't compete with GPT/Claude, but we'll use the same fundamental techniques and understand *why* each design choice was made.

---

## Learning Milestones

| Milestone | What You'll Understand | Step |
|-----------|----------------------|------|
| **M1** — First Token | How text becomes numbers and back | Steps 1-2 |
| **M2** — Attention Click | How self-attention lets tokens "talk" to each other | Step 3a |
| **M3** — First Loss Drop | How backpropagation teaches the model | Step 4 |
| **M4** — Coherent Output | How generation works token-by-token | Step 5 |
| **M5** — Modern Upgrades | Why RoPE, RMSNorm, SwiGLU replaced the originals | Step 6 |
| **M6** — Visual Understanding | What attention heads actually learn | Step 7 |
| **M7** — Efficient Inference | How KV-cache makes generation fast | Step 8 |
| **M8** — Interactive Demo | Sharing your model with a web UI | Step 9 |

---

## Project Structure

```
llm_basic/
├── data/
│   └── stories.txt                 # Training dataset (stories)
├── src/
│   ├── config.py                   # Centralized hyperparameters & settings
│   ├── 01_data_preparation.py      # Step 1: Load, analyze, and split data
│   ├── 02_tokenizer.py             # Step 2: Character-level tokenizer (+ BPE upgrade)
│   ├── 03_model.py                 # Step 3: Transformer model (classical)
│   ├── 03_model_modern.py          # Step 3b: Modern architecture (RoPE, RMSNorm, SwiGLU)
│   ├── 04_train.py                 # Step 4: Training loop with logging
│   ├── 05_inference.py             # Step 5: Generation with sampling strategies
│   ├── 06_visualize.py             # Step 6: Attention maps, embeddings, loss curves
│   └── 07_app.py                   # Step 7: Interactive web UI (Gradio)
├── notebooks/
│   ├── 01_explore_data.ipynb       # Interactive data exploration
│   ├── 02_attention_deep_dive.ipynb# Visualize attention step-by-step
│   └── 03_experiments.ipynb        # Try hyperparameter changes
├── models/                         # Saved model checkpoints
├── outputs/                        # Generated samples & visualizations
├── pyproject.toml                  # Project metadata & dependencies (uv/PEP 621)
├── uv.lock                         # Locked dependency versions (auto-generated)
├── .python-version                 # Pinned Python version for uv
├── PLAN.md
└── README.md
```

---

## Step-by-Step Implementation Plan

### Step 0: Environment Setup
**Script**: `src/config.py` + `pyproject.toml`

**Package Manager**: [**uv**](https://docs.astral.sh/uv/) — the modern Python package & project manager from Astral (creators of Ruff). It replaces `pip`, `venv`, `pip-tools`, and `pyenv` in a single tool.

**Why uv over pip?**
- **10-100× faster** than pip for dependency resolution and installs
- **Deterministic lockfile** (`uv.lock`) — everyone gets the exact same versions
- **Manages Python versions** — no need for pyenv/conda/pyinstaller
- **Built-in virtual environments** — no manual `venv` creation
- **PEP 621 standard** — uses `pyproject.toml` instead of `requirements.txt`
- **Single binary, no dependencies** — install once, works everywhere

**Install uv** (one-time):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Project setup with uv**:
```bash
# Initialize project (creates pyproject.toml, .python-version, etc.)
uv init

# Pin Python version (uv will auto-install if needed)
uv python pin 3.12

# Add core dependencies
uv add torch numpy tqdm

# Add visualization dependencies
uv add matplotlib seaborn

# Add optional dependencies (when you reach those steps)
uv add tiktoken              # Step 6: BPE tokenizer
uv add gradio                # Step 9: Web UI
uv add tensorboard           # Optional: training dashboards
```

**`pyproject.toml`** (what uv generates and manages):
```toml
[project]
name = "llm-basic"
version = "0.1.0"
description = "Build a miniature LLM from scratch — educational project"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.2",
    "numpy",
    "tqdm",
    "matplotlib",
    "seaborn",
]

[project.optional-dependencies]
# Install with: uv sync --extra all
all = [
    "tiktoken",
    "gradio",
    "tensorboard",
]
```

**Running scripts with uv**:
```bash
# uv automatically uses the project's virtual environment — no activate needed
uv run python src/04_train.py
uv run python src/05_inference.py

# Or activate the venv manually if you prefer
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate
```

**Centralized Hyperparameters** (`src/config.py`):
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Data
    block_size: int = 64        # Maximum context length
    batch_size: int = 32        # Sequences per batch

    # Model architecture
    vocab_size: int = 0         # Set from data
    n_embd: int = 128           # Embedding dimension
    n_head: int = 4             # Number of attention heads
    n_layer: int = 6            # Number of transformer blocks
    dropout: float = 0.1        # Dropout rate

    # Training
    max_iters: int = 5000       # Training iterations
    eval_interval: int = 250    # Evaluation frequency
    eval_iters: int = 200       # Batches per evaluation
    learning_rate: float = 3e-4 # Peak learning rate
    min_lr: float = 3e-5        # Minimum learning rate (cosine decay)
    warmup_iters: int = 100     # Linear warmup steps
    weight_decay: float = 0.1   # AdamW weight decay

    # Modern upgrades (flags)
    use_rope: bool = False      # Rotary Position Embeddings
    use_rmsnorm: bool = False   # RMSNorm instead of LayerNorm
    use_swiglu: bool = False    # SwiGLU instead of ReLU
    use_kv_cache: bool = False  # KV-Cache for inference
    use_gqa: bool = False       # Grouped Query Attention

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9          # Nucleus sampling
    repetition_penalty: float = 1.1
    max_new_tokens: int = 200

    # System
    device: str = "auto"        # "auto", "cuda", "mps", "cpu"
    compile_model: bool = False # torch.compile (PyTorch 2.0+)
    seed: int = 42
```

**Why these values?**
- `n_embd=128` and `n_layer=6` gives ~500K-2M parameters — small enough to train on CPU in minutes, large enough to learn real patterns.
- `learning_rate=3e-4` with warmup + cosine decay is the standard recipe from GPT-2/3 papers.
- `dropout=0.1` prevents overfitting on our small dataset.
- Modern upgrade flags let you toggle techniques on/off and see the difference.

---

### Step 1: Data Preparation
**Script**: `src/01_data_preparation.py`

**What it does**:
- Load story dataset from `data/stories.txt`
- Analyze vocabulary (unique characters), print statistics
- Create mappings: `char -> index` and `index -> char`
- Split data into train/validation sets (90/10)
- Print sample encoded/decoded sequences for verification

**Input**: Raw text file with stories
**Output**: Processed data, vocab size, encode/decode functions

**Key Concepts to Learn**:
- **Tokenization** is the bridge between human text and model numbers
- **Character-level tokenization**: simplest approach — each unique character gets an integer ID
- **Vocabulary**: the complete set of tokens the model knows
- **Train/val split**: essential to detect overfitting

**Try It Yourself**:
- [ ] What happens if you use a tiny dataset (100 words)? A large one?
- [ ] Count how many unique characters your dataset has. Why does this matter for model size?

---

### Step 2: Tokenizer
**Script**: `src/02_tokenizer.py`

**What it does**:
- Implement `encode(text) -> list[int]`
- Implement `decode(tokens) -> str`
- Implement `get_batch()` for creating training batches
- Create input-target pairs for next-token prediction

**Key Concepts to Learn**:
- **Next-token prediction**: the fundamental task — given tokens [1,2,3], predict [2,3,4]
- **Batching**: process multiple sequences in parallel for GPU efficiency
- **Context window** (`block_size`): maximum number of tokens the model can "see"

```
Example — how input-target pairs work:

Text:     "The cat sat on"
Tokens:   [54, 12, 89, 3, 67, 41, 23, 3, 14, 41, 23, 3, 45, 12]

Input:    [54, 12, 89, 3, 67, 41, 23]    # "The cat"
Target:   [12, 89, 3, 67, 41, 23, 3]     # "he cat " (shifted by 1)

At EACH position, the model predicts the next token:
  Position 0: given [54]           -> predict 12  ("T" -> "h")
  Position 1: given [54, 12]       -> predict 89  ("Th" -> "e")
  Position 2: given [54, 12, 89]   -> predict 3   ("The" -> " ")
  ...and so on
```

**Try It Yourself**:
- [ ] Encode "Hello World" and decode it back. Is it lossless?
- [ ] What is the shape of a training batch? Why `(batch_size, block_size)`?

---

### Step 3: Transformer Model Architecture (Classical)
**Script**: `src/03_model.py`

**What it implements** — building blocks from smallest to largest:

#### 3.1 Token + Positional Embeddings

```
Input token IDs:    [54, 12, 89, 3]
                         ↓
Token Embedding:    each ID → learned vector of size n_embd
Position Embedding: each position (0,1,2,3) → learned vector of size n_embd
                         ↓
Combined:           token_emb + position_emb → input to transformer
```

**Why positional embeddings?** Attention is permutation-invariant — without position info, "cat sat on mat" and "mat sat on cat" look identical to the model.

#### 3.2 Single Head of Self-Attention

The core mechanism that makes transformers work:

```
For each token, compute three vectors:
  Q (Query):  "What am I looking for?"
  K (Key):    "What do I contain?"
  V (Value):  "What information do I provide?"

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

The √d_k scaling prevents dot products from growing too large
(which would push softmax into regions with vanishing gradients).
```

**Causal mask**: A triangular matrix that prevents tokens from attending to future positions. Without this, the model could "cheat" by looking ahead.

#### 3.3 Multi-Head Attention

```
Multi-head attention = Concat(head_1, head_2, ..., head_h) × W_o

Each head uses smaller dimensions (d_k = n_embd / n_head)
Different heads can learn different types of relationships:
  - Head 1 might learn syntactic patterns
  - Head 2 might learn semantic similarity
  - Head 3 might learn positional patterns
```

#### 3.4 Feed-Forward Network (FFN)

```
FFN(x) = Linear₂(ReLU(Linear₁(x)))

Inner dimension = 4 × n_embd (standard expansion ratio)

This is where the model stores "knowledge" — factual associations
learned from the training data. Attention routes information,
FFN processes it.
```

#### 3.5 Transformer Block (putting it together)

```
x = x + MultiHeadAttention(LayerNorm(x))     # Communication step
x = x + FeedForward(LayerNorm(x))            # Computation step
```

**Pre-norm** (LayerNorm before attention/FFN) is used instead of post-norm because it produces more stable gradients — this was a key discovery that enabled training deeper models.

**Residual connections** (`x + ...`) allow gradients to flow directly through the network, preventing the vanishing gradient problem in deep networks.

#### 3.6 Complete GPT-like Model

```
Input Token IDs
       ↓
Token Embedding + Positional Embedding
       ↓
┌─────────────────────────────┐
│     Transformer Block       │
│  ┌───────────────────────┐  │
│  │ LayerNorm → MultiHead │──┤ (residual)
│  │     Attention         │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │ LayerNorm → FFN       │──┤ (residual)
│  └───────────────────────┘  │
└─────────────────────────────┘
       ↓  × n_layer
Final LayerNorm
       ↓
Linear (n_embd → vocab_size)
       ↓
Logits → Softmax → Probability over vocabulary
       ↓
Sample next token
```

**Try It Yourself**:
- [ ] Count the model's total parameters. Where do most parameters live?
- [ ] What happens if you set `n_head=1`? How about `n_layer=1`?
- [ ] Remove the causal mask — can the model still learn? (Hint: yes, but it's useless for generation)

---

### Step 4: Training Loop
**Script**: `src/04_train.py`

**What it does**:
1. Initialize model with random weights
2. Set up AdamW optimizer with learning rate schedule
3. Training loop:
   - Sample a random batch of data
   - Forward pass: compute predictions and loss
   - Backward pass: compute gradients via backpropagation
   - Update weights with optimizer
   - Periodically evaluate on validation set
   - Generate sample text to monitor quality
   - Log metrics to TensorBoard (optional)
4. Save best model checkpoint

**Learning Rate Schedule** (modern standard):
```
Learning Rate
│
│    /\
│   /  \_____
│  /         \______
│ /                  \___
│/                       \
└──────────────────────────── Iterations
 warmup    cosine decay
```

Linear warmup for stability, then cosine decay to a minimum LR. This is the standard schedule used in GPT-2, GPT-3, LLaMA, and most modern LLMs.

**Key Concepts to Learn**:
- **Cross-entropy loss**: measures how far predictions are from true next tokens
- **Backpropagation**: chain rule applied through the entire network
- **AdamW**: Adam with decoupled weight decay — the default optimizer for transformers
- **Gradient clipping**: prevents exploding gradients (max_norm=1.0)
- **Overfitting detection**: when train loss drops but val loss rises

**Training Output**:
```
Iter    0 | Train loss: 4.174 | Val loss: 4.171 | LR: 0.000003
Iter  250 | Train loss: 2.483 | Val loss: 2.511 | LR: 0.000300
Iter  500 | Train loss: 1.972 | Val loss: 2.053 | LR: 0.000287
...
Sample @ iter 500: "Once upon a time there was a little girl who lived..."
```

**Try It Yourself**:
- [ ] Plot training vs validation loss. Where does overfitting start?
- [ ] Try `learning_rate=0.01` — what happens? Why?
- [ ] Try `learning_rate=0.00001` — what changes?
- [ ] Remove warmup — is training less stable?

---

### Step 5: Inference / Generation
**Script**: `src/05_inference.py`

**What it does**:
- Load trained model checkpoint
- Interactive command-line chat interface
- Multiple sampling strategies:

**Sampling Strategies** (from simplest to most sophisticated):

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| **Greedy** | Always pick highest-probability token | Deterministic but repetitive |
| **Temperature** | Scale logits by T before softmax | T<1 = focused, T>1 = creative |
| **Top-k** | Only consider top k tokens | Removes low-probability noise |
| **Top-p (nucleus)** | Only consider tokens summing to p probability mass | Adaptive — fewer options when model is confident |
| **Repetition penalty** | Reduce probability of recently generated tokens | Prevents loops |

**Generation Process**:
```
User Input: "Once upon a time"
    ↓
Encode → Token IDs: [45, 12, 89, 34, 56]
    ↓
Feed to model → Get logits for ALL vocab positions
    ↓
Apply temperature, top-k, top-p filtering
    ↓
Sample from resulting distribution → Token 78 ("a")
    ↓
Append to sequence: [45, 12, 89, 34, 56, 78]
    ↓
Repeat until max_new_tokens reached
    ↓
Decode → "Once upon a time a little girl named Rose..."
```

**Try It Yourself**:
- [ ] Generate with temperature=0.1 vs temperature=2.0 — describe the difference
- [ ] What happens with top_k=1? (same as greedy)
- [ ] Try top_p=0.5 vs top_p=0.95 — which gives better outputs?

---

### Step 6: Modern Architecture Upgrades
**Script**: `src/03_model_modern.py`

This is where we upgrade from the 2017 "Attention Is All You Need" architecture to techniques used in 2024-2026 era models (LLaMA 3, Gemma 2, Mistral, DeepSeek).

#### 6.1 RoPE — Rotary Position Embeddings
**Replaces**: Learned positional embeddings

```
Classical: position_emb = nn.Embedding(block_size, n_embd)  # Fixed max length
RoPE:      Encodes position by ROTATING Q and K vectors       # Generalizes to any length

Why better:
- Relative position encoding (distance matters, not absolute position)
- Can extrapolate to longer sequences than seen during training
- No additional parameters to learn
- Used in: LLaMA, Gemma, Mistral, Qwen, DeepSeek
```

#### 6.2 RMSNorm — Root Mean Square Normalization
**Replaces**: LayerNorm

```
LayerNorm:  y = (x - mean) / std * γ + β     # Centers AND scales
RMSNorm:    y = x / RMS(x) * γ                # Only scales

Why better:
- 10-15% faster (no mean computation or bias parameter)
- Empirically works just as well for transformers
- Used in: LLaMA, Gemma, Mistral, GPT-4 (rumored)
```

#### 6.3 SwiGLU Activation
**Replaces**: ReLU in the Feed-Forward Network

```
Classical FFN:  Linear → ReLU → Linear                   (2 weight matrices)
SwiGLU FFN:     Linear_gate → SiLU × Linear_up → Linear  (3 weight matrices)

SwiGLU(x) = SiLU(W_gate · x) ⊙ (W_up · x)

Why better:
- Smoother gradient flow than ReLU (no "dead neuron" problem)
- Gating mechanism lets the network learn which features to pass through
- ~1% better performance in practice (significant at scale)
- Used in: LLaMA, Gemma, PaLM, Mistral
```

#### 6.4 Grouped Query Attention (GQA)
**Replaces**: Standard Multi-Head Attention (MHA)

```
MHA:  Each head has its own Q, K, V           (n_head × 3 projections)
GQA:  Each head has own Q, but K/V are SHARED  (n_head Q, n_kv_head K/V)

Example with 8 heads, 2 KV groups:
  Q heads: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]  — 8 unique
  K heads: [K1, K1, K1, K1, K2, K2, K2, K2]  — 2 shared
  V heads: [V1, V1, V1, V1, V2, V2, V2, V2]  — 2 shared

Why better:
- Dramatically reduces KV-cache memory during inference
- Minimal quality loss compared to full MHA
- Enables longer context windows and larger batch sizes
- Used in: LLaMA 2+, Gemma, Mistral
```

#### 6.5 KV-Cache for Efficient Inference
**Optimization for**: Generation speed

```
Without cache (naive): For each new token, recompute attention over ALL previous tokens
With KV-cache:         Store K and V from previous steps, only compute for NEW token

Token 1: Compute K₁, V₁ → store
Token 2: Compute K₂, V₂ → store, attend using [K₁,K₂], [V₁,V₂]
Token 3: Compute K₃, V₃ → store, attend using [K₁,K₂,K₃], [V₁,V₂,V₃]

Speedup: O(n²) → O(n) per new token
This is why chatbots respond token-by-token quickly, not all-at-once slowly.
```

**Upgrade Path** — toggle in `config.py`:
```python
# Start classical (understand the basics)
config = ModelConfig(use_rope=False, use_rmsnorm=False, use_swiglu=False)

# Upgrade one-by-one and compare
config = ModelConfig(use_rope=True)                    # Just RoPE
config = ModelConfig(use_rope=True, use_rmsnorm=True)  # + RMSNorm
config = ModelConfig(use_rope=True, use_rmsnorm=True, use_swiglu=True)  # Full modern
```

**Try It Yourself**:
- [ ] Train classical vs modern with same config — compare final validation loss
- [ ] Measure training speed (iters/sec) with RMSNorm vs LayerNorm
- [ ] Generate long text with and without KV-cache — compare wall-clock time

---

### Step 7: Visualization & Understanding
**Script**: `src/06_visualize.py`
**Notebooks**: `notebooks/02_attention_deep_dive.ipynb`

**What it does**:
- **Loss curves**: Plot train/val loss over training iterations
- **Attention maps**: Heatmaps showing which tokens attend to which
- **Embedding space**: PCA/t-SNE of token embeddings (which characters cluster together?)
- **Head specialization**: Compare what different attention heads learn
- **Generation confidence**: Plot probability distribution over tokens during generation

**Attention Map Example**:
```
              T  h  e     c  a  t     s  a  t
         T  [■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·]
         h  [■  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·]
         e  [■  ■  ■  ·  ·  ·  ·  ·  ·  ·  ·]
            [■  ·  ·  ■  ·  ·  ·  ·  ·  ·  ·]
         c  [·  ·  ·  ·  ■  ·  ·  ·  ·  ·  ·]
         a  [·  ·  ·  ·  ■  ■  ·  ·  ·  ·  ·]
         t  [·  ·  ·  ·  ■  ■  ■  ·  ·  ·  ·]
            [■  ·  ·  ■  ·  ·  ·  ■  ·  ·  ·]
         s  [·  ·  ·  ·  ■  ·  ·  ·  ■  ·  ·]   ← "sat" attends to "cat"
         a  [·  ·  ·  ·  ·  ■  ·  ·  ■  ■  ·]
         t  [·  ·  ·  ·  ·  ·  ■  ·  ■  ■  ■]

■ = high attention, · = low attention
Causal mask visible: upper-right triangle is always zero
```

---

### Step 8: Interactive Web UI
**Script**: `src/07_app.py`

**What it does**:
- Gradio-powered chat interface
- Real-time token-by-token generation (streaming)
- Adjustable generation parameters via sliders
- Side panel showing attention visualization
- Model info display (parameter count, architecture, training stats)

```python
# Minimal Gradio app sketch
import gradio as gr

def chat(message, history, temperature, top_k, top_p):
    response = model.generate(
        prompt=message,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return response

demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.8, label="Temperature"),
        gr.Slider(1, 100, value=50, label="Top-k"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top-p"),
    ],
    title="Mini-LLM Chat",
    description="Chat with your from-scratch language model",
)
```

---

### Step 9 (Bonus): Advanced Concepts Overview

These are **not implemented** but explained conceptually, so learners understand the full LLM pipeline:

#### 9.1 BPE Tokenization with tiktoken
```
Character-level:  "playing" → ['p','l','a','y','i','n','g']  (7 tokens)
BPE:              "playing" → ['play', 'ing']                 (2 tokens)

Fewer tokens = longer effective context window = better performance.
tiktoken is OpenAI's fast BPE implementation used in GPT-3.5/4.
```

#### 9.2 LoRA — Low-Rank Adaptation
```
Full fine-tuning:  Update ALL parameters              (expensive)
LoRA:              Freeze base weights, add small      (cheap)
                   low-rank matrices ΔW = A × B

For a 128×128 weight matrix:
  Full:  128 × 128 = 16,384 trainable params
  LoRA:  128 × 4 + 4 × 128 = 1,024 trainable params (rank=4)
```

#### 9.3 Quantization (GGUF)
```
Full precision:  32 bits per parameter × 1M params = 4 MB
8-bit quantized: 8 bits per parameter  × 1M params = 1 MB
4-bit quantized: 4 bits per parameter  × 1M params = 0.5 MB

Trade small accuracy loss for dramatically smaller model size.
This is how llama.cpp runs 70B models on consumer hardware.
```

#### 9.4 RLHF / DPO (Alignment)
```
Pre-training:    Learn language from raw text (what we build)
SFT:             Fine-tune on instruction-following examples
RLHF:            Train reward model on human preferences,
                 then optimize policy with PPO
DPO:             Skip the reward model — directly optimize
                 on preference pairs (simpler, newer)

This is what turns a "text completer" into a "helpful assistant."
```

#### 9.5 torch.compile (PyTorch 2.0+)
```python
model = GPT(config)
model = torch.compile(model)  # One line for ~2× training speedup

Uses kernel fusion, operator rewriting, and graph optimization.
Free performance — just requires PyTorch 2.0+.
```

---

## The Story Dataset

**File**: `data/stories.txt`

**Content Guidelines**:
- 10,000+ words of simple stories (more = better, up to ~1MB)
- Fairy tales, short stories, or creative writing
- Consistent style and vocabulary helps the model learn patterns
- Can use public domain texts (Grimm's Fairy Tales, Aesop's Fables, etc.)

**Recommended Sources**:
- [Project Gutenberg](https://www.gutenberg.org/) — free public domain books
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) — synthetic stories specifically designed for small LLM training
- Write your own for a more personal learning experience

**Example Content**:
```
Once upon a time, in a land far away, there lived a young princess named Aurora. 
She loved to explore the enchanted forest near her castle. One day, while wandering 
through the woods, she discovered a hidden cave behind a waterfall...
```

---

## Expected Results

Given our small scale (tiny model, small dataset):

| Metric | Classical Architecture | Modern Architecture |
|--------|----------------------|-------------------|
| Parameters | ~500K - 2M | ~500K - 2M (same budget) |
| Training Time | 5-30 min (CPU) / 1-5 min (GPU) | Similar or slightly faster |
| Final Val Loss | ~1.8 - 2.5 | ~1.6 - 2.3 |
| Output Quality | Coherent phrases, basic grammar | Slightly more coherent, fewer repetitions |
| Context Handling | Fixed max length | Can generalize to longer (RoPE) |

**Sample Output** (after training):
```
Input:  "Once upon a time"
Output: "Once upon a time there was a little girl who lived in a small house 
         near the forest. She loved to walk in the woods and look for flowers 
         and berries. One sunny morning she decided to visit her grandmother..."
```

---

## Extension Ideas

After completing the core project, consider these challenges:

| Extension | Difficulty | What You'll Learn |
|-----------|-----------|-------------------|
| BPE tokenizer with tiktoken | Medium | Sub-word tokenization, vocabulary design |
| Larger dataset (Project Gutenberg) | Easy | Scaling laws, data quality effects |
| Multi-GPU training (DDP) | Medium | Distributed training fundamentals |
| LoRA fine-tuning | Medium | Parameter-efficient adaptation |
| Export to GGUF format | Hard | Quantization, cross-platform deployment |
| Add instruction-following (SFT) | Hard | Alignment, chat formatting |
| Mixture of Experts (MoE) | Hard | Sparse computation, routing |
| Speculative decoding | Hard | Inference optimization, draft models |
| Flash Attention | Hard | Memory-efficient attention, IO awareness |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Token** | The smallest unit of text the model processes (character or sub-word) |
| **Embedding** | A learned dense vector representation of a token |
| **Attention** | Mechanism that lets tokens "communicate" with each other |
| **Causal mask** | Prevents tokens from seeing future positions during generation |
| **Logits** | Raw model output scores before softmax normalization |
| **Softmax** | Converts logits to a probability distribution that sums to 1 |
| **Cross-entropy** | Loss function measuring prediction quality |
| **Backpropagation** | Algorithm to compute gradients via the chain rule |
| **Residual connection** | Skip connection that adds input directly to output |
| **LayerNorm / RMSNorm** | Normalization techniques to stabilize training |
| **FFN** | Feed-Forward Network — the per-token MLP in each block |
| **KV-Cache** | Stored key/value tensors to avoid recomputation during generation |
| **RoPE** | Rotary Position Embedding — encodes position via rotation |
| **SwiGLU** | Gated activation function used in modern transformer FFNs |
| **BPE** | Byte-Pair Encoding — sub-word tokenization algorithm |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning technique |
| **RLHF** | Reinforcement Learning from Human Feedback — alignment technique |
| **DPO** | Direct Preference Optimization — simpler alternative to RLHF |

---

## References

### Foundational Papers
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — The original Transformer
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020)

### Modern Architecture References
4. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE, 2021)
5. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU, 2020)
6. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (GQA, 2023)
7. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (LLaMA, 2023)

### Learning Resources
8. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide
9. [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy's video
10. [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT implementation
11. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP's line-by-line walkthrough
12. [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Visual intuition for deep learning

---

## Quick Start Commands

```bash
# 1. Install uv (one-time — skip if already installed)
#    Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    macOS/Linux:
#    curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set up project (creates venv, installs Python + dependencies)
uv sync

# 3. Prepare your dataset
# Place your stories in data/stories.txt

# 4. Train the model (classical architecture)
uv run python src/04_train.py

# 5. Train with modern upgrades
uv run python src/04_train.py --use-rope --use-rmsnorm --use-swiglu

# 6. Chat with your model
uv run python src/05_inference.py

# 7. Visualize what the model learned
uv run python src/06_visualize.py

# 8. Launch web UI
uv add gradio        # install once
uv run python src/07_app.py

# ---
# Useful uv commands:
# uv sync              — install/sync all dependencies from lockfile
# uv add <package>     — add a new dependency
# uv remove <package>  — remove a dependency
# uv run <command>     — run command in project's virtual environment
# uv lock              — regenerate lockfile without installing
# uv tree              — show dependency tree
```

---

*This project demonstrates that the principles behind GPT-4, LLaMA 3, Gemma 2, and Claude are accessible and understandable. You'll build the same architecture, use the same techniques, and gain intuition that transfers directly to understanding frontier AI systems.*
