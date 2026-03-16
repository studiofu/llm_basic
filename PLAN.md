# Simple LLM Demonstration Project

> A step-by-step guide to building a miniature Large Language Model from scratch — using the same architecture and techniques behind GPT, LLaMA, Gemma, and Mistral.

Author: Johnson Fu ([johnson.fu@gmail.com](mailto:johnson.fu@gmail.com)) co-edited by KIMI 2.5 and Claude Opus 4.6

## Project Goal

Build a fully functional mini-LLM that can:

1. Learn from a story dataset using transformer architecture
2. Generate coherent text with multiple sampling strategies
3. Visualize what the model learns (attention maps, embeddings, loss curves)
4. Progressively upgrade from classical to modern techniques

**Philosophy**: Each step is a self-contained lesson. You will start with the simplest possible version, verify it works, then upgrade components one at a time — just like how the field evolved from the original Transformer (2017) to today's frontier models.

**Note**: This is for educational purposes — we won't herecompete with GPT/Claude, but we'll use the same fundamental techniques and understand *why* each design choice was made.

---

## Learning Milestones


| Milestone                     | What You'll Understand                              | Step      |
| ----------------------------- | --------------------------------------------------- | --------- |
| **M1** — First Token          | How text becomes numbers and back                   | Steps 1-2 |
| **M2** — Attention Click      | How self-attention lets tokens "talk" to each other | Step 3a   |
| **M3** — First Loss Drop      | How backpropagation teaches the model               | Step 4    |
| **M4** — Coherent Output      | How generation works token-by-token                 | Step 5    |
| **M5** — Modern Upgrades      | Why RoPE, RMSNorm, SwiGLU replaced the originals    | Step 6    |
| **M6** — Visual Understanding | What attention heads actually learn                 | Step 7    |
| **M7** — Efficient Inference  | How KV-cache makes generation fast                  | Step 8    |
| **M8** — Interactive Demo     | Sharing your model with a web UI                    | Step 9    |


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

**Package Manager**: **[uv](https://docs.astral.sh/uv/)** — the modern Python package & project manager from Astral (creators of Ruff). It replaces `pip`, `venv`, `pip-tools`, and `pyenv` in a single tool.

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
uv python pin 3.13

# Add visualization dependencies
uv add matplotlib seaborn

# Add optional dependencies (when you reach those steps)
uv add tiktoken              # Step 6: BPE tokenizer
uv add gradio                # Step 9: Web UI
uv add tensorboard           # Optional: training dashboards
```

**PyTorch Installation (Choose One)**:

Option A: CPU-only (simpler, works everywhere)

```bash
# CPU-only PyTorch (no GPU required, slower training)
uv pip install torch>=2.6.0 numpy tqdm --index-url https://download.pytorch.org/whl/cpu
```

**When to use CPU-only**:

- You don't have an NVIDIA GPU
- You just want to explore the code without training
- Training speed is not a priority (e.g., small experiments)

Option B: GPU (CUDA) — Recommended for training

```bash
# GPU PyTorch with CUDA 12.4 (10-100× faster training)
uv pip install torch>=2.6.0 numpy tqdm --index-url https://download.pytorch.org/whl/cu124

# Verify GPU is available
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Prerequisites**:

- NVIDIA GPU with CUDA compute capability 3.5+
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) installed
- NVIDIA drivers updated

**When to use GPU**:

- You have an NVIDIA GPU and want fast training
- You're doing serious experiments or full training runs
- You want to train models with larger configs

`**pyproject.toml`** (what uv generates and manages):

```toml
[project]
name = "llm-basic"
version = "0.1.0"
description = "Build a miniature LLM from scratch — educational project"
readme = "README.md"
requires-python = ">=3.13,<3.14"
dependencies = [
    "matplotlib>=3.10.8",
    "numpy>=2.4.3",
    "seaborn>=0.13.2",
    "torch>=2.6.0",
    "tqdm>=4.67.3",
]

# For GPU support (CUDA 12.4), add this index configuration:
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

[tool.uv]
index-strategy = "unsafe-best-match"
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

#### What do all these settings mean? (Beginner-Friendly Guide)

Don't worry if the names above look intimidating. Every single one maps to a simple idea. Think of building an LLM like teaching a student to write stories — these settings control *how* that student learns.

---

**DATA SETTINGS — How we feed text to the model**

`**vocab_size`** — The dictionary size

> Imagine you give a student a dictionary. `vocab_size` is how many unique "words" (in our case, unique characters like `a`, `b`, `!`,  ``) are in that dictionary. If your training text uses 65 unique characters, the vocab_size is 65. The model can only read and write characters it has seen in this dictionary.

`**block_size = 64**` — The reading window (context length)

> This is how many characters the model can look at *at once* when predicting the next character. Think of it like reading through a small window — the model slides a window of 64 characters across the text and tries to predict what comes next.
>
> - `block_size = 64` → the model sees 64 characters of context at a time
> - GPT-4 has a block_size of ~128,000 tokens. Ours is tiny for learning purposes.
>
> ```
> "Once upon a time, in a land far away, there lived a young princess na"
>  ├──────────── 64 characters (the model sees this) ──────────────────┤
>                                                  predict next → "m"
> ```
>
> **This directly controls how far back attention can look.** When the model computes attention scores (described in `n_head` below), it can ONLY look at characters inside this window. Anything before the window is gone — the model has no memory of it.
>
> ```
> Full story: "...the king had a daughter. Once upon a time, in a land far away, there lived a young princess na"
>              ├── OUTSIDE the window ──┤├───────────── INSIDE the window (64 chars) ──────────────────┤
>              attention CANNOT see this  attention CAN score all of these characters
> ```
>
> This is a fundamental limitation. With `block_size = 64`, the model can never learn patterns that span more than 64 characters. That's why real-world LLMs use much larger context windows — ChatGPT/Claude can see tens of thousands of tokens at once, letting them reference things from pages ago in a conversation.

`**batch_size = 32**` — How many text windows to study at once

> This works together with `block_size`. Here's exactly what happens in one training step:
>
> 1. Pick 32 **random starting positions** in the training text
> 2. From each starting position, grab a window of 64 characters (block_size)
> 3. Feed all 32 windows into the model **at the same time**
>
> ```
> Our training text (thousands of characters):
> "Once upon a time, in a land far away, there lived a young princess named Aurora.
>  She loved to explore the enchanted forest near her castle. One day, while wandering
>  through the woods, she discovered a hidden cave behind a waterfall. Inside the cave
>  she found a magical book that could grant wishes..."
>
> One training step with batch_size=32, block_size=64:
>
> Window  1: "Once upon a time, in a land far away, there lived a young pr"  (64 chars)
> Window  2: "cess named Aurora. She loved to explore the enchanted forest"  (64 chars)
> Window  3: "e enchanted forest near her castle. One day, while wandering"  (64 chars)
> Window  4: "discovered a hidden cave behind a waterfall. Inside the cave"  (64 chars)
>   ...
> Window 32: "magical book that could grant wishes. She opened it carefully"  (64 chars)
>
> All 32 windows processed SIMULTANEOUSLY in one step.
>
> The data shape going into the model: (32, 64)
>   32 = batch_size (number of windows)
>   64 = block_size (characters per window)
> ```
>
> **Why not process one window at a time?**
>
> - **Speed**: GPUs are designed to do many calculations in parallel. Processing 32 windows takes almost the same time as processing 1.
> - **Stability**: Each single window gives a slightly different "lesson." Averaging the lessons from 32 windows gives a more reliable signal for how to improve. It's like asking 32 students what they think the answer is and taking the average, instead of relying on one student's possibly-wrong guess.
>
> **Tradeoffs**:
>
> - Bigger batch (64, 128) = smoother learning, but uses more memory
> - Smaller batch (8, 16) = noisier learning, but fits on smaller hardware
> - If you get an "out of memory" error, the first thing to try is reducing batch_size
>
> **What is the TARGET?**
>
> Yes — **the target is also a 64-character window**, just shifted by 1 position. Input and target are the same length (64); we just slide the target one step to the right.
>
> To form one (input, target) pair we need **65 consecutive characters** from the text: the first 64 are the input, and the "next 64" (starting at position 1) are the target.
>
> ```
> Training text: "Once upon a time, in a land far away, there lived a young princess..."
>
> Pick random start position i (so we have at least 65 chars: i through i+64):
>
> Position:   i                            i+63   i+64
> Input X:   |O|n|c|e| |u|p|o|n|...|a|w|a|y|      ← 64 chars: positions [i] to [i+63]
>              ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓     ↓ ↓ ↓ ↓
> Target Y:  |n|c|e| |u|p|o|n|...|a|w|a|y|?|     ← 64 chars: positions [i+1] to [i+64]
>
> X = text[i : i+64]       # 64 characters starting at i
> Y = text[i+1 : i+65]    # the NEXT 64 characters (shifted by 1)
>
> Same length (64). Target is just "input shifted right by one character."
> ```
>
> **Why 64 for both?** Because at EACH of the 64 positions in the input, we predict the NEXT character:
>
> ```
> Position 0:  see "O"            → predict "n"
> Position 1:  see "On"           → predict "c"
> Position 2:  see "Onc"          → predict "e"
> ...
> Position 63: see "...y"         → predict "?"
> ```
>
> So in one training step with `batch_size=32`:
>
> - We process **32 input windows** → shape `(32, 64)`
> - We compare against **32 target windows** → shape `(32, 64)`
> - The model makes **64 predictions per sequence** = **2,048 total predictions** in parallel!
>
> This is what makes language models efficient — they learn from every position in every sequence simultaneously.

---

**MODEL ARCHITECTURE — The shape of the model's "brain"**

`**n_embd = 128`** — Embedding dimension (how rich each character's representation is)

> Computers can't understand letters directly — they need numbers. An "embedding" converts each character into a list of numbers. `n_embd` is how long that list is.
>
> Think of it this way: if you had to describe the letter "A" to an alien using exactly 128 numbers, you could capture a lot of information — is it a vowel? uppercase? how often does it appear? what letters usually come after it? More numbers = richer description = the model can learn subtler patterns.
>
> ```
> Character "a" → [0.23, -0.87, 0.45, 1.12, ..., -0.33]  (128 numbers)
> Character "b" → [0.91, 0.14, -0.67, 0.05, ..., 0.82]   (128 numbers)
>
> Characters with similar roles (like vowels) will end up with
> similar number lists — the model discovers this on its own!
> ```
>
> - GPT-3 uses `n_embd = 12,288`. Ours is 128 — much smaller, but enough to learn.

`**n_layer = 6**` — Number of transformer layers (how deep the thinking goes)

> A "layer" is one round of processing. Each layer takes the output of the previous layer, processes it further, and passes the result to the next layer. Think of it as re-reading a sentence multiple times — each pass *could* notice something new.
>
> ```
> Input text
>    ↓
> [ Layer 1 ]
>    ↓
> [ Layer 2 ]     We don't tell the layers what to learn.
>    ↓             Each one figures out on its own what's
> [ Layer 3 ]     useful for predicting the next character.
>    ↓
> [ Layer 4 ]     The only goal: minimize prediction error.
>    ↓             Everything else emerges automatically.
> [ Layer 5 ]
>    ↓
> [ Layer 6 ]
>    ↓
> Prediction: next character
> ```
>
> **Important — what we know vs. what we don't:**
>
> We do **not** program what each layer learns. There's no line of code that says "Layer 1, learn grammar" or "Layer 5, learn story structure." All 6 layers have identical code. The only thing driving what they learn is the training signal — "you predicted the wrong next character, adjust yourself."
>
> However, researchers have used interpretability tools (probing classifiers, activation analysis) to peek inside trained models and found a **rough tendency**:
>
> - **Earlier layers** (closer to input) tend to pick up simpler, more local patterns
> - **Later layers** (closer to output) tend to pick up more abstract, longer-range patterns
>
> But this is a *tendency*, not a rule. In practice:
>
> - A single layer often does many things simultaneously
> - The same pattern might be partially handled by multiple layers
> - Different training runs with different random seeds can produce different internal organizations
> - We still don't fully understand why specific layers learn specific things
>
> This is an active area of research called **mechanistic interpretability** — trying to reverse-engineer what neural networks actually learn inside their layers. It's one of the biggest open questions in AI.
>
> For our purposes: more layers = the model has more rounds of processing to build up understanding, but we can't predict *exactly* what each layer will do. We just give it enough layers and let it figure things out.
>
> More layers = potentially deeper understanding, but slower training and more memory.
> GPT-3 has 96 layers. Ours has 6 — plenty for learning from short stories.

`**n_head = 4`** — Number of attention heads (different ways of paying attention)

> This one needs a longer explanation. Before understanding *multiple* heads, you need to understand what *one* head of attention does.
>
> ---
>
> **First: What is "attention"?**
>
> When the model is trying to predict the next character, it needs to decide: **which of the earlier characters are most relevant right now?**
>
> For example, to predict what comes after `"The cat sat on the ma"`:
>
> ```
> T  h  e     c  a  t     s  a  t     o  n     t  h  e     m  a  [?]
> ```
>
> Not all earlier characters are equally useful. The `m` and `a` are very relevant (we're in the middle of a word). The `cat` might be relevant (subject of the sentence). The `o` in `on` is less important here.
>
> Attention is a mechanism that computes a **relevance score** between every pair of characters. It produces something like this:
>
> ```
> To predict [?], how much should I look at each earlier character?
>
> "T" → 2%     (not very relevant)
> "h" → 1%
> "e" → 1%
> " " → 1%
> "c" → 3%     (start of "cat" — somewhat relevant)
> "a" → 5%     (part of "cat")
> "t" → 6%     (end of "cat" — the subject)
> " " → 1%
> "s" → 2%
> "a" → 3%
> "t" → 4%     (part of "sat" — the verb)
> " " → 1%
> "o" → 1%
> "n" → 1%
> " " → 1%
> "t" → 3%
> "h" → 2%
> "e" → 2%
> " " → 2%
> "m" → 25%    (very relevant! we're spelling a word)
> "a" → 33%    (most relevant! the character right before)
>              ────
>              100%  (all scores add up to 100%)
> ```
>
> The model then uses these scores to create a **weighted mix** of information from all earlier characters, paying most attention to the high-scoring ones. This mixed information helps it predict that `[?]` should be `t` (completing "mat").
>
> ---
>
> **The problem with ONE head: it can only make ONE set of scores**
>
> Look at the example above — that single attention pattern mostly focused on nearby characters (`m`, `a`). That's useful for completing the current word. But what if the model ALSO needs to:
>
> - Know what the subject was (`cat`) to understand the sentence
> - Notice that `the` appeared before, making `the mat` a likely phrase
> - Track that we're inside a short sentence, not a long one
>
> One set of scores can't do all of this at once. If it focuses heavily on `m` and `a` (nearby characters), it can't simultaneously focus heavily on `cat` (far away).
>
> ---
>
> **The solution: multiple heads — run several attention patterns in parallel**
>
> With `n_head = 4`, the model runs **4 independent attention calculations** at the same time, each producing its own set of scores:
>
> ```
> Input: "The cat sat on the ma[?]"
>
>                    Head 1          Head 2          Head 3          Head 4
>                  (scores)        (scores)        (scores)        (scores)
>
>  "T"  ─────────   2%              8%              1%              3%
>  "h"  ─────────   1%              3%              1%              2%
>  "e"  ─────────   1%              2%              1%              2%
>  " "  ─────────   1%              1%              5%              1%
>  "c"  ─────────   3%              7%              1%              2%
>  "a"  ─────────   5%             10%              1%              3%
>  "t"  ─────────   6%             15%              2%              4%
>  " "  ─────────   1%              1%              8%              1%
>  "s"  ─────────   2%              3%              2%              3%
>  "a"  ─────────   3%              5%              2%              4%
>  "t"  ─────────   4%             10%              3%              5%
>  " "  ─────────   1%              1%              9%              1%
>  "o"  ─────────   1%              2%              3%              2%
>  "n"  ─────────   1%              2%              3%              2%
>  " "  ─────────   1%              1%             10%              1%
>  "t"  ─────────   3%              5%              5%             10%
>  "h"  ─────────   2%              3%              5%             12%
>  "e"  ─────────   2%              3%              5%             13%
>  " "  ─────────   2%              1%             11%              3%
>  "m"  ─────────  25%              5%              8%              8%
>  "a"  ─────────  33%             12%             14%             18%
>
>  Focuses on:     nearby chars    "cat"+"sat"     spaces/structure "the"+"the"
>                  (spelling)      (meaning)       (word boundaries) (repetition)
> ```
>
> Each head produces 128 / 4 = **32 numbers** of output (they split the embedding dimension between them). The 4 outputs are then **concatenated** (glued together) back into 128 numbers and mixed with a final multiplication.
>
> ```
> Head 1 output: [32 numbers]  ─┐
> Head 2 output: [32 numbers]  ─┤── concatenate ──→ [128 numbers] ──→ final mix ──→ [128 numbers]
> Head 3 output: [32 numbers]  ─┤
> Head 4 output: [32 numbers]  ─┘
> ```
>
> ---
>
> **Important caveats (same as with layers):**
>
> The labels I used above ("spelling", "meaning", "word boundaries", "repetition") are **made up for illustration**. In reality:
>
> - We don't assign roles to heads — they learn on their own
> - The actual attention scores would look messier than my clean example
> - Some heads learn interpretable patterns, others don't
> - Researchers have found that some heads in large models can be removed entirely with little effect
>
> **The math behind it** (optional — you'll see this in Step 3):
> Each head has its own small set of learnable weights (Q, K, V matrices). These weights start random and gradually adjust during training until the head produces attention patterns that help predict the next character. That's all there is to it — the "magic" is just matrix multiplications + training.
>
> **How far back can attention look?**
> Exactly `block_size` characters — in our case, 64. Every character in the window can attend to every earlier character in the same window, but nothing outside the window. This is why `block_size` is sometimes called the "context length" — it's the maximum range of attention.
>
> **So how do scores actually help predict the next character?**
>
> The scores tell the model how much to "listen to" each earlier character when deciding what comes next. Characters with high scores contribute more to the prediction, characters with low scores are mostly ignored.
>
> It's not picking just one character — it's a **weighted mix** of information from ALL characters, blended by the scores:
>
> ```
> Scores:       "m" = 25%,  "a" = 33%,  "t" (from "cat") = 6%,  "T" = 2%  ...
>
> What the model "hears":
>   Loud and clear:  "m" and "a"         → "we're spelling 'ma_' — probably 'mat' or 'map'"
>   Faintly:         "cat", "sat"        → "about a cat sitting, so 'mat' makes sense"
>   Barely:          "T", "h", "e"       → almost ignored for this prediction
>                                            ↓
>                                      predict: "t"  (completing "mat")
> ```
>
> The model takes a **weighted average** of information from all characters — 33% from `"a"`, 25% from `"m"`, 6% from `"t"`, 2% from `"T"`, etc. — and that blended signal feeds into the rest of the network to produce the final prediction. This is why the mechanism is called "attention" — it literally pays more attention to the relevant parts and less to the irrelevant parts.
>
> **Why 4 heads and not 1 or 100?**
>
> - `n_head = 1` → only one attention pattern per layer (limited)
> - `n_head = 4` → four parallel patterns (good balance for our small model)
> - `n_head = 128` → would mean each head only gets 128/128 = 1 number to work with (too small to be useful)
> - Rule of thumb: `n_embd / n_head` should be at least 16-32 for each head to have enough capacity

`**dropout = 0.1`** — Random forgetting (prevents memorizing)

> During training, the model randomly "turns off" 10% of its neurons at each step. This sounds destructive — why would disabling parts of the model help? Let's look at a concrete example.
>
> **The problem dropout solves: overfitting (memorizing instead of learning)**
>
> Imagine our training data contains these sentences:
>
> ```
> "The cat sat on the mat."
> "The dog sat on the rug."
> "The bird sat on the branch."
> ```
>
> What we WANT the model to learn:
>
> - After "The [animal] sat on the", predict a reasonable surface/place
> - General pattern: [animal] + [sat on] + [surface]
>
> What can go WRONG without dropout:
>
> - The model has 1.2 million parameters but maybe only 10,000 words of training data
> - It has more than enough capacity to memorize every single sentence exactly
> - Instead of learning "animals sit on surfaces", it memorizes "character #4,521 in the training data is 'm'"
> - Result: it perfectly predicts training data but generates garbage on new text
>
> ```
> Without dropout (overfitting):
>   Training data:  "The cat sat on the mat" → predicts perfectly ✓
>   New input:      "The fox sat on the ___" → predicts nonsense ✗
>   (it never saw "fox" in training, so it's lost)
>
> With dropout (generalization):
>   Training data:  "The cat sat on the mat" → predicts well (not perfect) ✓
>   New input:      "The fox sat on the ___" → predicts "log" or "rock" ✓
>   (it learned the general pattern, not exact memorization)
> ```
>
> **How dropout forces generalization:**
>
> Each training step, 10% of neurons are randomly disabled (set to zero):
>
> ```
> Step 1:  neurons active: [✓][✗][✓][✓][✗][✓][✓][✓][✗][✓]  (random 10% off)
> Step 2:  neurons active: [✓][✓][✗][✓][✓][✓][✗][✓][✓][✗]  (different 10% off)
> Step 3:  neurons active: [✗][✓][✓][✓][✓][✗][✓][✗][✓][✓]  (different 10% off)
> ```
>
> Because a different random set is disabled each time, the model can never rely on any specific neuron always being there. It's forced to spread knowledge across many neurons, building redundant pathways. This redundancy is what makes it generalize — if it stored "cat → mat" in one specific neuron, that neuron might be off next time, so the model learns the broader pattern instead.
>
> Think of it like a sports team where the coach randomly benches different players each practice. No single player can carry the team alone — everyone has to learn to play well, and the team becomes more resilient.
>
> **During generation (inference), dropout is turned OFF** — all neurons are active, giving the model its full power.
>
> - `dropout = 0.0` → no forgetting (risk of memorizing the training data exactly)
> - `dropout = 0.1` → forget 10% randomly each step (good for small datasets like ours)
> - `dropout = 0.3` → forget 30% (more aggressive, for very small datasets)
> - `dropout = 0.5` → forget 50% (very aggressive, rarely used in modern LLMs)

---

**TRAINING SETTINGS — How the model studies**

`**max_iters = 5000`** — Total training steps

> One "iteration" = the model looks at one batch of data, makes predictions, checks how wrong it was, and adjusts itself. 5000 iterations means 5000 rounds of practice.
>
> Think of it as doing 5000 practice tests. Early iterations = wild guessing. Later iterations = the model has learned real patterns.

`**learning_rate = 3e-4`** — Step size (how much to adjust each time)

> After each practice round, the model needs to adjust its internal settings. The learning rate controls *how big* each adjustment is.
>
> `3e-4` means `0.0003` — a very small number. This is deliberate:
>
> - **Too big** (like 0.01): The model overcorrects wildly, like a student who panics after every mistake and rewrites everything
> - **Too small** (like 0.000001): The model barely changes, like a student who ignores feedback
> - **Just right** (0.0003): Small, steady improvements each round
>
> ```
> learning_rate = 0.01    →  Too big  — model explodes, loss goes to infinity
> learning_rate = 0.0003  →  Sweet spot — steady improvement
> learning_rate = 0.000001 → Too small — model barely learns, wastes time
> ```

`**warmup_iters = 100**` — Gentle start

> For the first 100 iterations, the learning rate starts at nearly zero and gradually increases to `3e-4`. This is like warming up before exercise — jumping straight into a high learning rate when the model's weights are still random can cause instability.

`**min_lr = 3e-5**` — Slow down near the end

> As training progresses, the learning rate gradually decreases from `3e-4` down to `3e-5` (10× smaller). This is like taking smaller steps as you get closer to the answer — big adjustments early on, fine-tuning later.

`**weight_decay = 0.1**` — Prevent laziness

> This gently pushes the model's internal numbers toward zero, preventing them from growing unnecessarily large. It's a form of regularization — keeping the model "lean" so it learns general patterns instead of overly specific ones.

`**eval_interval = 250**` — How often to check progress

> Every 250 training steps, we pause and test the model on text it has *never seen* (the validation set). This tells us if the model is actually learning vs. just memorizing.

`**eval_iters = 200`** — How thorough each check is

> When we do check progress, we test on 200 different batches and average the results. More batches = more reliable measurement of how good the model is.

---

**GENERATION SETTINGS — How the model writes text**

`**temperature = 0.8`** — Creativity dial

> Controls how "random" the model's word choices are when generating text.
>
> Think of it as a confidence dial:
>
> - `temperature = 0.1` → Very confident, always picks the most likely next character. Output is repetitive but safe: *"the the the the"*
> - `temperature = 0.8` → Mostly confident but sometimes surprises you. Good balance.
> - `temperature = 2.0` → Very random, often picks unlikely characters. Output is creative but often nonsensical: *"th$e xqat zza"*
>
> ```
> Model thinks next character probabilities are:
>   "e" = 60%, "a" = 25%, "o" = 10%, "x" = 5%
>
> temperature = 0.1  →  "e" almost every time (focused)
> temperature = 0.8  →  usually "e", sometimes "a" (balanced)
> temperature = 2.0  →  could be anything, even "x" (chaotic)
> ```

`**top_k = 50**` — Only consider the best options

> Before picking the next character, throw away everything except the top 50 most likely candidates. This removes obviously bad choices (like putting "Z" after "th").

`**top_p = 0.9**` — Smart cutoff (nucleus sampling)

> Instead of a fixed number of candidates (top_k), keep adding candidates from most-likely to least-likely until their combined probability reaches 90%. When the model is very confident, this might mean only 2-3 options. When it's uncertain, it might keep 40+. It adapts automatically.

`**repetition_penalty = 1.1**` — Don't repeat yourself

> Reduces the probability of tokens that already appeared recently in the generated text. Without this, models often get stuck in loops: *"the cat the cat the cat the cat..."*
>
> **Why does repetition happen in the first place?**
>
> The model was trained with a single objective: *predict the next token*. It was never explicitly told "don't say the same thing again." So several things push it toward repeating:
>
> 1. **Frequent tokens stay likely** — In normal text, common words like "the", "a", "and" appear over and over. The model learned that after many contexts, "the" is a safe guess. So when it has already generated "the", the next-token distribution might still put high probability on "the" again, because in training data "the the" sometimes appears (e.g. "the ... the") and very common words get high probability in general.
> 2. **Short loops are consistent** — Once the model outputs "the cat ", the context looks a lot like "the cat " again. So it may assign high probability to "the" again (starting "the cat the cat..."). The prediction is locally consistent — it's a plausible "next token" — but globally it's a boring loop.
> 3. **No built-in "memory" of what was just said** — The model only sees the sequence so far. It has no separate signal that says "you just said this, try something different." So if "cat" is a likely next token in many contexts, it can keep picking "cat" again and again.
> 4. **Small models repeat more** — Our tiny model has limited capacity. When it finds a pattern that works a bit ("the cat" is valid English), it may overuse it instead of exploring other options.
>
> ```
> Without penalty:
>   Generated: "Once upon a time the cat sat on the mat. The cat was happy. The cat..."
>   The model keeps choosing "the" and "cat" because they're still high-probability next tokens.
>
> With repetition_penalty = 1.1:
>   We divide the probability of any token that already appeared by 1.1 (or apply a stronger penalty the more often it appeared).
>   So "the" and "cat" get down-weighted → the model is nudged to pick different words.
>   Generated: "Once upon a time the cat sat on the mat. It was happy. It purred..."
> ```
>
> **How it works (conceptually):** Before sampling, we look at the tokens we've already generated. For each candidate next token, if it appeared recently, we reduce its probability (e.g. divide by 1.1, or by a factor that grows with how many times it appeared). So repeated tokens become less likely and the model is encouraged to say something new.

`**max_new_tokens = 200`** — When to stop writing

> The maximum number of characters the model will generate per response. Without a limit, it would write forever.

---

**SYSTEM SETTINGS — Hardware and reproducibility**

`**device = "auto"`** — Where to run the computation

> - `"cuda"` → NVIDIA GPU (fastest, 10-100× faster than CPU)
> - `"mps"` → Apple Silicon GPU (Mac M1/M2/M3)
> - `"cpu"` → Regular processor (slowest, but always available)
> - `"auto"` → Automatically picks the best available option

`**compile_model = False`** — Speed optimization

> When set to `True`, PyTorch analyzes the entire model and optimizes it as one unit, making training ~2× faster. We keep it off by default because it requires PyTorch 2.0+ and adds startup time. Turn it on once you've verified everything works.

`**seed = 42`** — Reproducibility

> Neural networks use random numbers (for initialization, dropout, batching). Setting a seed means the "random" numbers are the same every time you run the code. This way, if you and a friend both run the same code, you'll get identical results. (42 is a tradition in programming — a reference to *The Hitchhiker's Guide to the Galaxy*.)

---

**MODERN UPGRADE FLAGS — Toggle new techniques on/off**

These are all `False` by default. You'll start with the classical 2017 architecture, then flip these on one-by-one in Step 6 to see how modern LLMs improved on the original design.


| Flag           | What it changes                                 | When to enable |
| -------------- | ----------------------------------------------- | -------------- |
| `use_rope`     | How the model knows word position in a sentence | Step 6.1       |
| `use_rmsnorm`  | How the model normalizes numbers between layers | Step 6.2       |
| `use_swiglu`   | How neurons "activate" inside each layer        | Step 6.3       |
| `use_gqa`      | How attention heads share work                  | Step 6.4       |
| `use_kv_cache` | How fast the model generates text               | Step 6.5       |


Don't worry about understanding these yet — each one gets a full explanation in Step 6.

---

**How these settings relate to model size (number of parameters)**:

A "parameter" is one adjustable number inside the model. More parameters = the model can memorize more patterns, but also needs more data and time to train. Here's how our settings translate:

```
Main parameter costs:
  Token embeddings:      vocab_size × n_embd       =  65 × 128     =     8,320
  Position embeddings:   block_size × n_embd       =  64 × 128     =     8,192
  Per transformer layer:
    Attention (Q,K,V,O): 4 × n_embd × n_embd      =  4 × 128×128  =    65,536
    Feed-forward:        2 × n_embd × (4 × n_embd) = 2 × 128×512  =   131,072
    Layer norms:         2 × n_embd                =  2 × 128      =       256
  × n_layer:            6 layers                    =  6 × 196,864  = 1,181,184
  Final linear:          n_embd × vocab_size        =  128 × 65     =     8,320
                                                                    ___________
  Total:                                                            ≈ 1,206,016

That's about 1.2 million parameters — tiny by industry standards
(GPT-3 has 175 BILLION), but perfect for learning.
```

**What are "Attention," "Feed-forward," and "Layer norms"?**

These are the three main building blocks inside each transformer layer. Here’s what each one is and why it has that many parameters.

---

**1. Attention layer (Q, K, V, O)** -- "Who should I look at?"

**Terms for the vector at each position**

- **Initial / static embedding** — Fixed lookup by token (or character). Same token gives the same vector every time; no context.
- **Hidden state** — The vector at a position after one or more layers (e.g. "hidden state at layer 2, position 5"). Standard in RNN/Transformer.
- **Representation** or **contextual representation** — Same idea: the current vector at that position after processing. "Contextual" means it depends on surrounding tokens, not just the token itself.

So: **static embedding** = fixed per token; **hidden state** (or **representation**) = current vector at that position after processing.

**Attention in one formula block**

Input: at each position we have a **hidden state** (or representation) of dimension `n_embd` (e.g. 128). We multiply it by three learned matrices to get Q, K, V:

- **Q** = hidden_state × W_Q   (Query)
- **K** = hidden_state × W_K   (Key)
- **V** = hidden_state × W_V   (Value)

Then, for each position whose output we want (e.g. the last position), we:

1. **Scores:** For each position j, compute `score_j = Q · K_j` (dot product). In practice we often scale by √d before the next step.
2. **Weights:** Apply softmax to the scores so they sum to 1: `weights = softmax(scores)`.
3. **Output:** Return the weighted sum of the V vectors, not a single V:
  `output = weight_0×V_0 + weight_1×V_1 + … + weight_{63}×V_63`

So we **use** all V vectors and **return** that weighted combination. In short: **scores = Q·K → softmax → weights → output = Σ (weight_j × V_j)**.

This is the part that computes the attention scores and the weighted mix (the mechanism we already discussed in the `n_head` section). But how does the model actually *compute* those percentage scores? That's where Q, K, V come in.

Remember our example: `"The cat sat on the ma[?]"` -- the model needed to score every earlier character by relevance. But each character starts as just a 128-number embedding vector. The model needs a way to compare them. It does this by transforming each character's vector into **three different versions**, each with a specific purpose.

**Think of it like a library:**

Imagine each character is a person standing in a library. To figure out who to talk to:

- Each person writes a **question** on a card: "I'm looking for ___" -- this is **Q (Query)**
- Each person wears a **name tag** describing what they know: "I contain ___" -- this is **K (Key)**
- Each person holds a **notebook** with useful info to share: "Here's my info: ___" -- this is **V (Value)**

To compute attention scores, you **match questions (Q) against name tags (K)**. If your question matches someone's name tag well, that person gets a high score. Then you **read the notebooks (V)** of the high-scoring people.

**What does "match" mean, and how do we "read the notebooks"? (More detail)**

- **Where does the Query Q come from? (Why do we have a Q for "[?]"?)**
We don't have an embedding for the next character "[?]" — it doesn't exist yet. What we have is **64 positions** (0 to 63), each with a character and an embedding: position 0 = "T", position 1 = "h", ..., position 63 = "a" (the last character in "The cat sat on the ma"). For **each** of these positions we compute a Q, K, and V from that position's embedding. So the **Query we use to predict what comes after "a"** is **the Q at the last position** — position 63 — which comes from the embedding of "a". In other words: the last character's embedding is turned into a Query that asks "which of the previous positions (including me) have the right information to help predict the next character?" So when we say "for the position we're predicting" we really mean "for the **last** position in the window" (the one right before the next token). That last position has a character ("a"), an embedding, and therefore a Q. We use that Q to look at all the Keys and blend the Values; the result at that last position is what we use to predict the next token.
- **Matching Q and K**  
So we have one **Query** vector Q — it's the Q of the **last position** (e.g. from "a"). For every position (including that one) we have a **Key** vector K. "Match" means we measure how similar Q is to each K. The math used is the **dot product**: for each position j we compute  
`score_j = Q · K_j`  
(sum the products of the 128 numbers in Q and the 128 numbers in K_j). If Q and K_j point in similar directions, the dot product is large (high score); if they're very different, it's small or negative (low score). So we get one raw number per position — "how relevant is this position to me?"
- **Turning scores into percentages**  
Those raw scores can be any size and don't sum to 1. We pass them through **softmax**:  
  - exponentiate each score,  
  - then divide each by the sum of all of them.  
  After softmax, we have one number per position that is between 0 and 1 and **all of them add up to 100%**. Those are the "attention weights" (the 33%, 25%, 6%, 2%, ... from our example).
- **Reading the notebooks (V) — the math step by step**  
Each position has a **Value** vector V — a list of 128 numbers (the "notebook"). We also have one **weight** (percentage) per position from the previous step, e.g. 0.33 for "a", 0.25 for "m", 0.06 for "t", ..., and they sum to 1.0.
**"Blend" means: weighted sum.** We form one new vector of 128 numbers. For **each of the 128 slots** we do:
  - Take that slot's value from position 0, multiply by weight 0; add  
  - that slot's value from position 1, multiply by weight 1; add  
  - … same for all positions.
  So for slot (dimension) `d`:
  `result[d] = w_0 * V_0[d] + w_1 * V_1[d] + w_2 * V_2[d] + ... + w_63 * V_63[d]`
  We do this for `d = 0, 1, 2, ..., 127`. So we get one number per dimension, i.e. one 128-d vector. That vector is the "blended notebook" — mostly from positions with high weight (e.g. "a" and "m") but with a small contribution from every position. That blended vector is what the model uses for the next step.
  **Minimal numerical example (3 positions, 4 dimensions):**
  Suppose we have only 3 positions and V has 4 numbers (instead of 64 and 128). Weights from softmax: position 0 = 0.5, position 1 = 0.3, position 2 = 0.2 (sum = 1.0). Value vectors:
  ```
  V_0 = [1.0,  0.0,  0.5, -0.2]   (position 0, weight 0.5)
  V_1 = [0.2,  0.8,  0.1,  0.3]   (position 1, weight 0.3)
  V_2 = [0.0,  0.1,  0.4,  0.6]   (position 2, weight 0.2)
  ```
  Blend (weighted sum) for each dimension:
  ```
  result[0] = 0.5×1.0 + 0.3×0.2 + 0.2×0.0  = 0.50 + 0.06 + 0.00  = 0.56
  result[1] = 0.5×0.0 + 0.3×0.8 + 0.2×0.1  = 0.00 + 0.24 + 0.02  = 0.26
  result[2] = 0.5×0.5 + 0.3×0.1 + 0.2×0.4  = 0.25 + 0.03 + 0.08  = 0.36
  result[3] = 0.5×(-0.2) + 0.3×0.3 + 0.2×0.6 = -0.10 + 0.09 + 0.12 = 0.11
  ```
  So the blended vector is `result = [0.56, 0.26, 0.36, 0.11]`. Position 0 had the biggest weight (0.5), so its numbers (1.0, 0.0, 0.5, -0.2) dominate the result; positions 1 and 2 add smaller contributions. In the real model we do the same thing with 64 positions and 128 dimensions: one weighted sum per dimension.

**Tiny numerical example (idea only):**

```
Last position ("a") has  Q = [0.5, 0.1, -0.3, ...]   (Q comes from "a"'s embedding)
Q = E @ W_Q.T

Position "a" has  K_a = [0.6, 0.0, -0.2, ...]   →  Q·K_a = 0.36  (high — similar direction)
Position "m" has  K_m = [0.4, 0.2, -0.1, ...]   →  Q·K_m = 0.28
Position "T" has  K_T = [-0.2, 0.8, 0.1, ...]   →  Q·K_T = -0.02 (low — different direction)

Raw scores:  [0.36, 0.28, ..., -0.02, ...]
      ↓  softmax (make positive, sum to 1)
Weights:     [0.33, 0.25, ..., 0.02, ...]   (33%, 25%, ..., 2%, ...)

Result = 0.33×V_a + 0.25×V_m + ... + 0.02×V_T + ...
       = one 128-d vector, heavy on "a" and "m"
```

So: **Q and K decide who gets weight (scores → percentages); V is what we actually blend (the content).**

**Concretely, here's what happens for position [?] in `"The cat sat on the ma[?]"`:**

```
Step 1: Transform each character's embedding into Q, K, V vectors
        using three separate learned weight matrices (each 128 x 128):

        Character "a" (the last one before [?]):
          embedding: [0.23, -0.87, 0.45, ...]   (128 numbers)
                          | multiply by W_Q matrix
          Q_a =          [0.51, 0.12, -0.33, ...]  (128 numbers) -- its "question"
                          | multiply by W_K matrix
          K_a =          [-0.22, 0.88, 0.07, ...]  (128 numbers) -- its "name tag"
                          | multiply by W_V matrix
          V_a =          [0.67, -0.41, 0.55, ...]  (128 numbers) -- its "notebook"

        (Same transformation for every other character: "T", "h", "e", "c", "t", ...)

Step 2: Compute scores by comparing Q of [?] with K of every earlier character.
        High match between Q and K = high score.

        score("a", [?]) = Q_[?] . K_a = high   (33%)
        score("m", [?]) = Q_[?] . K_m = high   (25%)
        score("t", [?]) = Q_[?] . K_t = medium (6%)
        score("T", [?]) = Q_[?] . K_T = low    (2%)
        ...
        (These are the percentage scores from our earlier example!)

Step 3: Use scores to take a weighted average of V vectors.

        result = 33% x V_a  +  25% x V_m  +  6% x V_t  +  2% x V_T  + ...
                                    |
                 A single 128-number vector that blends information
                 mostly from "a" and "m" (the high-scoring characters)
```

**Yes — scores are only over the block.** When we say "compare Q of [?] with K of every earlier character," that "every" means **every position inside the current context window**, i.e. exactly **block_size** positions (64 in our config). There are no Q, K, or V vectors for anything outside that window; the model never sees or scores positions beyond block_size. So for each of the 64 positions, we compute up to 64 attention scores (with causal masking: position 0 scores 1 key, position 1 scores 2 keys, ..., position 63 scores 64 keys). The attention layer never looks beyond block_size.

**So the three matrices do three different jobs:**


| Matrix | Name  | Size      | What it produces                 | Purpose                  |
| ------ | ----- | --------- | -------------------------------- | ------------------------ |
| W_Q    | Query | 128 x 128 | A "question" vector per position | "What am I looking for?" |
| W_K    | Key   | 128 x 128 | A "name tag" vector per position | "What do I contain?"     |
| W_V    | Value | 128 x 128 | A "notebook" vector per position | "What info do I share?"  |


The **Q** and **K** vectors are only used to compute the scores (they get compared, then thrown away). The **V** vectors carry the actual information that gets mixed by those scores.

**What about O?**

**W_O (Output)** is a fourth matrix (also 128 x 128). Remember that with `n_head = 4`, each head produces a small 32-number result. **O** takes the 4 concatenated head outputs (4 x 32 = 128 numbers) and mixes them together:

```
Head 1 result: [32 numbers] --+
Head 2 result: [32 numbers] --+-- concatenate --> [128 numbers] --> x W_O --> [128 numbers]
Head 3 result: [32 numbers] --+                                              final output
Head 4 result: [32 numbers] --+
```

**Total: 4 matrices, each 128 x 128 = 65,536 parameters** for the attention layer.

The key insight: **Q, K, V, O are NOT hand-designed**. They start as random numbers. During training, the model adjusts them so that the Q-K comparisons produce useful attention scores. The model *discovers* what "questions" and "name tags" work best for predicting the next character.

**Does each position have its own Q, K, V? And do they change or stay static?**

- **Yes — each position has its own Q, K, V vectors.** For each of the 64 positions we take that position's embedding (128 numbers) and multiply it by the same three matrices W_Q, W_K, W_V. So we get 64 different Q vectors, 64 different K vectors, and 64 different V vectors — one set per position. The *matrices* W_Q, W_K, W_V are shared; the *vectors* are different because each position has a different embedding.
- **What changes vs what stays static:**
  - **Q, K, V vectors** — Not stored. They are **recomputed on every forward pass** from the current embeddings and the current weight matrices. So they change whenever the input sequence changes (different text, different batch).
  - **W_Q, W_K, W_V, W_O (the weight matrices)** — These are the **learned parameters**. They start random, get **updated during training** (via backpropagation), and after training we **freeze them** (keep them static). At inference time we use these fixed weights to compute Q, K, V from whatever input the user sends.

So: the *weights* are learned once and then stay static; the *Q, K, V vectors* are computed on the fly each time from the input and those fixed weights.

**2. Feed-forward (FFN)** — "Think about what you just saw"

After attention has mixed information from other positions, the feed-forward layer processes **each position by itself** (no cross-position lookups here). It’s a small 2‑layer network per position:

```
Input (128 numbers) 
    → Linear: 128 → 512   (expand)
    → ReLU
    → Linear: 512 → 128   (compress back)
    → Output (128 numbers)
```

- First matrix: 128 × 512 = 65,536 (plus bias terms we’re not counting separately here).
- Second matrix: 512 × 128 = 65,536.
- So **2 × 128 × 512 ≈ 131,072** parameters. The "4" in the formula is the expansion ratio: 128 × 4 = 512. So "feed-forward" = this per-position MLP that adds most of the parameters inside each layer.

---

**3. Layer norms** — "Keep numbers in a healthy range"

Layer norm doesn’t mix different positions; it **normalizes** the 128 numbers at each position so they don’t grow too big or too small (mean ~0, scale ~1). That keeps training stable.

- We have **2** layer norms per transformer block: one before attention, one before feed-forward.
- Each layer norm has a scale (and usually a shift) **per dimension**: 128 numbers for scale, 128 for shift → 256 parameters per norm, so **2 × 128 = 256** (often written as "2 × n_embd" when we only count the main learnable scale/shift).

So "layer norms" = these two small normalizers (one before attention, one before feed-forward) that use very few parameters compared to attention and feed-forward.

---

**Summary**


| Block             | Role                                      | Parameters (per layer) |
| ----------------- | ----------------------------------------- | ---------------------- |
| Attention Q,K,V,O | Look at other positions, blend by scores  | 65,536                 |
| Feed-forward      | Per-position "thinking" (expand → shrink) | 131,072                |
| Layer norms       | Stabilize activations (scale/shift)       | 256                    |


So in one transformer layer: **attention** does the "look around" step, **feed-forward** does the "think per position" step, and **layer norms** keep values in a good range. You’ll implement all three in Step 3.

---

**Why these values?**

- `n_embd=128` and `n_layer=6` gives ~1.2M parameters — small enough to train on CPU in minutes, large enough to learn real patterns from stories.
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

**What "create mappings" means:**  
We need a fixed way to turn characters into integers (for the model) and integers back into characters (for us). So we build two lookup structures from the dataset's unique characters: (1) **char → index**: e.g. `{'a': 0, 'b': 1, ' ': 2, ...}` so `encode("a")` gives `0`; (2) **index → char**: e.g. `{0: 'a', 1: 'b', 2: ' ', ...}` so `decode([0])` gives `"a"`. Together these are the "mappings."

**What "print sample encoded/decoded sequences" means:**  
After we have `encode` and `decode`, we run them on a short snippet of text and print the result so we can check they work. For example: take `"The cat"` → encode → `[45, 12, 89, 3, 67, 41, 23]` → decode → `"The cat"`. If the printed decoded text matches the original, the pipeline is correct. It's a sanity check.

**How to analyze vocab count:**  
Vocabulary = the set of unique characters (or tokens) in your data. To get the count:

1. **Load the raw text** (e.g. read `data/stories.txt` into one string).
2. **Collect unique characters:** e.g. `chars = sorted(set(text))` — `set(text)` removes duplicates, `sorted` gives a stable order so the same dataset always yields the same mapping.
3. **Vocab size** = number of unique characters: `vocab_size = len(chars)` (e.g. 65).
4. **Optional — print statistics:** total characters `len(text)`, vocab size `len(chars)`, and maybe the list of characters or a frequency table (how many times each character appears). Example:
  ```python
   from collections import Counter
   counts = Counter(text)
   print("Vocab size:", len(counts))
   print("Total characters:", len(text))
   print("Most common:", counts.most_common(10))
  ```

The mappings (char→index, index→char) are then built from `chars`: for example, `char_to_idx = {c: i for i, c in enumerate(chars)}` and `idx_to_char = {i: c for i, c in enumerate(chars)}`.

**Input**: Raw text file with stories
**Output**: Processed data, vocab size, encode/decode functions

**Key Concepts to Learn**:

- **Tokenization** is the bridge between human text and model numbers
- **Character-level tokenization**: simplest approach — each unique character gets an integer ID
- **Vocabulary**: the complete set of tokens the model knows
- **Train/val split**: essential to detect overfitting

**Try It Yourself**:

- What happens if you use a tiny dataset (100 words)? A large one?
- Count how many unique characters your dataset has. Why does this matter for model size?

---

**Implementation Notes**:

The script `src/01_data_preparation.py` was created with the following implementation:

- **Data Loading**: Reads `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt` from the `data/` folder (files already split, no 90/10 split needed). File paths are **configurable** via `config.py`:
  ```python
  data_dir: str = "data"
  train_file: str = "TinyStoriesV2-GPT4-train.txt"
  val_file: str = "TinyStoriesV2-GPT4-valid.txt"
  ```
  To use different data, edit these values in `src/config.py` or pass a custom config to `prepare_data(config=my_config)`.
- **Vocabulary Analysis**: Combines train and validation text, extracts unique characters using `sorted(set(text))`, resulting in 230 unique characters (includes ASCII letters, digits, punctuation, and various Unicode characters).
- **Mappings**: Creates `stoi` (char→index) and `itos` (index→char) dictionaries.
- **Encode/Decode**: Simple functions using list comprehensions.
- **Verification**: Samples 100 characters from training data, encodes then decodes, confirms perfect reconstruction.

**Dataset Stats**:

- Training: 2,226,845,268 characters (~2.2B)
- Validation: 22,493,387 characters
- Vocabulary: 230 unique characters

---

### Step 2: Tokenizer

**Script**: `src/02_tokenizer.py`

**What it does**:

- Implement `encode(text) -> list[int]`
- Implement `decode(tokens) -> str`
- Implement `get_batch()` for creating training batches
- Create input-target pairs for next-token prediction

**What "get_batch()" means:**  
The training loop needs to feed the model many small chunks of text at once (a "batch"). `get_batch()` is a function that, each time you call it, returns one batch: it picks random starting positions in the training text, slices out `block_size` characters from each, and stacks them into two tensors (or arrays): one for **inputs** and one for **targets**. So you get something like `(X, Y)` where `X` has shape `(batch_size, block_size)` and `Y` has shape `(batch_size, block_size)` (targets are the same length but shifted by one, as below). The "creating training batches" part is: sample random windows, encode them, and package as input-target pairs ready for the model.

**Example — one call to get_batch() with batch_size=2, block_size=8:**

```
Training text (encoded):  ... 54, 12, 89, 3, 67, 41, 23, 3, 14, 41, 23, 3, 45, 12, 89, 5, ... 
                           (e.g. "...The cat sat on the..." )

Random start 1: index 10  →  slice 8 tokens  →  [23, 3, 14, 41, 23, 3, 45, 12]
Random start 2: index 20  →  slice 8 tokens  →  [89, 5, 67, 41, 23, 3, 14, 41]

Stack into X (inputs):     shape (2, 8)
  Row 0: [23, 3, 14, 41, 23, 3, 45, 12]
  Row 1: [89, 5, 67, 41, 23, 3, 14, 41]

Build Y (targets) = each row shifted left by 1 (predict "next" at every position):
  Row 0: [3, 14, 41, 23, 3, 45, 12, 89]   (next after 23 is 3, next after 3 is 14, ...)
  Row 1: [5, 67, 41, 23, 3, 14, 41, 23]

get_batch() returns (X, Y). The model gets X and is trained to predict Y.
```

**What "input-target pairs for next-token prediction" means:**  
The model’s job is to predict the **next** character at every position. So for each chunk of text we need two sequences of the same length: **input** = the characters the model sees (e.g. positions 0..62), **target** = the character we want it to predict at each position (the "next" one: positions 1..63). So target is just input shifted by one. For example, if the chunk is `"The cat"` (7 chars), input might be `[T,h,e, ,c,a,t]` and target `[h,e, ,c,a,t,?]` — at position 0 we predict "h", at position 1 we predict "e", and so on. Creating these pairs means: from the full encoded text, take a window of length `block_size`, call it `x`, and form `y` as the same window shifted by one (so `y[i] = x[i+1]`). Every training example is one such (input, target) pair; `get_batch()` returns many of them at once.

**Example — one row of the batch (input vs target):**

```
One row of X (input):   [23, 3, 14, 41, 23, 3, 45, 12]   →  e.g. "t sat o"
One row of Y (target):  [ 3, 14, 41, 23, 3, 45, 12, 89]  →  e.g. " sat on"

At position 0: model sees 23 → should predict 3   (see "t" → predict " ")
At position 1: model sees 3  → should predict 14  (see " " → predict "s")
At position 2: model sees 14 → should predict 41  (see "s" → predict "a")
...
At position 7: model sees 12 → should predict 89  (see "o" → predict "n")
```

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

**Important — what we actually feed in:**  
We do **not** pass only "T" (or only [54]) as input. We pass the **entire block** (e.g. all 64 token IDs) in one forward pass. The model receives one vector per position (embedding of that token + position), and with **causal masking** each position can only "see" itself and earlier positions. So at position 0 the effective context is [54]; at position 1 it is [54, 12]; at position 2 it is [54, 12, 89]; etc. That is why we say "given [54]" or "given [54, 12]" — it means "what that position is allowed to look at," not "what we put into the model." One forward pass through the transformer produces one predicted next-token distribution at every position; we compare all of them to the target sequence in one go.

**Try It Yourself**:

```python
from src.tokenizer import Tokenizer

# Create vocabulary mappings from some text
text = "Hello World! This is a test."
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
itos = {i: ch for i, ch in enumerate(chars)}  # int to string

# Create tokenizer with the mappings
tokenizer = Tokenizer(stoi, itos)

# Encode "Hello World" to token IDs
text = "Hello World"
tokens = tokenizer.encode(text)
print(f"Text:   '{text}'")
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: '{decoded}'")

# Verify it's lossless
assert text == decoded, "Encoding/decoding should be lossless!"
print("✓ Lossless encoding verified!")
```

**Questions to explore**:

- Is the encoding lossless? (Does `decode(encode(text)) == text` always hold?)
- What happens if you encode characters not in the training data?
- What is the shape of a training batch? Why `(batch_size, block_size)`?

---

### Step 3: Transformer Model Architecture (Classical)

**Script**: `src/03_model.py`

**What it implements** — building blocks from smallest to largest:

**Summary: Step 0 through Attention** (concepts + examples for readers weak in math):

**Step 0 (config).** We use: `block_size` = 64 (max tokens per sequence), `n_embd` = 128 (every vector is 128 numbers), `n_head` = 4 (number of attention heads).

**Example — what do we pass into the block?** Use the text **"the cat is"** (10 characters). We pass the **whole sequence at once**, not one letter at a time. Each character becomes a row of 128 numbers:

```
Position:    0    1    2    3    4    5    6    7    8    9
Character:  "t"  "h"  "e"  " "  "c"  "a"  "t"  " "  "i"  "s"

x = one matrix with 10 rows and 128 columns:
    row 0 = 128 numbers for "t"
    row 1 = 128 numbers for "h"
    ...
    row 9 = 128 numbers for "s"

Shape of x: (10, 128)   →   n × 128, with n = 10 (length), max n = block_size (64).
```

So **x** is the block input: **n rows × 128 columns**. Each row is one token’s 128-d vector.

**Example — where does the position vector appear?** Only in the **first** step, when we build **x**. For each position we do: **token_emb + position_emb** (add two 128-number vectors). Example for position 0 (character "t", token ID 54):

- **Token embedding:** Look up ID 54 in the token table → one vector of 128 numbers, e.g. `[0.12, -0.34, 0.56, ..., 0.01]`.
- **Position embedding:** Look up position 0 in the position table → one vector of 128 numbers, e.g. `[0.02, 0.11, -0.05, ..., 0.08]`.
- **Add them (element-wise):** First number: 0.12 + 0.02 = 0.14. Second: -0.34 + 0.11 = -0.23. … Same for all 128 positions. Result = **x[0]**, the 128 numbers for "t" at position 0.

So the **position vector** is just that second lookup (128 numbers per position index). We do **not** add position again inside the Q/K/V attention step; it’s already inside **x**.

**Example — why must attention output be 128 numbers?** Each block does **output = x + Attention(x)** (residual). So we add two things:

- **x:** 128 numbers per position.
- **Attention(x):** must also be 128 numbers per position.

If Attention(x) were 512 numbers, we couldn’t do "128 + 512" element-wise. So we design attention so its output is 128 — same as **x**.

**Example — why 32 per head, not 128 per head?** We have 4 heads. If each head produced 128 numbers, we’d get 4 × 128 = **512** after concat, and we’d need an extra layer to squash 512 → 128. Instead we **split** 128 across heads: 128 ÷ 4 = **32 per head**. Each head outputs 32 numbers; we glue them back: 32 + 32 + 32 + 32 = **128**. So no extra squeeze: block stays 128 in, 128 out.

**Recap (no math):** Step 0 gives block_size, 128-d, 4 heads. We pass a whole sequence as **n × 128**. We build it as token_emb + position_emb (position only here). Attention output is 128 so we can add it to **x**. Four heads each do 32 numbers, concat to 128.

#### 3.1 Token + Positional Embeddings

```
Input token IDs:    [54, 12, 89, 3]
                         ↓
Token Embedding:    each ID → learned vector of size n_embd
Position Embedding: each position (0,1,2,3) → learned vector of size n_embd
                         ↓
Combined:           token_emb + position_emb → input to transformer
```

**Why positional embeddings?** Attention is permutation-invariant — without position info, "cat sat on mat" and "mat sat on cat" look identical to the model. So we have to inject position somehow. That position information is **stored in the position vector** (one vector per position index). Those vectors are **trained** (learned like other weights), and the scheme is **tied to block_size**: we have exactly one position vector for each slot in the window (0 to block_size−1), so the table has shape (block_size, n_embd).

**How position embedding works (step by step):**

- We have a **learned lookup table** for positions, similar to the token embedding table. Its shape is **(block_size, n_embd)** — one vector of length 128 for each position index 0, 1, 2, ..., 63.
- For each position in the sequence we do:
  1. **Token embedding:** Look up the vector for that token ID (e.g. 54 → 128 numbers). Same as before; static per token.
  2. **Position embedding:** Look up the vector for that **position index** (0, 1, 2, ..., 63). So position 0 always gets the same 128 numbers, position 1 gets another 128 numbers, etc. These are **learned** during training (not a fixed formula); they start random and get updated like other weights.
  3. **Combine:** Add the two vectors element-wise: `input_at_position = token_emb + position_emb`. So each position ends up with one 128-d vector that encodes both "what token is here" and "where it is in the sequence."
- Example: At position 2 we have token "e" (ID 89). We look up token 89 → vector A (128-d). We look up position 2 → vector B (128-d). We pass A + B into the transformer. So the model always sees both identity and position.

So: **token embedding** = "what character?" (same for every "e"); **position embedding** = "which slot?" (same for every position 2). Both are learned; we add them to get the actual input vector at each position. (In the modern RoPE variant, position is encoded differently — see Step 6 — but the idea is the same: we inject position so the model knows order.)

**Summary (takeaways):**

- **Both tables are trained.** The token embedding table and the position embedding table are learned parameters (updated by gradient descent). "Static" means the same token ID (or position index) always uses the same row, not that the values in that row are fixed forever.
- **Position = "where in the current window."** Position is relative to the context window (0 = first token in this window, 1 = second, …), not absolute position in the document. So when we slide the window, the same character (e.g. "C") can sit at different positions (4 → 3 → 2 → 1) and correctly gets a different position embedding each time.
- **The position table contents do not change per forward pass.** The 64×128 numbers are fixed at inference; only during training do they get updated. What changes from pass to pass is **which row we look up** for each token — because the same character moves to a different slot in the new window. So the lookup index changes; the table does not.
- **Why a lookup table and add?** Attention is permutation-invariant, so we must inject position. A learned vector per position (lookup table) is simple and flexible; adding it to the token embedding keeps dimension the same and lets the model use both "what" and "where" in one vector.

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

- Count the model's total parameters. Where do most parameters live?
- What happens if you set `n_head=1`? How about `n_layer=1`?
- Remove the causal mask — can the model still learn? (Hint: yes, but it's useless for generation)

---

### Step 4: Training Loop

**Script**: `src/04_train.py`

**Big picture**: In Step 3 you built the "brain" (the transformer model). Step 4 is where you actually *teach* that brain using data. This is where the model sees millions of input-target pairs and slowly adjusts its weights so that its next-character predictions get better over time.

**What you actually build in this step** is not a new model, but a **training script** that coordinates all the pieces you already have (data pipeline, tokenizer, model) and makes them learn together.

At a high level, in `04_train.py` you will:

1. **Build the training setup**
   - Load the TinyStories text and create a `get_batch(split)` function for `"train"` and `"val"`.
   - Load `ModelConfig` (hyperparameters) and set up the random seed and device (CPU/GPU).
2. **Instantiate the model and optimizer**
   - Create the transformer model from Step 3 and move it to the device.
   - Create an **AdamW optimizer** and a **learning rate schedule** (warmup + cosine decay).
3. **Implement the training loop**
   - For `max_iters` steps:
     - sample a random batch of `(X, Y)` pairs from the training data,
     - run a **forward pass** (compute predictions and cross-entropy loss),
     - run a **backward pass** (compute gradients with `loss.backward()`),
     - optionally clip gradients,
     - step the optimizer and update the learning rate,
     - every `eval_interval` steps:
       - compute validation loss on held-out data,
       - print a training log line (iter, train loss, val loss, LR),
       - optionally generate a short sample story from the current model.
4. **Handle outputs and checkpoints**
   - Track the **best validation loss** seen so far.
   - When you get a new best, save a checkpoint to `models/` (e.g. `models/tinystories_step4_best.pt`).
   - Optionally, log metrics to TensorBoard for later visualization.

**End result of Step 4**:

- A **trained (or partially trained) model checkpoint** on disk that Step 5 can load for generation.
- A record of training progress (printed logs, and optionally TensorBoard logs) that shows how loss and sample quality improved over time.

#### 4.1 Initializing the Model and Optimizer

**Initialize the model**:

- You construct the model from `ModelConfig`, for example:
  - `block_size = 64`, `n_embd = 128`, `n_head = 4`, `n_layer = 6`.
- All weights start out **random** (small random numbers). At this point, the model's predictions are basically noise.
- You then move it to the device:
  - GPU if available (`cuda`),
  - Apple Silicon (`mps`),
  - otherwise CPU.

**Initialize the optimizer (AdamW)**:

- You tell PyTorch:
  - "Here are all the model parameters" (`model.parameters()`),
  - "Use this base learning rate" (`learning_rate` from config),
  - "Use this weight decay" (`weight_decay` from config).
- AdamW is a smarter version of gradient descent that:
  - keeps a moving average of gradients (momentum),
  - keeps a moving average of squared gradients (to adapt step sizes),
  - applies **weight decay** in a clean way (shrinks weights toward zero a tiny bit each step to reduce overfitting).

Conceptually, for each parameter `w` and its gradient `g`, AdamW does:

- smooth the gradient over time (so updates are less noisy),
- take a step in the direction that reduces loss,
- slightly shrink `w` toward zero (weight decay).

You do not need to implement the math yourself; PyTorch's `torch.optim.AdamW` handles it.

#### 4.2 The Training Step (One Iteration)

One training iteration looks like this in plain language:

1. **Sample a batch**  
   - Call something like `x, y = get_batch("train")`.
   - `x` has shape `(batch_size, block_size)` — token IDs for input.
   - `y` has the same shape — token IDs for the targets (shifted by 1).

2. **Forward pass (prediction)**  
   - Call `logits, loss = model(x, y)`.
   - The model:
     - turns token IDs into embeddings,
     - runs them through all transformer blocks,
     - applies the final linear layer to get **logits** (unnormalized scores) for each vocab token at each position,
     - computes **cross-entropy loss** by comparing logits to the true target tokens `y`.

3. **Backward pass (gradient computation)**  
   - Call `optimizer.zero_grad()` to clear old gradients.
   - Call `loss.backward()`.
   - PyTorch walks backward through all operations (this is **backpropagation**) and:
     - computes, for each parameter, how much a tiny change in that parameter would change the loss.
     - stores that in `param.grad`.

4. **Gradient clipping (safety)**  
   - Sometimes gradients can become huge ("exploding gradients").
   - You can call `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
   - This rescales gradients so that their total norm is at most 1.0.

5. **Optimizer step (update weights)**  
   - Call `optimizer.step()`.
   - AdamW looks at each parameter and its gradient and updates it:
     - moves it a tiny amount in the direction that should reduce loss,
     - applies weight decay if configured.

6. **Update learning rate**  
   - Either:
     - use a built-in scheduler, or
     - manually compute the current learning rate based on the iteration number (warmup + cosine decay) and set it on the optimizer.

After one iteration, the model's weights have changed slightly. After thousands of iterations, these small changes accumulate and the model's predictions become much better.

#### 4.3 Cross-Entropy Loss (What We Minimize)

For each position in each sequence:

- The model outputs a vector of length `vocab_size` of logits (scores) — one score per possible next character.
- We apply `softmax` internally to turn logits into probabilities.
- Cross-entropy loss answers: "How wrong was our predicted probability distribution compared to the true next token?"

Intuition:

- If the true next token is "a" and the model assigns:
  - 80% probability to "a" → loss is small (good).
  - 1% probability to "a" → loss is large (bad).
- Cross-entropy averages this over:
  - all positions in the block,
  - all sequences in the batch.

So on a batch of shape `(batch_size, block_size)` you are effectively computing the average "wrongness" over `batch_size * block_size` next-token predictions in one go.

#### 4.4 Learning Rate Schedule (Warmup + Cosine Decay)

If the learning rate is too:

- high → training can blow up, loss becomes `nan`,
- low → training is very slow and might get stuck.

Modern transformers use a schedule instead of a fixed learning rate:

1. **Warmup phase**  
   - Start with a very small learning rate, for example:
     - at iteration 0: `lr = min_lr` or `0`,
     - grow linearly to `learning_rate` by `warmup_iters`.
   - This gently "nudges" the model instead of slamming it with huge updates when weights are still random.

2. **Cosine decay phase**  
   - After warmup, slowly decrease the learning rate from `learning_rate` down to `min_lr` using a cosine-shaped curve over the remaining iterations.
   - Early on, the model takes big steps (learns fast); later it takes smaller and smaller steps (fine-tuning).

Visually:

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

This is the schedule used in many real-world LLMs (GPT-2, GPT-3, LLaMA, etc.) because it gives stable and efficient training.

#### 4.5 Train vs Validation and Overfitting

During training you track two losses:

- **Train loss**:
  - computed on batches from the training split,
  - usually goes down steadily if training is working.
- **Validation (val) loss**:
  - computed on batches from a held-out validation split,
  - tells you how well the model generalizes to text it has not directly trained on.

Signs of overfitting:

- Train loss keeps going down.
- Validation loss stops going down and starts to rise.

When that happens, the model is starting to "memorize" training text instead of learning general patterns. You typically:

- save the checkpoint with the **lowest validation loss**,
- use that one for inference.

#### 4.6 Example Training Log

You might see logs like:

```
Iter    0 | Train loss: 4.174 | Val loss: 4.171 | LR: 0.000003
Iter  250 | Train loss: 2.483 | Val loss: 2.511 | LR: 0.000300
Iter  500 | Train loss: 1.972 | Val loss: 2.053 | LR: 0.000287
...
Sample @ iter 500: "Once upon a time there was a little girl who lived..."
```

Interpretation:

- Early on, both losses are high (around the log of vocab size).
- As training continues, both go down.
- Periodically you also print a **sample generation** so you can see qualitatively how the model is improving (does it still output gibberish, or does it start forming sentences?).

#### 4.7 Try It Yourself

- Plot training vs validation loss over time. Where does validation loss stop improving?
- Try `learning_rate = 0.01` (too high for this model) — what happens to loss?
- Try `learning_rate = 0.00001` (very low) — does the model still learn, just more slowly?
- Remove warmup — do you see more instability early in training (loss spikes, `nan`, etc.)?
- Turn off weight decay — does validation loss get worse (more overfitting)?

Once you finish Step 4 and have a checkpoint with reasonable validation loss, you are ready for **Step 5: Inference/Generation**, where you actually talk to your mini-LLM.

---

### Step 5: Inference / Generation

**Script**: `src/05_inference.py`

**What it does**:

- Load trained model checkpoint
- Interactive command-line chat interface
- Multiple sampling strategies:

**Sampling Strategies** (from simplest to most sophisticated):


| Strategy               | Description                                        | Tradeoff                                         |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **Greedy**             | Always pick highest-probability token              | Deterministic but repetitive                     |
| **Temperature**        | Scale logits by T before softmax                   | T<1 = focused, T>1 = creative                    |
| **Top-k**              | Only consider top k tokens                         | Removes low-probability noise                    |
| **Top-p (nucleus)**    | Only consider tokens summing to p probability mass | Adaptive — fewer options when model is confident |
| **Repetition penalty** | Reduce probability of recently generated tokens    | Prevents loops                                   |


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

- Generate with temperature=0.1 vs temperature=2.0 — describe the difference
- What happens with top_k=1? (same as greedy)
- Try top_p=0.5 vs top_p=0.95 — which gives better outputs?

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

- Train classical vs modern with same config — compare final validation loss
- Measure training speed (iters/sec) with RMSNorm vs LayerNorm
- Generate long text with and without KV-cache — compare wall-clock time

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

**Where to download (best options)**:


| Source                          | Why use it                                                                                        | How to get it                                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TinyStories (Hugging Face)**  | **Best for this project.** Short, simple sentences; designed for small LMs; clean and consistent. | [TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories) — use `datasets` or the Hub UI; pick a split (e.g. `train`), load as text, then concatenate and save as `data/stories.txt`. |
| **Project Gutenberg**           | Free, public-domain books; good for classic fairy tales and prose.                                | [gutenberg.org](https://www.gutenberg.org/) — search e.g. "Grimm" or "Aesop"; open a book → "Plain Text UTF-8" → copy or download. Paste/append into `data/stories.txt`.                                          |
| **WikiText, OpenWebText, etc.** | Larger, more varied; better for scaling up later.                                                 | [Hugging Face Datasets](https://huggingface.co/datasets) — search "wikitext" or "openwebtext"; load and export a subset as one text file.                                                                         |


**Recommended:** Start with **TinyStories** (or a few **Project Gutenberg** fairy tales) so the dataset is small, coherent, and easy to inspect. Move to larger corpora once the pipeline works.

**Other options**: Write your own stories, or use any plain-text file you have (one story or book per line or concatenated) and save it as `data/stories.txt`.

**Example Content**:

```
Once upon a time, in a land far away, there lived a young princess named Aurora. 
She loved to explore the enchanted forest near her castle. One day, while wandering 
through the woods, she discovered a hidden cave behind a waterfall...
```

---

## Expected Results

Given our small scale (tiny model, small dataset):


| Metric           | Classical Architecture          | Modern Architecture                       |
| ---------------- | ------------------------------- | ----------------------------------------- |
| Parameters       | ~500K - 2M                      | ~500K - 2M (same budget)                  |
| Training Time    | 5-30 min (CPU) / 1-5 min (GPU)  | Similar or slightly faster                |
| Final Val Loss   | ~1.8 - 2.5                      | ~1.6 - 2.3                                |
| Output Quality   | Coherent phrases, basic grammar | Slightly more coherent, fewer repetitions |
| Context Handling | Fixed max length                | Can generalize to longer (RoPE)           |


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


| Extension                          | Difficulty | What You'll Learn                        |
| ---------------------------------- | ---------- | ---------------------------------------- |
| BPE tokenizer with tiktoken        | Medium     | Sub-word tokenization, vocabulary design |
| Larger dataset (Project Gutenberg) | Easy       | Scaling laws, data quality effects       |
| Multi-GPU training (DDP)           | Medium     | Distributed training fundamentals        |
| LoRA fine-tuning                   | Medium     | Parameter-efficient adaptation           |
| Export to GGUF format              | Hard       | Quantization, cross-platform deployment  |
| Add instruction-following (SFT)    | Hard       | Alignment, chat formatting               |
| Mixture of Experts (MoE)           | Hard       | Sparse computation, routing              |
| Speculative decoding               | Hard       | Inference optimization, draft models     |
| Flash Attention                    | Hard       | Memory-efficient attention, IO awareness |


---

## Glossary


| Term                    | Definition                                                            |
| ----------------------- | --------------------------------------------------------------------- |
| **Token**               | The smallest unit of text the model processes (character or sub-word) |
| **Embedding**           | A learned dense vector representation of a token                      |
| **Attention**           | Mechanism that lets tokens "communicate" with each other              |
| **Causal mask**         | Prevents tokens from seeing future positions during generation        |
| **Logits**              | Raw model output scores before softmax normalization                  |
| **Softmax**             | Converts logits to a probability distribution that sums to 1          |
| **Cross-entropy**       | Loss function measuring prediction quality                            |
| **Backpropagation**     | Algorithm to compute gradients via the chain rule                     |
| **Residual connection** | Skip connection that adds input directly to output                    |
| **LayerNorm / RMSNorm** | Normalization techniques to stabilize training                        |
| **FFN**                 | Feed-Forward Network — the per-token MLP in each block                |
| **KV-Cache**            | Stored key/value tensors to avoid recomputation during generation     |
| **RoPE**                | Rotary Position Embedding — encodes position via rotation             |
| **SwiGLU**              | Gated activation function used in modern transformer FFNs             |
| **BPE**                 | Byte-Pair Encoding — sub-word tokenization algorithm                  |
| **LoRA**                | Low-Rank Adaptation — efficient fine-tuning technique                 |
| **RLHF**                | Reinforcement Learning from Human Feedback — alignment technique      |
| **DPO**                 | Direct Preference Optimization — simpler alternative to RLHF          |


---

## References

### Foundational Papers

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — The original Transformer
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020)

### Modern Architecture References

1. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE, 2021)
2. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU, 2020)
3. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (GQA, 2023)
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (LLaMA, 2023)

### Learning Resources

1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide
2. [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy's video
3. [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT implementation
4. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP's line-by-line walkthrough
5. [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Visual intuition for deep learning

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