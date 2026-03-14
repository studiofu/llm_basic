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

#### What do all these settings mean? (Beginner-Friendly Guide)

Don't worry if the names above look intimidating. Every single one maps to a simple idea. Think of building an LLM like teaching a student to write stories — these settings control *how* that student learns.

---

**DATA SETTINGS — How we feed text to the model**

**`vocab_size`** — The dictionary size
> Imagine you give a student a dictionary. `vocab_size` is how many unique "words" (in our case, unique characters like `a`, `b`, `!`, ` `) are in that dictionary. If your training text uses 65 unique characters, the vocab_size is 65. The model can only read and write characters it has seen in this dictionary.

**`block_size = 64`** — The reading window (context length)
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

**`batch_size = 32`** — How many text windows to study at once
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
> - **Speed**: GPUs are designed to do many calculations in parallel. Processing 32 windows takes almost the same time as processing 1.
> - **Stability**: Each single window gives a slightly different "lesson." Averaging the lessons from 32 windows gives a more reliable signal for how to improve. It's like asking 32 students what they think the answer is and taking the average, instead of relying on one student's possibly-wrong guess.
>
> **Tradeoffs**:
> - Bigger batch (64, 128) = smoother learning, but uses more memory
> - Smaller batch (8, 16) = noisier learning, but fits on smaller hardware
> - If you get an "out of memory" error, the first thing to try is reducing batch_size

---

**MODEL ARCHITECTURE — The shape of the model's "brain"**

**`n_embd = 128`** — Embedding dimension (how rich each character's representation is)
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

**`n_layer = 6`** — Number of transformer layers (how deep the thinking goes)
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
> - **Earlier layers** (closer to input) tend to pick up simpler, more local patterns
> - **Later layers** (closer to output) tend to pick up more abstract, longer-range patterns
>
> But this is a *tendency*, not a rule. In practice:
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

**`n_head = 4`** — Number of attention heads (different ways of paying attention)
>
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
> - `n_head = 1` → only one attention pattern per layer (limited)
> - `n_head = 4` → four parallel patterns (good balance for our small model)
> - `n_head = 128` → would mean each head only gets 128/128 = 1 number to work with (too small to be useful)
> - Rule of thumb: `n_embd / n_head` should be at least 16-32 for each head to have enough capacity

**`dropout = 0.1`** — Random forgetting (prevents memorizing)
> During training, the model randomly "turns off" 10% of its neurons at each step. This sounds destructive — why would disabling parts of the model help? Let's look at a concrete example.
>
> **The problem dropout solves: overfitting (memorizing instead of learning)**
>
> Imagine our training data contains these sentences:
> ```
> "The cat sat on the mat."
> "The dog sat on the rug."
> "The bird sat on the branch."
> ```
>
> What we WANT the model to learn:
> - After "The [animal] sat on the", predict a reasonable surface/place
> - General pattern: [animal] + [sat on] + [surface]
>
> What can go WRONG without dropout:
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

**`max_iters = 5000`** — Total training steps
> One "iteration" = the model looks at one batch of data, makes predictions, checks how wrong it was, and adjusts itself. 5000 iterations means 5000 rounds of practice.
>
> Think of it as doing 5000 practice tests. Early iterations = wild guessing. Later iterations = the model has learned real patterns.

**`learning_rate = 3e-4`** — Step size (how much to adjust each time)
> After each practice round, the model needs to adjust its internal settings. The learning rate controls *how big* each adjustment is.
>
> `3e-4` means `0.0003` — a very small number. This is deliberate:
> - **Too big** (like 0.01): The model overcorrects wildly, like a student who panics after every mistake and rewrites everything
> - **Too small** (like 0.000001): The model barely changes, like a student who ignores feedback
> - **Just right** (0.0003): Small, steady improvements each round
>
> ```
> learning_rate = 0.01    →  Too big  — model explodes, loss goes to infinity
> learning_rate = 0.0003  →  Sweet spot — steady improvement
> learning_rate = 0.000001 → Too small — model barely learns, wastes time
> ```

**`warmup_iters = 100`** — Gentle start
> For the first 100 iterations, the learning rate starts at nearly zero and gradually increases to `3e-4`. This is like warming up before exercise — jumping straight into a high learning rate when the model's weights are still random can cause instability.

**`min_lr = 3e-5`** — Slow down near the end
> As training progresses, the learning rate gradually decreases from `3e-4` down to `3e-5` (10× smaller). This is like taking smaller steps as you get closer to the answer — big adjustments early on, fine-tuning later.

**`weight_decay = 0.1`** — Prevent laziness
> This gently pushes the model's internal numbers toward zero, preventing them from growing unnecessarily large. It's a form of regularization — keeping the model "lean" so it learns general patterns instead of overly specific ones.

**`eval_interval = 250`** — How often to check progress
> Every 250 training steps, we pause and test the model on text it has *never seen* (the validation set). This tells us if the model is actually learning vs. just memorizing.

**`eval_iters = 200`** — How thorough each check is
> When we do check progress, we test on 200 different batches and average the results. More batches = more reliable measurement of how good the model is.

---

**GENERATION SETTINGS — How the model writes text**

**`temperature = 0.8`** — Creativity dial
> Controls how "random" the model's word choices are when generating text.
>
> Think of it as a confidence dial:
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

**`top_k = 50`** — Only consider the best options
> Before picking the next character, throw away everything except the top 50 most likely candidates. This removes obviously bad choices (like putting "Z" after "th").

**`top_p = 0.9`** — Smart cutoff (nucleus sampling)
> Instead of a fixed number of candidates (top_k), keep adding candidates from most-likely to least-likely until their combined probability reaches 90%. When the model is very confident, this might mean only 2-3 options. When it's uncertain, it might keep 40+. It adapts automatically.

**`repetition_penalty = 1.1`** — Don't repeat yourself
> Reduces the probability of characters/words that already appeared recently. Without this, models tend to get stuck in loops: *"the cat the cat the cat the cat..."*

**`max_new_tokens = 200`** — When to stop writing
> The maximum number of characters the model will generate per response. Without a limit, it would write forever.

---

**SYSTEM SETTINGS — Hardware and reproducibility**

**`device = "auto"`** — Where to run the computation
> - `"cuda"` → NVIDIA GPU (fastest, 10-100× faster than CPU)
> - `"mps"` → Apple Silicon GPU (Mac M1/M2/M3)
> - `"cpu"` → Regular processor (slowest, but always available)
> - `"auto"` → Automatically picks the best available option

**`compile_model = False`** — Speed optimization
> When set to `True`, PyTorch analyzes the entire model and optimizes it as one unit, making training ~2× faster. We keep it off by default because it requires PyTorch 2.0+ and adds startup time. Turn it on once you've verified everything works.

**`seed = 42`** — Reproducibility
> Neural networks use random numbers (for initialization, dropout, batching). Setting a seed means the "random" numbers are the same every time you run the code. This way, if you and a friend both run the same code, you'll get identical results. (42 is a tradition in programming — a reference to *The Hitchhiker's Guide to the Galaxy*.)

---

**MODERN UPGRADE FLAGS — Toggle new techniques on/off**

These are all `False` by default. You'll start with the classical 2017 architecture, then flip these on one-by-one in Step 6 to see how modern LLMs improved on the original design.

| Flag | What it changes | When to enable |
|------|----------------|----------------|
| `use_rope` | How the model knows word position in a sentence | Step 6.1 |
| `use_rmsnorm` | How the model normalizes numbers between layers | Step 6.2 |
| `use_swiglu` | How neurons "activate" inside each layer | Step 6.3 |
| `use_gqa` | How attention heads share work | Step 6.4 |
| `use_kv_cache` | How fast the model generates text | Step 6.5 |

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
