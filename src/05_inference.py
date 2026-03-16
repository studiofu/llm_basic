"""Step 5: Inference / Generation - Chat with your trained LLM.

This script loads a trained checkpoint and provides an interactive
command-line chat interface to generate stories and text.

Usage:
    $ uv run python src/05_inference.py
    
    # Or with a specific checkpoint:
    $ uv run python src/05_inference.py --checkpoint models/tinystories_best.pt

Chat Commands (type during chat):
    /help               Show available commands
    /quit or /exit      Exit the chat
    /reset              Clear conversation history
    /temp <value>       Set temperature (0.1-2.0, default: 0.8)
    /topk <value>       Set top-k filtering (1-100, default: 50)
    /topp <value>       Set top-p/nucleus filtering (0.1-1.0, default: 0.9)
    /max <value>        Set max tokens to generate (10-500, default: 200)
    /repetition <value> Set repetition penalty (1.0-2.0, default: 1.1)
    /params             Show current generation parameters
    /prompt <text>      Set a custom system prompt prefix

Example Session:
    > Once upon a time
    Once upon a time there was a little girl named Lily. She loved to play
    in the garden and watch the butterflies...
    
    > /temp 1.2
    Temperature set to 1.2
    
    > Tell me about a brave knight
    In a kingdom far away, there lived a knight named Sir Cedric. He was
    known throughout the land for his courage...
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from config import ModelConfig, get_config


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
GPTLanguageModel = _model.GPTLanguageModel


class ChatSession:
    """Interactive chat session with the language model.
    
    Manages conversation history, generation parameters, and provides
    a command-line interface for chatting with the model.
    
    Attributes:
        model: The GPT language model
        tokenizer: Character-level tokenizer
        config: ModelConfig with generation parameters
        device: Device to run inference on
        history: List of (role, text) tuples for conversation context
        system_prompt: Optional prefix for generation
    """
    
    def __init__(
        self,
        model: GPTLanguageModel,
        tokenizer: Tokenizer,
        config: ModelConfig,
    ):
        """Initialize chat session.
        
        Args:
            model: Trained GPT language model
            tokenizer: Tokenizer for encoding/decoding
            config: Configuration with generation parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.history = []
        self.system_prompt = ""
        
        # Generation parameters (can be changed via commands)
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.max_new_tokens = config.max_new_tokens
        self.repetition_penalty = config.repetition_penalty
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate(self, prompt: str, include_history: bool = False) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text to continue from
            include_history: Whether to prepend conversation history
        
        Returns:
            Generated text string
        """
        # Build full context
        if include_history and self.history:
            # Use last exchange as context (simplified)
            context = ""
            for role, text in self.history[-2:]:  # Last 2 exchanges
                context += text + "\n\n"
            full_prompt = context + prompt
        else:
            full_prompt = self.system_prompt + prompt
        
        # Encode prompt
        input_ids = self.tokenizer.encode(full_prompt)
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            # Use model's generate method
            generated = self.model.generate(
                idx,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k if self.top_k > 0 else None,
            )
        
        # Decode full sequence
        full_text = self.tokenizer.decode(generated[0].tolist())
        
        # Extract only the new generated part
        generated_text = full_text[len(full_prompt):]
        
        return generated_text.strip()
    
    def show_params(self):
        """Display current generation parameters."""
        print("\n" + "=" * 50)
        print("Current Generation Parameters:")
        print("=" * 50)
        print(f"  Temperature:       {self.temperature:.2f}")
        print(f"  Top-k:             {self.top_k}")
        print(f"  Top-p:             {self.top_p:.2f}")
        print(f"  Max new tokens:    {self.max_new_tokens}")
        print(f"  Repetition penalty: {self.repetition_penalty:.2f}")
        print(f"  System prompt:     {repr(self.system_prompt) if self.system_prompt else '(none)'}")
        print("=" * 50 + "\n")
    
    def show_help(self):
        """Display help message with available commands."""
        help_text = """
╔══════════════════════════════════════════════════════════════════╗
║                        CHAT COMMANDS                             ║
╠══════════════════════════════════════════════════════════════════╣
║  /help, /?          Show this help message                       ║
║  /quit, /exit       Exit the chat                                ║
║  /reset             Clear conversation history                   ║
║                                                                    ║
║  GENERATION PARAMETERS:                                          ║
║  /temp <value>      Set temperature (0.1-2.0)                    ║
║                      • Low (0.1-0.5): Focused, deterministic     ║
║                      • Medium (0.6-1.0): Balanced                ║
║                      • High (1.1-2.0): Creative, random          ║
║  /topk <value>      Set top-k filtering (1-100, 0=disabled)      ║
║                      • Low (1-10): Only most likely tokens       ║
║                      • High (50-100): More variety               ║
║  /topp <value>      Set nucleus filtering (0.1-1.0)              ║
║                      • Low (0.1-0.5): Very focused               ║
║                      • High (0.9-1.0): More randomness           ║
║  /max <value>       Set max tokens to generate (10-500)          ║
║  /repetition <val>  Set repetition penalty (1.0-2.0)             ║
║                      • Higher = less repetition                  ║
║                                                                    ║
║  OTHER COMMANDS:                                                 ║
║  /params            Show current parameters                      ║
║  /prompt <text>     Set system prompt prefix                     ║
║  /clearprompt       Clear system prompt                          ║
╚══════════════════════════════════════════════════════════════════╝

Tips:
• Start with short prompts for better results
• Temperature 0.8-1.0 works well for creative stories
• Lower temperature (0.5-0.7) for more coherent text
• The model works best with story-style prompts
"""
        print(help_text)
    
    def handle_command(self, command: str) -> bool:
        """Handle a chat command.
        
        Args:
            command: Command string (without the leading /)
        
        Returns:
            True to continue chat, False to exit
        """
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return True
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ('quit', 'exit', 'q'):
            print("\nGoodbye! 👋\n")
            return False
        
        elif cmd in ('help', 'h', '?'):
            self.show_help()
        
        elif cmd == 'reset':
            self.history.clear()
            print("\n✓ Conversation history cleared.\n")
        
        elif cmd == 'params':
            self.show_params()
        
        elif cmd == 'temp':
            try:
                temp = float(arg)
                if 0.0 < temp <= 2.0:
                    self.temperature = temp
                    print(f"\n✓ Temperature set to {temp:.2f}\n")
                else:
                    print("\n✗ Temperature must be between 0.0 and 2.0\n")
            except ValueError:
                print("\n✗ Invalid temperature value\n")
        
        elif cmd == 'topk':
            try:
                topk = int(arg)
                if 0 <= topk <= 200:
                    self.top_k = topk
                    print(f"\n✓ Top-k set to {topk}\n")
                else:
                    print("\n✗ Top-k must be between 0 and 200\n")
            except ValueError:
                print("\n✗ Invalid top-k value\n")
        
        elif cmd == 'topp':
            try:
                topp = float(arg)
                if 0.0 < topp <= 1.0:
                    self.top_p = topp
                    print(f"\n✓ Top-p set to {topp:.2f}\n")
                else:
                    print("\n✗ Top-p must be between 0.0 and 1.0\n")
            except ValueError:
                print("\n✗ Invalid top-p value\n")
        
        elif cmd == 'max':
            try:
                max_tokens = int(arg)
                if 10 <= max_tokens <= 1000:
                    self.max_new_tokens = max_tokens
                    print(f"\n✓ Max tokens set to {max_tokens}\n")
                else:
                    print("\n✗ Max tokens must be between 10 and 1000\n")
            except ValueError:
                print("\n✗ Invalid max tokens value\n")
        
        elif cmd == 'repetition':
            try:
                rep = float(arg)
                if 1.0 <= rep <= 2.0:
                    self.repetition_penalty = rep
                    print(f"\n✓ Repetition penalty set to {rep:.2f}\n")
                else:
                    print("\n✗ Repetition penalty must be between 1.0 and 2.0\n")
            except ValueError:
                print("\n✗ Invalid repetition penalty value\n")
        
        elif cmd == 'prompt':
            if arg:
                self.system_prompt = arg + " "
                print(f"\n✓ System prompt set to: {repr(self.system_prompt)}\n")
            else:
                print("\n✗ Please provide a prompt text\n")
        
        elif cmd == 'clearprompt':
            self.system_prompt = ""
            print("\n✓ System prompt cleared.\n")
        
        else:
            print(f"\n✗ Unknown command: /{cmd}")
            print("Type /help for available commands.\n")
        
        return True
    
    def chat_loop(self):
        """Run the interactive chat loop."""
        print("\n" + "=" * 60)
        print("  🚀 Mini-LLM Chat Interface")
        print("=" * 60)
        print("\nType your prompt and press Enter to generate.")
        print("Type /help for available commands, /quit to exit.\n")
        
        # Show initial parameters
        self.show_params()
        
        while True:
            try:
                # Get user input
                user_input = input("> ").strip()
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:]  # Remove leading /
                    if not self.handle_command(command):
                        break
                    continue
                
                # Generate response
                print("\n" + "-" * 40)
                print("Generating...\n")
                
                generated = self.generate(user_input, include_history=False)
                
                # Print response
                print(generated)
                print("-" * 40 + "\n")
                
                # Save to history
                self.history.append(('user', user_input))
                self.history.append(('assistant', generated))
                
                # Trim history if too long (keep last 4 exchanges)
                if len(self.history) > 8:
                    self.history = self.history[-8:]
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.\n")
            except Exception as e:
                print(f"\n✗ Error: {e}\n")


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def load_checkpoint(
    checkpoint_path: Path,
    device: str = "auto",
) -> tuple[GPTLanguageModel, Tokenizer, ModelConfig]:
    """Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Resolve device before loading
    resolved_device = _resolve_device(device)
    
    # Load checkpoint with resolved device
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    
    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✓ Loaded config from checkpoint")
    else:
        config = get_config()
        print("  ! Using default config (no config in checkpoint)")
    
    # Override device if specified
    config.device = resolved_device
    
    # Load data to get tokenizer
    print("\nLoading data for tokenizer...")
    train_text, val_text, stoi, itos, vocab_size, encode, decode = prepare_data(config)
    tokenizer = Tokenizer(stoi, itos)
    
    # Update vocab size in config
    config.vocab_size = vocab_size
    
    # Create model
    print("\nCreating model...")
    model = GPTLanguageModel(config)
    model = model.to(config.device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Model weights loaded")
    
    # Show checkpoint info
    if 'iter_num' in checkpoint:
        print(f"  Checkpoint from iteration: {checkpoint['iter_num']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel has {total_params:,} parameters")
    
    return model, tokenizer, config


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Chat with your trained mini-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load default checkpoint and chat
  uv run python src/05_inference.py
  
  # Load specific checkpoint
  uv run python src/05_inference.py --checkpoint models/my_model.pt
  
  # Use CPU even if GPU is available
  uv run python src/05_inference.py --device cpu
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: models/tinystories_best.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to run on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        project_root = Path(__file__).parent.parent
        checkpoint_path = project_root / "models" / "tinystories_best.pt"
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        print("\nPlease train a model first:")
        print("  uv run python src/04_train.py")
        print("\nOr specify a different checkpoint:")
        print(f"  uv run python src/05_inference.py --checkpoint <path>")
        return
    
    # Load checkpoint
    try:
        model, tokenizer, config = load_checkpoint(checkpoint_path, args.device)
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create and start chat session
    chat = ChatSession(model, tokenizer, config)
    chat.chat_loop()


if __name__ == "__main__":
    main()
