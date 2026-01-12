"""
Inference script for text generation
"""

import torch
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import LLMConfig
from tokenizer import SimpleTokenizer
from model import AIModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureInferenceEngine:
    """Inference engine for text generation with safety checks."""
    
    def __init__(
            self,
            model: AIModel,
            tokenizer: SimpleTokenizer,
            device: str = 'cuda',
            max_length: int = 512
    ):
        """Initialize inference engine."""
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Inference engine ready on {self.device}")

    def _validate_prompt(self, prompt:str) -> str:
        """Validate and clean input prompt."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string.")
        
        # remove null bytes
        prompt = prompt.replace('\x00', '')

        # check length
        if len(prompt) > 10000:
            logger.warning("Prompt too long, truncating")
            prompt = prompt[:10000]

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        return prompt.strip()
    
    def _validate_generatioin_params(
            self,
            max_new_tokens: int,
            temperature: float,
            top_k: Optional[int],
            top_p: Optional[float]
        ) -> tuple:
        """
        Validate generation parameters.

        Returns:
            Tuple of validated parameters
        """
        # Validate max_new_tokens
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        
        max_new_tokens = min(max_new_tokens, self.max_length)

        # Validate temperature
        temperature = max(0.1, min(temperature, 2.0))

        # validate top_k
        if top_k is not None:
            if top_k < 1:
                logger.warning("top_k must be positive, disabling")
                top_k = None
            else:
                top_k = min(top_k, self.model.config.vocab_size)
        
        # Validate top_p
        if top_p is not None:
            if not 0 < top_p <= 1:
                logger.warning("top_p must be  in (0, 1], disabling")
                top_p = None
        return max_new_tokens, temperature, top_k, top_p
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1
    ) -> list[str]:
        """
        Generate text from prompt with safety constraints.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        # Validate inputs
        prompt = self._validate_prompt(prompt)
        max_new_tokens, temperature, top_k, top_p = self._validate_generatioin_params(
            max_new_tokens, temperature, top_k, top_p
        )
        
        if num_return_sequences < 1 or num_return_sequences > 10:
            raise ValueError("num _return_sequences must be in [1,10]")
        
        # Encode Prompt
        try:
            token_ids = self.tokenizer.encode(prompt)
        except Exception as e:
            raise ValueError(f"Failed to encode prompt: {e}")
        
        if not token_ids:
            raise ValueError("Prompt encoding resulted in empty sequence")
        
        # Create  input tensor
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        idx = idx.repeat(num_return_sequences, 1)
        
        logger.info(
           f"Generating {num_return_sequences} sequence(s) "
           f"(max_new_tokens={max_new_tokens}, temperature={temperature})"
        )
        
        # Generate
        try:
            generated_ids = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        except Exception as e:
            logger.error(F"Generation failed: {e}")
            raise
        
        # Decode sequences
        results = []
        for  i in range(num_return_sequences):
            token_list = generated_ids[i].tolist()
            try:
                text = self.tokenizer.decode(token_list)
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to decode sequence {i}: {e}")
                results.append("[DECODING ERROR]")
        
        return results
    
    def interactive_mode(self):
        """
        Run interactive  generation mode.
        Allows continuous text generation with user prompts.
        """
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE GENERATION MODE")
        logger.info("="*60)
        logger.info("Type 'quit' or 'exit' to stop")
        logger.info("Type 'help' for commands")
        logger.info("="*60 + "\n")
        
        params = {
            'max_new_tokens': 100,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95
        }
        
        while True:
            try:
                # Get user input
                prompt = input("\nPrompt: ").strip()
                
                # Handle commands
                if prompt.lowwer() in ['quit', 'exit', 'q']:
                    logger.info("Exiting interactive mode")
                    break
                
                if prompt.lower() == 'help':
                    print("\nCommands:")
                    print("  quit/exit - Exit interactive mode")
                    print("  help - Show this help message")
                    print("  set <param> <value> - Set generation parameter")
                    print("\nCurrent parameters:")
                    for key, value in params.items():
                        print(f"  {key}: {value}")
                    continue
                
                if prompt.lower().startswith('set'):
                    parts = prompt.split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        if param in params:
                            try:
                                if param == 'top_k':
                                    params[param] = int(value) if value != 'None' else None
                                elif param == 'top_p':
                                    params[param] = float(value) if value != 'None' else None
                                else:
                                    params[param] = type(params[param])(value)
                                print(f"Invalid Value for {param}")
                                print(f"Set {param} = {params[param]}")
                            except ValueError:
                                print(f"Invalid value for {param}")
                        else:
                            print(f"Unknown parameter: {param}")
                    else:
                        print("Usage: set <param> <value>")
                    continue
                
                if not prompt:
                    continue
                
                # Generate
                results = self.generate(
                    prompt,
                    max_new_tokens=params['max_new_tokens'],
                    temperature=params['temperature'],
                    top_k=params["top_k"],
                    top_p=params["top_p"]
                )
                
                # Display result
                print("\n" + "-"*60)
                print("Generated:")
                print("-"*60)
                print(results[0])
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
                
                
def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device:str = 'auto'
) -> tuple:
    """
    Load trained model and tokenizer.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to load on
    
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer.vocab)}")
    
    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
    
    # Create config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = LLMConfig(**config_dict)
    else:
        # Fallback to default config
        logger.warning("Config not found in checkpoint, using defaults")
        config = LLMConfig(vocab_size=len(tokenizer.vocab))
    
    # initialize model
    model = AIModel(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_from_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Model loaded successfully")
    
    return model, tokenizer, config

def main(args):
    """Main inference function."""
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(
            args.model_path,
            args.tokenizer_path,
            args.device
        )
        
        # Create inference engine
        engine = SecureInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device = args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu',
            max_length=args.max_length
        )
        
        # Run appropriate mode
        if args.interactive:
            engine.interactive_mode()
        else:
            if not args.prompt:
                logger.error('Prompt required in non-interactive mode')
                sys.exit(1)
            
            # Generate from prompt
            results = engine.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_sequences
            )
            
            # Display results
            for i, text in enumerate(results, 1):
                print(f"\n{'='*60}")
                print(f"Generated Sequence {i}:")
                print(f"{'='*60}")
                print(text)
                print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Secure Text Generation Engine"
    )
    
    # Model and tokenizer paths
    parser.add_argument(
        '--model-path', type=str, default='models/final_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tokenizer-path', type=str, default='tokenizer.json',
        help='Path to tokenizer file'
    )
    
    # Generation parameters
    parser.add_argument(
        '--prompt', type=str,
        help='Input prompt for generation'
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=100,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k', type=int, default=50,
        help='Top-k filtering (0 to disable)'
    )
    parser.add_argument(
        '--top-p', type=float, default=0.95,
        help='Nucleus sampling threshold (0 to disable)'
    )
    parser.add_argument(
        '--num-sequences', type=int, default=1,
        help='Number of sequences to generate'
    )
    
    # System settings
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Maximum total sequence length'
    )
    
    # Mode
    parser.add_argument(
        '--interactive', action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Validate top-k and top-p
    if args.top_k == 0:
        args.top_k = None
    if args.top_p == 0:
        args.top_p = None
    
    main(args)
