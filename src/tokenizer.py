"""
Secure BPE Tokenizer
Security: Input validation, resource limits, safe serialization
Adheres to: CWE-20, CWE-502, CWE-400
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """
    Byte Pair Encoding tokenizer with security hardening.
    - Uses JSON instead of pickle (CWE-502 mitigation)
    - Input validation and sanitization
    - Resource exhaustion prevention
    """
    
    def __init__(self, max_vocab_size: int = 50000):
        """
        Initialize tokenizer with security constraints.
        
        Args:
            max_vocab_size: Maximum vocabulary size (resource limit)
        """
        if max_vocab_size < 256 or max_vocab_size > 100000:
            raise ValueError(f"max_vocab_size must be in [256, 100000]")
        
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[tuple, int] = {}
        self.max_vocab_size = max_vocab_size
        self._is_trained = False
    
    def train(self, text: str, vocab_size: int = 10000, 
              max_text_length: int = 300_000_000) -> 'SimpleTokenizer':
        """
        Train BPE tokenizer with security validation.
        
        Args:
            text: Training text (validated)
            vocab_size: Target vocabulary size
            max_text_length: Maximum allowed text length (DoS prevention)
        
        Returns:
            Self for chaining
        
        Raises:
            ValueError: If inputs violate security constraints
        """
        # Input validation (CWE-20)
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        
        if len(text) > max_text_length:
            raise ValueError(
                f"text length ({len(text)}) exceeds maximum ({max_text_length})"
            )
        
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256")
        
        if vocab_size > self.max_vocab_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) exceeds maximum ({self.max_vocab_size})"
            )
        
        logger.info(f"Training tokenizer on {len(text)} characters...")
        
        # Initialize with byte-level tokens
        try:
            tokens = list(text.encode('utf-8'))
        except UnicodeEncodeError as e:
            raise ValueError(f"Text contains invalid characters: {e}")
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        
        # BPE algorithm with progress tracking
        iteration = 0
        while len(self.vocab) < vocab_size:
            # Safety check: prevent infinite loops
            if iteration > vocab_size * 2:
                logger.warning("BPE training terminated: too many iterations")
                break
            
            # Count adjacent pairs
            pairs = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token
            new_token = len(self.vocab)
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_token
            
            # Replace pairs in tokens (optimized)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
            iteration += 1
            
            # Progress logging
            if iteration % 100 == 0:
                logger.info(f"Vocabulary size: {len(self.vocab)}")
        
        self._is_trained = True
        logger.info(f"Training complete. Final vocab size: {len(self.vocab)}")
        return self
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text into token IDs with validation (OPTIMIZED).
        
        Args:
            text: Input text
            max_length: Maximum sequence length (resource limit)
        
        Returns:
            List of token IDs
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        
        if not text:
            return []
        
        if max_length and len(text) > max_length:
            raise ValueError(f"text length ({len(text)}) exceeds maximum ({max_length})")
        
        import time
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm not available, progress bar disabled")
        
        start_time = time.time()
        
        try:
            tokens = list(text.encode('utf-8'))
        except UnicodeEncodeError as e:
            raise ValueError(f"Text contains invalid characters: {e}")
        
        initial_tokens = len(tokens)
        logger.info(f"Starting BPE encoding: {initial_tokens:,} initial byte tokens")
        
        # Sort merges by their index
        if not hasattr(self, '_sorted_merges'):
            self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        # Progress bar setup
        if use_tqdm:
            pbar = tqdm(
                self._sorted_merges,
                desc="Encoding",
                unit="merge",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            pbar = self._sorted_merges
        
        # Apply each merge
        for merge_idx, (pair, new_token) in enumerate(pbar):
            i = 0
            new_tokens = []
            
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
            
            # Update progress bar description
            if use_tqdm:
                compression = initial_tokens / len(tokens)
                pbar.set_postfix({
                    'tokens': f"{len(tokens):,}",
                    'compression': f"{compression:.2f}x"
                })
            
            # Early exit
            if len(tokens) == 1:
                if use_tqdm:
                    pbar.close()
                break
        
        if use_tqdm and not pbar.disable:
            pbar.close()
        
        total_time = time.time() - start_time
        final_compression = initial_tokens / len(tokens)
        
        logger.info(
            f"✓ Encoding complete: {initial_tokens:,} → {len(tokens):,} tokens "
            f"({final_compression:.2f}x compression) in {total_time:.1f}s"
        )
        
        return tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back into text with validation.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text
        
        Raises:
            ValueError: If token IDs are invalid
        """
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError("token_ids must be a list or tuple")
        
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        # Validate token IDs
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(f"Invalid token ID type: {type(token_id)}")
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
        
        # Decode
        byte_array = bytearray()
        for token_id in token_ids:
            byte_array.extend(self.vocab[token_id])
        
        return byte_array.decode('utf-8', errors='replace')
    
    def save(self, filepath: str) -> None:
        """
        Save tokenizer to JSON (secure alternative to pickle).
        Implements CWE-502 mitigation.
        
        Args:
            filepath: Path to save file
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained tokenizer")
        
        # Convert vocab bytes to lists for JSON serialization
        vocab_serializable = {
            str(k): list(v) for k, v in self.vocab.items()
        }
        
        # Convert merge keys (tuples) to strings
        merges_serializable = {
            f"{k[0]},{k[1]}": v for k, v in self.merges.items()
        }
        
        data = {
            'vocab': vocab_serializable,
            'merges': merges_serializable,
            'max_vocab_size': self.max_vocab_size,
            'version': '1.0'
        }
        
        # Secure file write with error handling
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Tokenizer saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save tokenizer: {e}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SimpleTokenizer':
        """
        Load tokenizer from JSON file.
        Secure alternative to pickle deserialization.
        
        Args:
            filepath: Path to tokenizer file
        
        Returns:
            Loaded tokenizer instance
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load tokenizer: {e}")
        
        # Validate data structure
        required_keys = {'vocab', 'merges', 'max_vocab_size'}
        if not required_keys.issubset(data.keys()):
            raise ValueError("Invalid tokenizer file format")
        
        # Create instance
        tokenizer = cls(max_vocab_size=data['max_vocab_size'])
        
        # Restore vocab
        tokenizer.vocab = {
            int(k): bytes(v) for k, v in data['vocab'].items()
        }
        
        # Restore merges
        tokenizer.merges = {
            tuple(map(int, k.split(','))): v 
            for k, v in data['merges'].items()
        }
        
        tokenizer._is_trained = True
        logger.info(f"Tokenizer loaded from {filepath}")
        return tokenizer