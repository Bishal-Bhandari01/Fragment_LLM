# API Reference

Detailed API documentation for all Fragment_LLM classes and functions.

## `src` Package

### `src.model`

#### `class src.model.AIModel(config: LLMConfig)`
GPT-style transformer model for text generation.

**Parameters:**
- `config` (`LLMConfig`): Configuration object containing model hyperparameters.

**Methods:**

- `forward(idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`
  
  Performs the forward pass of the model.
  - `idx`: Input token indices of shape (Batch, Time).
  - `targets`: Target token indices for loss computation.
  - **Returns**: A tuple of `(logits, loss)`.

- `generate(idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor`
  
  Generates new tokens autoregressively.
  - `idx`: Starting token indices context.
  - `max_new_tokens`: Number of tokens to generate.
  - `temperature`: Sampling temperature.
  - `top_k`: Top-k sampling parameter.
  - `top_p`: Top-p (nucleus) sampling parameter.
  - **Returns**: Tensor containing the original sequence plus generated tokens.

#### `class src.model.TransformerBlock(config: LLMConfig)`
A single Transformer block containing self-attention and MLP layers with pre-normalization.

#### `class src.model.CausalSelfAttention(embed_size, num_heads, block_size, dropout, bias=False)`
Multi-head causal self-attention mechanism.

#### `class src.model.MLP(config: LLMConfig)`
Feed-forward neural network layer.

### `src.tokenizer`

#### `class src.tokenizer.SimpleTokenizer(max_vocab_size: int = 50000)`
Byte Pair Encoding (BPE) tokenizer implementation.

**Methods:**

- `train(text: str, vocab_size: int = 10000, max_text_length: int = 300_000_000) -> 'SimpleTokenizer'`
  
  Trains the tokenizer on the provided text.
  - `text`: Training corpus.
  - `vocab_size`: Desired vocabulary size.
  - `max_text_length`: Safety limit for input text length.

- `encode(text: str, max_length: Optional[int] = None) -> List[int]`
  
  Encodes a string into a list of token IDs.

- `decode(token_ids: List[int]) -> str`
  
  Decodes a list of token IDs back into a string.

- `save(filepath: str) -> None`
  
  Saves the tokenizer vocabulary to a JSON file.

- `load(cls, filepath: str) -> 'SimpleTokenizer'`
  
  Class method to load a tokenizer from a JSON file.

### `src.dataset`

#### `class src.dataset.TextDataset(text_file: str, tokenizer, block_size: int = 512, max_file_size_mb: int = 500, allowed_base_dirs: tuple = ('data/',))`
PyTorch Dataset for loading text data with security checks.

**Parameters:**
- `text_file`: Path to the input text file.
- `tokenizer`: Trained tokenizer instance.
- `block_size`: Length of sequence chunks.
- `max_file_size_mb`: Safety limit for file size.
- `allowed_base_dirs`: Tuple of allowed directory paths for security.

#### `function src.dataset.create_dataloader(file_path: str, tokenizer, batch_size: int = 4, block_size: int = 256, device: str = 'cpu', shuffle: bool = True, drop_last: bool = True, num_workers: int = 0, start_idx: int = 0, max_file_size_mb: int = 500, allowed_base_dirs: tuple = ('data/',))`
Helper function to create a PyTorch DataLoader.

### `src.inference`

#### `class src.inference.SecureInferenceEngine`
Helper class for managing text generation with safety checks.

**Methods:**
- `generate(...)`: specific generation method wrapping model.generate with safety checks.

### `src.config`

#### `class src.config.LLMConfig`
Dataclass for model hyperparameters.

#### `class src.config.SecurityConfig`
Dataclass for security-related settings.

### `src.trainer`

#### `class src.trainer.Trainer`
Handles the training loop, validation, and checkpointing.

**Methods:**
- `train()`: Main training loop.
- `train_epoch(epoch_idx)`: Runs a single epoch of training.
- `validate()`: validation loop.
- `save_checkpoint(filepath)`: Saves training state.

## `scripts` Package

### `scripts.preprocessor`

#### `class scripts.preprocessor.Preprocessor`
Handles data cleaning and splitting for WikiText-style datasets.

**Methods:**
- `preprocess_wkitext()`: Main method to execute the preprocessing pipeline.

---
**Back to**: [Main Documentation](../README.md)
