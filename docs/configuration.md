# Configuration Reference

Complete reference for Fragment_LLM configuration options.

## LLMConfig

**Location**: `src/config.py`

Main configuration class for model and training.

### Model Architecture Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `vocab_size` | int | 5000 | [256, 100000] | Vocabulary size |
| `block_size` | int | 256 | [128, 2048] | Context length (sequence length) |
| `n_layer` | int | 4 | [1, 48] | Number of transformer layers |
| `n_head` | int | 4 | [1, 32] | Number of attention heads |
| `n_embd` | int | 256 | [64, 2048] | Embedding dimension |
| `dropout` | float | 0.1 | [0.0, 0.9] | Dropout rate |
| `bias` | bool | True | - | Use bias in linear layers |

### Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `batch_size` | int | 16 | [1, 256] | Batch size |
| `gradient_accumulation_steps` | int | 4 | - | Gradient accumulation steps |
| `learning_rate` | float | 3e-4 | (0, 1e-2] | Learning rate |
| `max_iters` | int | 10000 | - | Maximum iterations |
| `weight_decay` | float | 1e-1 | [0, 1] | Weight decay (L2 regularization) |
| `grad_norm_clip` | float | 1.0 | (0, 10] | Gradient clipping threshold |
| `use_amp` | bool | True | - | Use mixed precision training |

### Security Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_file_size_mb` | int | 500 | Maximum file size |
| `max_sequence_length` | int | 2048 | Maximum sequence length |
| `allowed_file_extension` | tuple | ('.txt', '.parquet') | Allowed file types |
| `max_generation_tokens` | int | 512 | Max tokens to generate |
| `temperature_min` | float | 0.1 | Minimum temperature |
| `temperature_max` | float | 2.0 | Maximum temperature |

## Configuration Examples

### Tiny Model (Low-End PC)

```python
from src.config import LLMConfig

config = LLMConfig(
    vocab_size=5000,
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
    batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    use_amp=True
)
```

**Memory**: ~2GB GPU / 1GB CPU  
**Parameters**: ~15M  
**Use case**: Testing, low resources

### Small Model (Default)

```python
config = LLMConfig(
    vocab_size=5000,
    block_size=512,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    use_amp=True
)
```

**Memory**: ~4GB GPU / 2GB CPU  
**Parameters**: ~38M  
**Use case**: General purpose

### Medium Model (High-End PC)

```python
config = LLMConfig(
    vocab_size=10000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    use_amp=True
)
```

**Memory**: ~8GB GPU / 4GB CPU  
**Parameters**: ~124M  
**Use case**: Best quality

## SecurityConfig

**Location**: `src/config.py`

Security configuration for file operations and validation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_absolute_path` | bool | False | Allow absolute file paths |
| `allowed_base_dirs` | tuple | ('data/', 'models/', 'checkpoints/') | Allowed directories |
| `max_input_length` | int | 10000000 | Max input text length (10MB) |
| `sanitization_inputs` | bool | True | Enable input sanitization |
| `max_requests_per_minute` | int | 60 | Rate limiting |
| `enable_content_filter` | bool | True | Enable content filtering |
| `block_harmful_generation` | bool | True | Block harmful outputs |
| `enable_audit_logging` | bool | True | Enable audit logs |
| `log_level` | str | "INFO" | Logging level |

### Usage

```python
from src.config import SecurityConfig

security = SecurityConfig(
    allowed_base_dirs=('data/', 'custom_data/'),
    max_input_length=50000000,  # 50MB
    enable_audit_logging=True
)

# Validate file path
is_valid = security.validate_file_path('data/train.txt')
```

## Validation Rules

All parameters are automatically validated:

```python
config = LLMConfig(
    vocab_size=50,  # ❌ Too small
    n_layer=100,    # ❌ Too many layers
    dropout=1.5     # ❌ Out of range
)
# Raises ValueError with detailed message
```

## Parameter Relationships

### n_embd must be divisible by n_head

```python
# ✅ Valid
config = LLMConfig(n_embd=384, n_head=6)  # 384/6 = 64

# ❌ Invalid
config = LLMConfig(n_embd=384, n_head=5)  # 384/5 = 76.8
```

### Effective Batch Size

```
effective_batch_size = batch_size × gradient_accumulation_steps
```

Example:
```python
config = LLMConfig(
    batch_size=16,
    gradient_accumulation_steps=4
)
# Effective batch size = 64
```

## Performance Tuning

### For Speed

```python
config = LLMConfig(
    n_layer=4,           # Fewer layers
    n_embd=256,          # Smaller embedding
    batch_size=32,       # Larger batches
    use_amp=True         # Mixed precision
)
```

### For Quality

```python
config = LLMConfig(
    n_layer=12,          # More layers
    n_embd=768,          # Larger embedding
    block_size=1024,     # Longer context
    dropout=0.1          # Regularization
)
```

### For Memory Efficiency

```python
config = LLMConfig(
    batch_size=8,        # Smaller batches
    gradient_accumulation_steps=8,  # Accumulate gradients
    n_layer=6,           # Moderate size
    use_amp=True         # FP16
)
```

## Related Documentation

- [Model Architecture](model.md) - How parameters affect model
- [Training](training.md) - Using configuration in training
- [Security](security.md) - Security features

---

**Next**: Learn about [security features](security.md)
