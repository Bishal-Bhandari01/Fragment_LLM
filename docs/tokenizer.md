# Tokenizer Documentation

Byte Pair Encoding (BPE) tokenizer implementation with security features.

## Overview

**Location**: `src/tokenizer.py`

The `SimpleTokenizer` class implements a secure BPE tokenizer that:
- Uses JSON for serialization (no pickle)
- Validates all inputs
- Prevents resource exhaustion
- Provides progress tracking

## Quick Start

### Training a Tokenizer

```python
from src.tokenizer import SimpleTokenizer

# Create tokenizer
tokenizer = SimpleTokenizer(max_vocab_size=10000)

# Train on text
text = "Your training text here..."
tokenizer.train(text, vocab_size=5000)

# Save
tokenizer.save('tokenizer.json')
```

### Loading a Tokenizer

```python
tokenizer = SimpleTokenizer.load('tokenizer.json')
```

### Encoding Text

```python
text = "Hello, world!"
token_ids = tokenizer.encode(text)
# Output: [245, 128, 67, 89, ...]
```

### Decoding Tokens

```python
decoded_text = tokenizer.decode(token_ids)
# Output: "Hello, world!"
```

## BPE Algorithm

### How It Works

1. **Initialize**: Start with byte-level tokens (256 characters)
2. **Count Pairs**: Find most frequent adjacent token pairs
3. **Merge**: Create new token from most frequent pair
4. **Repeat**: Until vocabulary reaches target size

### Example

```
Text: "low lower lowest"

Step 1: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ...]
Step 2: Most frequent pair: ('l', 'o') → merge to 'lo'
Step 3: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ...]
Step 4: Most frequent pair: ('lo', 'w') → merge to 'low'
...
```

## API Reference

### Class: `SimpleTokenizer`

#### `__init__(max_vocab_size=50000)`

Initialize tokenizer.

**Parameters**:
- `max_vocab_size` (int): Maximum vocabulary size [256-100000]

**Example**:
```python
tokenizer = SimpleTokenizer(max_vocab_size=10000)
```

#### `train(text, vocab_size=10000, max_text_length=300_000_000)`

Train tokenizer on text.

**Parameters**:
- `text` (str): Training text
- `vocab_size` (int): Target vocabulary size
- `max_text_length` (int): Maximum text length (DoS prevention)

**Returns**: `self` (for chaining)

**Example**:
```python
with open('data.txt', 'r') as f:
    text = f.read()

tokenizer.train(text, vocab_size=5000)
```

**Progress Output**:
```
Training tokenizer on 1000000 characters...
Vocabulary size: 300
Vocabulary size: 400
...
Training complete. Vocab size: 5000
```

#### `encode(text, max_length=None)`

Encode text to token IDs.

**Parameters**:
- `text` (str): Input text
- `max_length` (int, optional): Maximum sequence length

**Returns**: `List[int]` - Token IDs

**Example**:
```python
tokens = tokenizer.encode("Hello world")
# [245, 128, 67]
```

**With Progress**:
```
Starting BPE encoding: 1,000 initial byte tokens
Encoding: 100%|████████| 4743/4743
✓ Encoding complete: 1,000 → 150 tokens (6.67x compression)
```

#### `decode(token_ids)`

Decode token IDs to text.

**Parameters**:
- `token_ids` (List[int]): Token IDs to decode

**Returns**: `str` - Decoded text

**Example**:
```python
text = tokenizer.decode([245, 128, 67])
# "Hello world"
```

#### `save(filepath)`

Save tokenizer to JSON file.

**Parameters**:
- `filepath` (str): Path to save file

**Example**:
```python
tokenizer.save('my_tokenizer.json')
```

**File Format**:
```json
{
  "vocab": {
    "0": [72],
    "1": [101],
    ...
  },
  "merges": {
    "0,1": 256,
    ...
  },
  "max_vocab_size": 10000,
  "version": "1.0"
}
```

#### `load(filepath)` (classmethod)

Load tokenizer from JSON file.

**Parameters**:
- `filepath` (str): Path to tokenizer file

**Returns**: `SimpleTokenizer` instance

**Example**:
```python
tokenizer = SimpleTokenizer.load('my_tokenizer.json')
```

## Security Features

### 1. No Pickle Usage

Uses JSON instead of pickle to prevent arbitrary code execution (CWE-502).

```python
# ✅ Safe
tokenizer.save('tokenizer.json')  # JSON serialization

# ❌ Unsafe (not used)
pickle.dump(tokenizer, f)  # Could execute arbitrary code
```

### 2. Input Validation

All inputs are validated:

```python
# Text validation
if not isinstance(text, str):
    raise TypeError("text must be a string")

# Length validation
if len(text) > max_text_length:
    raise ValueError(f"text length exceeds maximum")

# Vocab size validation
if vocab_size < 256 or vocab_size > max_vocab_size:
    raise ValueError(f"vocab_size must be in [256, {max_vocab_size}]")
```

### 3. Resource Limits

Prevents resource exhaustion:

```python
# Maximum text length
max_text_length = 300_000_000  # 300MB

# Maximum vocabulary size
max_vocab_size = 100_000

# Iteration limit (prevents infinite loops)
if iteration > vocab_size * 2:
    logger.warning("BPE training terminated: too many iterations")
    break
```

## Performance

### Training Speed

| Text Size | Vocab Size | Time |
|-----------|-----------|------|
| 1MB | 5,000 | ~5s |
| 10MB | 10,000 | ~30s |
| 100MB | 50,000 | ~5min |

### Encoding Speed

| Text Size | Tokens | Time |
|-----------|--------|------|
| 1KB | ~150 | <1s |
| 1MB | ~150K | ~10s |
| 10MB | ~1.5M | ~2min |

### Compression Ratio

Typical compression: **5-7x**

```
Original bytes: 1,000
Encoded tokens: 150
Compression: 6.67x
```

## Best Practices

1. **Train Once**: Train tokenizer on representative data, reuse for all models
2. **Appropriate Vocab Size**: 
   - Small datasets: 5,000-10,000
   - Large datasets: 30,000-50,000
3. **Save Tokenizer**: Always save after training
4. **Version Control**: Track tokenizer files with your model

## Common Issues

### Issue: Slow Encoding

**Solution**: Encoding is O(n×m) where n=text length, m=merges. For very long texts, consider chunking.

### Issue: Poor Compression

**Solution**: Train on more representative data or increase vocab size.

### Issue: Unicode Errors

**Solution**: Tokenizer handles UTF-8. Use `errors='replace'` in decode if needed.

## Advanced Usage

### Custom Vocabulary Size

```python
# Small model
tokenizer.train(text, vocab_size=5000)

# Large model
tokenizer.train(text, vocab_size=50000)
```

### Inspect Vocabulary

```python
# View vocabulary
print(f"Vocab size: {len(tokenizer.vocab)}")

# View merges
print(f"Number of merges: {len(tokenizer.merges)}")

# View specific token
token_id = 256
token_bytes = tokenizer.vocab[token_id]
print(f"Token {token_id}: {token_bytes}")
```

## Related Documentation

- [Dataset](dataset.md) - How tokenizer integrates with dataset
- [Training](training.md) - Using tokenizer in training
- [Configuration](configuration.md) - Tokenizer configuration

---

**Next**: Learn about [dataset handling](dataset.md)
