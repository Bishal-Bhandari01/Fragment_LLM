# Dataset Documentation

Secure text dataset implementation with path validation and memory efficiency.

## Overview

**Location**: `src/dataset.py`

The `TextDataset` class provides:
- Secure file loading with path validation
- Memory-efficient data handling
- Automatic tokenization
- Input validation

## Quick Start

```python
from src.dataset import TextDataset, create_dataloader
from src.tokenizer import SimpleTokenizer

# Load tokenizer
tokenizer = SimpleTokenizer.load('tokenizer.json')

# Create dataset
dataset = TextDataset(
    text_file='data/processed/train.txt',
    tokenizer=tokenizer,
    block_size=512
)

# Create dataloader
dataloader = create_dataloader(
    'data/processed/train.txt',
    tokenizer,
    batch_size=16,
    block_size=512
)
```

## TextDataset Class

### Initialization

```python
dataset = TextDataset(
    text_file='data/processed/train.txt',
    tokenizer=tokenizer,
    block_size=512,
    max_file_size_mb=500,
    allowed_base_dirs=('data/',)
)
```

**Parameters**:
- `text_file`: Path to text file
- `tokenizer`: Trained tokenizer instance
- `block_size`: Sequence length [1-2048]
- `max_file_size_mb`: Max file size (DoS prevention)
- `allowed_base_dirs`: Allowed directory prefixes

### How It Works

1. **Path Validation**: Checks file path for security
2. **File Size Check**: Prevents loading huge files
3. **Text Loading**: Reads file content
4. **Tokenization**: Converts text to token IDs
5. **Sequence Creation**: Creates overlapping sequences

### Data Format

```python
# Input text file
"Hello world! This is a test."

# After tokenization
tokens = [245, 128, 67, 89, 12, 45, ...]

# Dataset returns (input, target) pairs
dataset[0] = (
    [245, 128, 67, 89],  # input
    [128, 67, 89, 12]    # target (shifted by 1)
)
```

## Security Features

### Path Validation

Prevents directory traversal attacks (CWE-22):

```python
# ✅ Safe paths
'data/processed/train.txt'
'./data/train.txt'

# ❌ Blocked paths
'../../../etc/passwd'
'/etc/passwd'
'data/../../../secrets.txt'
```

### File Size Limits

Prevents resource exhaustion (CWE-400):

```python
max_file_size_mb = 500  # Default limit

# Files larger than 500MB are rejected
if file_size_mb > max_file_size_mb:
    raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds limit")
```

### Input Validation

All parameters are validated:

```python
# Block size validation
if not 1 <= block_size <= 2048:
    raise ValueError(f"block_size must be in [1, 2048]")

# File existence check
if not path.exists():
    raise FileNotFoundError(f"File not found: {filepath}")
```

## DataLoader Creation

### Basic Usage

```python
dataloader = create_dataloader(
    file_path='data/processed/train.txt',
    tokenizer=tokenizer,
    batch_size=16,
    block_size=512
)
```

### Advanced Options

```python
dataloader = create_dataloader(
    file_path='data/processed/train.txt',
    tokenizer=tokenizer,
    batch_size=32,
    block_size=1024,
    num_workers=0,  # 0 for low-end PCs
    max_file_size_mb=1000,
    allowed_base_dirs=('data/', 'custom_data/')
)
```

**Parameters**:
- `num_workers`: Data loading workers (0 for single-threaded)
- `shuffle`: Shuffle data (default: True)
- `drop_last`: Drop incomplete batches (default: True)
- `pin_memory`: Pin memory for GPU (auto-detected)

## Memory Efficiency

### Streaming vs Loading

The dataset loads the entire file into memory after tokenization, but uses efficient storage:

```python
# Text file: 10MB
# Tokenized: ~1.5M tokens × 4 bytes = 6MB
# Much smaller than raw text!
```

### For Very Large Datasets

If your dataset is too large for memory, consider:

1. **Split into chunks**:
```bash
split -l 100000 large_file.txt chunk_
```

2. **Train on chunks sequentially**:
```python
for chunk in ['chunk_aa', 'chunk_ab', ...]:
    dataloader = create_dataloader(chunk, ...)
    train_one_epoch(dataloader)
```

## Supported File Formats

### Text Files (.txt)

```
Plain text, UTF-8 encoded
One document or continuous text
```

### Parquet Files (.parquet)

Supported via preprocessing script. See [Preprocessing](preprocessing.md).

## Custom Datasets

### Extending TextDataset

```python
class CustomDataset(TextDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # Custom processing
        return x, y
```

### Custom Data Loading

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size):
        # Your implementation
        pass
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return (input, target) tensors
        return x, y
```

## Best Practices

1. **Validate Paths**: Always use `allowed_base_dirs`
2. **Check File Sizes**: Set appropriate `max_file_size_mb`
3. **Use Appropriate Block Size**: Match your model's `block_size`
4. **Monitor Memory**: Watch RAM usage during data loading

## Troubleshooting

### Issue: "File not found"

**Solution**: Check file path and ensure preprocessing completed

### Issue: "Dataset too small"

**Solution**: Need at least `block_size + 1` tokens. Use more data or smaller block size.

### Issue: "Out of memory during data loading"

**Solution**: 
- Reduce file size
- Split into smaller files
- Increase `max_file_size_mb` limit carefully

## Related Documentation

- [Tokenizer](tokenizer.md) - Tokenization details
- [Training](training.md) - Using datasets in training
- [Preprocessing](preprocessing.md) - Data preparation

---

**Next**: Learn about [text generation](inference.md)
