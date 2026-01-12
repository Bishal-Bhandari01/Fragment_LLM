# Data Preprocessing

Guide to preparing data for Fragment_LLM training.

## Overview

**Location**: `scripts/preprocessor.py`

The preprocessor handles:
- File validation and security checks
- Text cleaning and normalization
- Format conversion (parquet → txt)
- Train/val/test splitting

## Quick Start

```bash
# Place your data in data/raw/
mkdir -p data/raw data/processed

# Run preprocessor
python scripts/preprocessor.py
```

## Supported Input Formats

### Text Files (.txt)

Plain UTF-8 text files:
```
data/raw/train.txt
data/raw/validation.txt
data/raw/test.txt
```

### Parquet Files (.parquet)

WikiText-2 format with 'text' column:
```
data/raw/train-00000-of-00001.parquet
data/raw/validation-00000-of-00001.parquet
```

## Preprocessing Pipeline

1. **File Discovery**: Searches for train/validation/test files
2. **Validation**: Checks file paths and sizes
3. **Loading**: Reads file content
4. **Cleaning**: Removes empty lines, normalizes whitespace
5. **Sanitization**: Removes null bytes
6. **Saving**: Writes to `data/processed/`

## Output Files

```
data/processed/
├── train.txt      # Training data
├── val.txt        # Validation data
└── test.txt       # Test data (if available)
```

## Preprocessing Class

### Initialization

```python
from scripts.preprocessor import Preprocessor

preprocessor = Preprocessor(
    raw_dir='data/raw',
    processed_dir='data/processed',
    max_file_size_mb=500,
    allowed_extensions=('.parquet', '.txt')
)
```

### Run Preprocessing

```python
preprocessor.preprocess_wkitext()
```

## Text Cleaning

### Operations Performed

1. **Strip whitespace**: Remove leading/trailing spaces
2. **Remove section markers**: Skip lines like `========`
3. **Normalize whitespace**: Convert multiple spaces to single
4. **Remove null bytes**: Security sanitization
5. **Filter empty lines**: Remove blank lines

### Example

**Before**:
```
= = = Article Title = = =

This  is   some    text.


More text here.
```

**After**:
```
This is some text.
More text here.
```

## Security Features

### Path Validation

```python
# Validates directory paths
path = Path(dir_path).resolve()

# Checks for traversal attempts
if ".." in str(path):
    raise ValueError("Invalid path")
```

### File Size Limits

```python
max_file_size_mb = 500  # Default

# Prevents loading huge files
if file_size > max_file_size_mb:
    raise ValueError("File too large")
```

### Extension Validation

```python
allowed_extensions = ('.txt', '.parquet')

# Only processes allowed file types
if file.suffix not in allowed_extensions:
    logger.warning(f"Skipping {file.name}")
```

## Custom Data Preparation

### Preparing Your Own Data

1. **Create text file**:
```bash
cat > data/raw/train.txt << EOF
Your training text here.
Multiple lines are fine.
EOF
```

2. **Run preprocessor**:
```bash
python scripts/preprocessor.py
```

### Multiple Files

The preprocessor searches for patterns:
- `train*.txt` or `train*.parquet`
- `validation*.txt` or `validation*.parquet`
- `test*.txt` or `test*.parquet`

### Custom Preprocessing

Extend the `Preprocessor` class:

```python
class CustomPreprocessor(Preprocessor):
    def _sanitize_text(self, text):
        # Custom cleaning
        text = super()._sanitize_text(text)
        text = text.lower()  # Lowercase
        text = remove_urls(text)  # Remove URLs
        return text
```

## Best Practices

1. **Check Data Quality**: Review processed files before training
2. **Appropriate Size**: 1MB+ for meaningful training
3. **Clean Data**: Remove irrelevant content
4. **UTF-8 Encoding**: Ensure proper encoding
5. **Backup Raw Data**: Keep original files

## Troubleshooting

### Issue: "No files found for train split"

**Solution**: Ensure files match naming pattern:
- `train.txt` or `train*.parquet`
- Place in `data/raw/`

### Issue: "File size exceeds limit"

**Solution**: 
- Split large files
- Increase `max_file_size_mb`
- Use streaming approach

### Issue: "Column 'text' not found"

**Solution**: Parquet files must have a 'text' column

## Related Documentation

- [Dataset](dataset.md) - Using processed data
- [Training](training.md) - Training with your data
- [Tokenizer](tokenizer.md) - Tokenization process

---

**Next**: Learn about [configuration options](configuration.md)
