# Security Features

Comprehensive security documentation for Fragment_LLM.

## Security Overview

Fragment_LLM implements defense-in-depth security following:
- **OWASP ASVS 4.0** - Application Security Verification Standard
- **CWE Top 25** - Common Weakness Enumeration mitigations
- **CIS Benchmarks** - Security best practices

## Key Security Features

### 1. No Pickle Usage (CWE-502)

**Threat**: Arbitrary code execution via malicious pickle files

**Mitigation**: JSON-based serialization

```python
# ✅ Safe - JSON serialization
tokenizer.save('tokenizer.json')
tokenizer = SimpleTokenizer.load('tokenizer.json')

# ❌ Unsafe - Pickle (NOT USED)
import pickle
pickle.dump(tokenizer, f)  # Could execute arbitrary code
```

### 2. Path Traversal Prevention (CWE-22)

**Threat**: Access to unauthorized files via path traversal

**Mitigation**: Path validation and sanitization

```python
# ✅ Safe paths
'data/processed/train.txt'
'./data/train.txt'

# ❌ Blocked paths
'../../../etc/passwd'
'/etc/passwd'
'data/../../../secrets.txt'
```

**Implementation**:
```python
def _validate_path(filepath, allowed_base_dirs):
    path = Path(filepath).resolve()
    
    # Block path traversal
    if ".." in str(filepath):
        raise ValueError("Path traversal detected")
    
    # Verify within allowed directories
    if not any(path_str.startswith(base) for base in allowed_base_dirs):
        raise ValueError("Path not in allowed directories")
```

### 3. Resource Exhaustion Prevention (CWE-400)

**Threat**: Denial of service via resource exhaustion

**Mitigation**: Limits on all resources

```python
# File size limits
max_file_size_mb = 500

# Sequence length limits
max_sequence_length = 2048

# Generation limits
max_generation_tokens = 512

# Text length limits
max_text_length = 300_000_000  # 300MB
```

### 4. Input Validation (CWE-20)

**Threat**: Improper input causing crashes or exploits

**Mitigation**: Comprehensive validation

```python
# All parameters validated
config = LLMConfig(
    vocab_size=10000,  # Must be in [256, 100000]
    batch_size=16,     # Must be in [1, 256]
    n_layer=6          # Must be in [1, 48]
)
# Raises ValueError if invalid
```

## Security by Component

### Tokenizer Security

**File**: `src/tokenizer.py`

1. **JSON Serialization**: No pickle
2. **Input Validation**: Type and length checks
3. **Resource Limits**: Max vocab size, max text length
4. **Safe Deserialization**: Validates loaded data

```python
# Validation in load()
required_keys = {'vocab', 'merges', 'max_vocab_size'}
if not required_keys.issubset(data.keys()):
    raise ValueError("Invalid tokenizer file format")
```

### Dataset Security

**File**: `src/dataset.py`

1. **Path Validation**: Prevents directory traversal
2. **File Size Limits**: Prevents DoS
3. **Input Sanitization**: Removes null bytes
4. **Type Validation**: Ensures correct data types

```python
# Path validation
self.text_file = self._validate_path(text_file, allowed_base_dirs)

# Size check
if file_size_mb > max_file_size_mb:
    raise ValueError(f"File size exceeds limit")

# Sanitization
text = text.replace('\x00', '')
```

### Model Security

**File**: `src/model.py`

1. **Input Validation**: Checks tensor dimensions
2. **Range Checks**: Validates token indices
3. **Temperature Clamping**: Prevents extreme values
4. **Generation Limits**: Max tokens enforced

```python
# Input validation
if T > self.config.block_size:
    raise ValueError(f"Sequence length {T} exceeds block_size")

if idx.min() < 0 or idx.max() >= self.config.vocab_size:
    raise ValueError("Token indices out of range")

# Temperature clamping
temperature = max(temp_min, min(temperature, temp_max))
```

### Inference Security

**File**: `src/inference.py`

1. **Prompt Validation**: Length limits, sanitization
2. **Parameter Validation**: Safe ranges
3. **Resource Limits**: Max generation tokens
4. **Input Sanitization**: Removes dangerous characters

```python
# Prompt validation
def _validate_prompt(self, prompt):
    # Remove null bytes
    prompt = prompt.replace('\x00', '')
    
    # Length limit
    if len(prompt) > 10000:
        prompt = prompt[:10000]
    
    # Non-empty check
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
```

## Security Best Practices

### 1. Always Validate Paths

```python
# ✅ Good
dataset = TextDataset(
    'data/processed/train.txt',
    tokenizer,
    allowed_base_dirs=('data/',)
)

# ❌ Bad
dataset = TextDataset(user_input_path, tokenizer)
```

### 2. Use Provided Limits

```python
# ✅ Good - respects limits
config = LLMConfig(
    max_file_size_mb=500,
    max_sequence_length=2048
)

# ❌ Bad - disabling limits
config.max_file_size_mb = float('inf')
```

### 3. Never Disable Validation

```python
# ✅ Good - validation enabled
config = LLMConfig(vocab_size=10000)  # Validated

# ❌ Bad - bypassing validation
config.vocab_size = 999999999  # Direct assignment
```

### 4. Use JSON, Not Pickle

```python
# ✅ Good
tokenizer.save('tokenizer.json')

# ❌ Bad
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
```

## Threat Model

### Threats Mitigated

| Threat | CWE | Mitigation |
|--------|-----|------------|
| Arbitrary code execution | CWE-502 | JSON serialization |
| Path traversal | CWE-22 | Path validation |
| Resource exhaustion | CWE-400 | Resource limits |
| Improper input validation | CWE-20 | Comprehensive validation |
| Injection attacks | CWE-74 | Input sanitization |

### Threats Not Addressed

- **Network security**: No network code
- **Authentication**: Not applicable (local tool)
- **Encryption**: Data not encrypted at rest
- **Side-channel attacks**: Not mitigated

## Audit Logging

Security-relevant events are logged:

```python
logger.info(f"Validated: {self.text_file}")
logger.warning("Prompt too long, truncating")
logger.error(f"Failed to load tokenizer: {e}")
```

## Security Testing

### Manual Testing

1. **Path Traversal**:
```python
# Should raise ValueError
dataset = TextDataset('../../../etc/passwd', tokenizer)
```

2. **Large Files**:
```python
# Should raise ValueError
dataset = TextDataset('huge_file.txt', tokenizer, max_file_size_mb=1)
```

3. **Invalid Parameters**:
```python
# Should raise ValueError
config = LLMConfig(vocab_size=999999999)
```

## Reporting Security Issues

If you find a security vulnerability:

1. **Do not** open a public issue
2. Contact maintainers privately
3. Provide detailed reproduction steps
4. Allow time for patch before disclosure

## Related Documentation

- [Configuration](configuration.md) - Security configuration
- [Dataset](dataset.md) - Dataset security features
- [Tokenizer](tokenizer.md) - Tokenizer security

---

**Next**: Learn about [system architecture](architecture.md)
