# Security Rules & Guidelines

## Overview
This document outlines the comprehensive security measures implemented in this LLM training codebase, aligned with OWASP ASVS 4.0, CWE Top 25, and CIS Benchmarks.

---

## 1. Input Validation (CWE-20)

### Rule 1.1: All User Inputs Must Be Validated
**Severity: CRITICAL**

All inputs from external sources MUST be validated before processing:

```python
# ✅ CORRECT
if not 1 <= batch_size <= 256:
    raise ValueError(f"batch_size must be in [1, 256]")

# ❌ INCORRECT
model.train(batch_size=user_input)  # No validation
```

**Implementation:**
- Model configuration: Range validation in `__post_init__`
- Text inputs: Length and character validation
- File paths: Path traversal prevention
- Numeric parameters: Range and type checking

---

## 2. Path Traversal Prevention (CWE-22)

### Rule 2.1: Never Trust User-Provided File Paths
**Severity: CRITICAL**

All file paths MUST be validated to prevent directory traversal attacks:

```python
# ✅ CORRECT
def _validate_path(filepath: str, allowed_base_dirs: tuple) -> Path:
    path = Path(filepath).resolve()
    if '..' in str(filepath):
        raise ValueError("Path traversal detected")
    # Verify within allowed directories
    ...

# ❌ INCORRECT
with open(user_provided_path, 'r') as f:  # Direct file access
    data = f.read()
```

**Protected Operations:**
- Dataset loading
- Model checkpoint saving/loading
- Tokenizer serialization
- Log file access

---

## 3. Deserialization Security (CWE-502)

### Rule 3.1: Never Use Pickle for Untrusted Data
**Severity: CRITICAL**

Use JSON for serialization instead of pickle to prevent arbitrary code execution:

```python
# ✅ CORRECT - Use JSON
tokenizer.save('tokenizer.json')  # Safe JSON serialization

# ❌ INCORRECT - Pickle vulnerability
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)  # Can execute arbitrary code
```

**Mitigations:**
- Tokenizer uses JSON format only
- Model checkpoints use `torch.save` with `weights_only=True`
- No pickle usage anywhere in codebase

---

## 4. Resource Exhaustion Prevention (CWE-400)

### Rule 4.1: Enforce Resource Limits
**Severity: HIGH**

All resource-intensive operations MUST have limits:

```python
# ✅ CORRECT
if file_size_mb > self.max_file_size_mb:
    raise ValueError(f"File too large: {file_size_mb}MB")

if len(text) > max_text_length:
    raise ValueError(f"Text too long: {len(text)}")

# ❌ INCORRECT
text = input_file.read()  # No size limit
tokens = tokenizer.encode(unlimited_text)  # No limit
```

**Resource Limits:**
- File sizes: 500MB default maximum
- Text lengths: 100MB for training, 10MB for inference
- Vocabulary size: 100,000 maximum
- Sequence length: 2,048 maximum
- Generation tokens: 512 maximum

---

## 5. Memory Safety for Low-End Systems

### Rule 5.1: Optimize for Limited Resources
**Severity: MEDIUM**

Code MUST be efficient for systems with limited memory:

**Optimizations Implemented:**
- Reduced model size: 6 layers (vs 12), 384 embedding (vs 768)
- Gradient accumulation: Effective batch size 64 with actual batch 16
- Mixed precision training: FP16 to reduce memory usage
- Efficient attention: Combined QKV projection
- Streaming data loading: No full dataset in memory

```python
# Memory-efficient configuration for low-end PCs
config = LLMConfig(
    n_layer=6,           # Reduced layers
    n_embd=384,          # Smaller embeddings
    batch_size=16,       # Small batch
    gradient_accumulation_steps=4,  # Effective batch: 64
    use_amp=True         # Mixed precision
)
```

---

## 6. Secure Model Checkpointing

### Rule 6.1: Atomic File Operations
**Severity: MEDIUM**

Checkpoint saves MUST be atomic to prevent corruption:

```python
# ✅ CORRECT - Atomic write
temp_path = path.with_suffix('.tmp')
torch.save(checkpoint, temp_path)
temp_path.replace(path)  # Atomic operation

# ❌ INCORRECT - Can corrupt on failure
torch.save(checkpoint, path)  # Direct write
```

### Rule 6.2: Use `weights_only=True` for Loading
**Severity: CRITICAL**

When loading checkpoints, ALWAYS use `weights_only=True`:

```python
# ✅ CORRECT
checkpoint = torch.load(path, weights_only=True)

# ❌ INCORRECT - Allows arbitrary code execution
checkpoint = torch.load(path)
```

---

## 7. Safe Text Generation

### Rule 7.1: Clamp Generation Parameters
**Severity: MEDIUM**

All generation parameters MUST be within safe ranges:

```python
# ✅ CORRECT - Parameter clamping
temperature = max(0.1, min(temperature, 2.0))
max_new_tokens = min(max_new_tokens, MAX_GENERATION_TOKENS)

# ❌ INCORRECT - Unbounded parameters
model.generate(temperature=user_input)  # Could be 1000
```

**Safe Ranges:**
- Temperature: [0.1, 2.0]
- Top-k: [1, vocab_size]
- Top-p: (0.0, 1.0]
- Max tokens: [1, 512]

---

## 8. Logging and Monitoring

### Rule 8.1: Log Security-Relevant Events
**Severity: LOW**

Security events MUST be logged:

```python
logger.warning("Path traversal detected: {path}")
logger.error("File size exceeds limit: {size}MB")
logger.info("Model loaded successfully")
```

**Events to Log:**
- Path validation failures
- Resource limit violations
- Authentication attempts (if added)
- Model loading/saving
- Generation requests

---

## 9. Error Handling

### Rule 9.1: Never Expose Internal Details
**Severity: MEDIUM**

Error messages MUST NOT reveal internal implementation:

```python
# ✅ CORRECT
except Exception as e:
    logger.error(f"Processing failed: {e}")
    raise ValueError("Failed to process input")

# ❌ INCORRECT - Exposes internal paths
raise ValueError(f"File not found: /home/user/secret/path.txt")
```

---

## 10. Code Execution Prevention

### Rule 10.1: No Dynamic Code Execution
**Severity: CRITICAL**

NEVER use `eval()`, `exec()`, or `__import__()` on user input:

```python
# ❌ FORBIDDEN
eval(user_input)
exec(user_code)
__import__(user_module)
```

**This codebase:** No dynamic code execution anywhere.

---

## Security Checklist

Before deploying or modifying code, verify:

- [ ] All inputs are validated with range checks
- [ ] File paths are validated against directory traversal
- [ ] No pickle deserialization used
- [ ] Resource limits enforced on all operations
- [ ] Checkpoint operations are atomic
- [ ] `weights_only=True` used for torch.load
- [ ] Generation parameters are clamped
- [ ] Security events are logged
- [ ] Error messages don't expose internals
- [ ] No dynamic code execution

---

## Vulnerability Mapping

| CWE ID | Vulnerability | Mitigation |
|--------|--------------|------------|
| CWE-20 | Improper Input Validation | All inputs validated in constructors |
| CWE-22 | Path Traversal | Path validation in dataset/preprocessor |
| CWE-400 | Resource Exhaustion | File size, text length limits |
| CWE-502 | Deserialization | JSON only, no pickle |
| CWE-78 | OS Command Injection | No shell commands used |
| CWE-89 | SQL Injection | No database usage |
| CWE-94 | Code Injection | No eval/exec usage |

---

## OWASP ASVS 4.0 Compliance

| Category | Level | Status |
|----------|-------|--------|
| V1: Architecture | L1 | ✅ Compliant |
| V5: Validation | L1 | ✅ Compliant |
| V8: Data Protection | L1 | ✅ Compliant |
| V10: Malicious Code | L1 | ✅ Compliant |
| V12: Files and Resources | L1 | ✅ Compliant |

---

## CIS Benchmark Alignment

- **CIS Control 3**: Data Protection - Encrypted checkpoints supported
- **CIS Control 8**: Audit Logs - Comprehensive logging implemented
- **CIS Control 16**: Application Security - Secure coding practices
- **CIS Control 18**: Penetration Testing - Security review completed

---

## Incident Response

If a security issue is discovered:

1. **DO NOT** deploy the code
2. Log the issue with details
3. Review the security rules violated
4. Implement mitigation following this document
5. Test the fix thoroughly
6. Document the incident

---

## Updates and Maintenance

This security document MUST be updated when:
- New features are added
- New vulnerabilities are discovered
- Security standards are updated
- Code architecture changes

**Last Updated:** 2024  
**Review Frequency:** Quarterly  
**Next Review:** Q1 2025