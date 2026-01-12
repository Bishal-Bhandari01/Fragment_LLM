# Inference Guide

Complete guide to text generation with Fragment_LLM.

## Overview

**Location**: `src/inference.py`

The inference engine provides:
- Interactive text generation
- Batch generation
- Multiple sampling strategies
- Safety constraints

## Quick Start

### Interactive Mode

```bash
python src/inference.py --interactive
```

```
Prompt: Once upon a time
Generated: Once upon a time, there was a small village nestled in the mountains...

Prompt: The future of AI
Generated: The future of AI holds tremendous potential for transforming...
```

### Single Generation

```bash
python src/inference.py \
    --prompt "Hello world" \
    --max-new-tokens 100 \
    --temperature 0.8
```

## Command Line Options

```bash
--model-path PATH          # Model checkpoint (default: models/final_model.pt)
--tokenizer-path PATH      # Tokenizer file (default: tokenizer.json)
--prompt TEXT              # Input prompt
--max-new-tokens INT       # Tokens to generate (default: 100)
--temperature FLOAT        # Sampling temperature (default: 1.0)
--top-k INT                # Top-k filtering (default: 50)
--top-p FLOAT              # Nucleus sampling (default: 0.95)
--num-sequences INT        # Number of sequences (default: 1)
--device {auto,cuda,cpu}   # Device to use (default: auto)
--max-length INT           # Max sequence length (default: 512)
--interactive              # Interactive mode
```

## Generation Parameters

### Temperature

Controls randomness:

```bash
# Low temperature (0.1-0.5): More focused, deterministic
python src/inference.py --prompt "The capital of France" --temperature 0.3

# Medium temperature (0.7-1.0): Balanced
python src/inference.py --prompt "Write a story" --temperature 0.8

# High temperature (1.0-2.0): More creative, random
python src/inference.py --prompt "Imagine" --temperature 1.5
```

**Effect**:
- Lower = more likely tokens chosen
- Higher = more diverse outputs

### Top-K Sampling

Limits to K most likely tokens:

```bash
# Restrictive (top-10)
python src/inference.py --prompt "Hello" --top-k 10

# Balanced (top-50, default)
python src/inference.py --prompt "Hello" --top-k 50

# Permissive (top-100)
python src/inference.py --prompt "Hello" --top-k 100
```

### Top-P (Nucleus) Sampling

Limits to tokens with cumulative probability P:

```bash
# Conservative (p=0.9)
python src/inference.py --prompt "Hello" --top-p 0.9

# Balanced (p=0.95, default)
python src/inference.py --prompt "Hello" --top-p 0.95

# Diverse (p=0.99)
python src/inference.py --prompt "Hello" --top-p 0.99
```

### Combining Parameters

```bash
python src/inference.py \
    --prompt "Write a poem about" \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 0.95 \
    --max-new-tokens 200
```

## Interactive Mode

### Basic Usage

```bash
python src/inference.py --interactive
```

### Commands

```
Prompt: your text here          # Generate text
help                             # Show help
set <param> <value>              # Change parameter
quit / exit / q                  # Exit
```

### Setting Parameters

```
Prompt: set temperature 0.7
Set temperature = 0.7

Prompt: set max_new_tokens 150
Set max_new_tokens = 150

Prompt: set top_k 30
Set top_k = 30
```

### Example Session

```
Prompt: Once upon a time
Generated: Once upon a time, there was a brave knight...

Prompt: set temperature 1.2
Set temperature = 1.2

Prompt: In a galaxy far away
Generated: In a galaxy far away, ancient civilizations thrived...

Prompt: quit
Exiting interactive mode
```

## Programmatic Usage

### Basic Generation

```python
from src.inference import SecureInferenceEngine, load_model_and_tokenizer

# Load model and tokenizer
model, tokenizer, config = load_model_and_tokenizer(
    model_path='models/final_model.pt',
    tokenizer_path='tokenizer.json',
    device='cuda'
)

# Create inference engine
engine = SecureInferenceEngine(
    model=model,
    tokenizer=tokenizer,
    device='cuda',
    max_length=512
)

# Generate text
results = engine.generate(
    prompt="Hello world",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

print(results[0])
```

### Batch Generation

```python
# Generate multiple sequences
results = engine.generate(
    prompt="Once upon a time",
    max_new_tokens=200,
    temperature=0.9,
    num_return_sequences=5
)

for i, text in enumerate(results, 1):
    print(f"\n=== Sequence {i} ===")
    print(text)
```

## Safety Features

### Input Validation

```python
# Prompt validation
- Removes null bytes
- Truncates if too long (>10,000 chars)
- Rejects empty prompts

# Parameter validation
- Temperature clamped to [0.1, 2.0]
- max_new_tokens limited by config
- top_k/top_p validated
```

### Resource Limits

```python
# Maximum generation length
max_generation_tokens = 512  # From config

# Automatic clamping
if max_new_tokens > max_generation_tokens:
    max_new_tokens = max_generation_tokens
```

## Performance Tips

### 1. Use GPU

```bash
python src/inference.py --device cuda --interactive
```

**Speed**: ~10-50x faster than CPU

### 2. Reduce Max Tokens

```bash
python src/inference.py --max-new-tokens 50
```

**Speed**: Proportional to tokens generated

### 3. Batch Generation

Generate multiple sequences at once:

```bash
python src/inference.py \
    --prompt "Hello" \
    --num-sequences 5 \
    --max-new-tokens 100
```

### 4. Use Appropriate Temperature

Lower temperature = faster (fewer calculations):

```bash
python src/inference.py --temperature 0.5
```

## Common Use Cases

### Story Generation

```bash
python src/inference.py \
    --prompt "In a world where magic exists" \
    --max-new-tokens 300 \
    --temperature 0.9 \
    --top-p 0.95
```

### Code Completion

```bash
python src/inference.py \
    --prompt "def fibonacci(n):" \
    --max-new-tokens 150 \
    --temperature 0.3 \
    --top-k 20
```

### Question Answering

```bash
python src/inference.py \
    --prompt "Q: What is the capital of France?\nA:" \
    --max-new-tokens 50 \
    --temperature 0.2
```

### Creative Writing

```bash
python src/inference.py \
    --prompt "Write a haiku about" \
    --max-new-tokens 100 \
    --temperature 1.2 \
    --top-p 0.98
```

## Troubleshooting

### Issue: Repetitive Output

**Solutions**:
- Increase temperature
- Use top-p sampling
- Increase top-k

### Issue: Nonsensical Output

**Solutions**:
- Decrease temperature
- Reduce top-k
- Use lower top-p

### Issue: Slow Generation

**Solutions**:
- Use GPU (`--device cuda`)
- Reduce max_new_tokens
- Use smaller model

### Issue: Out of Memory

**Solutions**:
- Reduce max_length
- Use CPU
- Close other applications

## Related Documentation

- [Model Architecture](model.md)
- [Training](training.md)
- [Configuration](configuration.md)

---

**Next**: Learn about [data preprocessing](preprocessing.md)
