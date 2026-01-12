# Model Architecture

Detailed documentation of the Fragment_LLM transformer architecture.

## Overview

Fragment_LLM implements a GPT-style decoder-only transformer optimized for efficiency and security. The architecture follows the standard transformer design with several optimizations for low-end hardware.

## Architecture Diagram

```
Input Text
    ↓
[Tokenizer] → Token IDs
    ↓
[Token Embedding] (vocab_size × n_embd)
    ↓
[Position Embedding] (block_size × n_embd)
    ↓
[Dropout]
    ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  ├─ Layer Norm              │
│  ├─ Multi-Head Attention    │
│  ├─ Residual Connection     │
│  ├─ Layer Norm              │
│  ├─ Feed-Forward Network    │
│  └─ Residual Connection     │
└─────────────────────────────┘
    ↓
    ... (repeat n_layer times)
    ↓
[Final Layer Norm]
    ↓
[Language Model Head] (n_embd → vocab_size)
    ↓
Output Logits
```

## Components

### 1. Token Embedding

**Location**: `src/model.py` - `AIModel.__init__`

```python
self.wte = nn.Embedding(config.vocab_size, config.n_embd)
```

**Purpose**: Converts token IDs to dense vector representations

**Parameters**:
- Input: Token IDs (batch_size, sequence_length)
- Output: Embeddings (batch_size, sequence_length, n_embd)

**Default Config**:
- vocab_size: 5,000
- n_embd: 384

### 2. Position Embedding

**Location**: `src/model.py` - `AIModel.__init__`

```python
self.wpe = nn.Embedding(config.block_size, config.n_embd)
```

**Purpose**: Adds positional information to token embeddings

**Parameters**:
- Input: Position indices (1, sequence_length)
- Output: Position embeddings (1, sequence_length, n_embd)

**Default Config**:
- block_size: 512
- n_embd: 384

### 3. Multi-Head Self-Attention

**Location**: `src/model.py` - `CausalSelfAttention`

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout, bias):
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
```

**Key Features**:
- **Causal masking**: Prevents attending to future tokens
- **Combined QKV projection**: More efficient than separate Q, K, V
- **Scaled dot-product attention**: Standard attention mechanism

**Attention Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Parameters**:
- num_heads: 6 (default)
- head_dim: n_embd / num_heads = 64
- dropout: 0.1

**Memory Optimization**:
- Uses single linear layer for Q, K, V instead of three separate layers
- Reduces parameter count by ~33%

### 4. Feed-Forward Network (MLP)

**Location**: `src/model.py` - `MLP`

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
```

**Architecture**:
```
Input (n_embd)
    ↓
Linear (n_embd → 4*n_embd)
    ↓
GELU Activation
    ↓
Linear (4*n_embd → n_embd)
    ↓
Dropout
    ↓
Output (n_embd)
```

**Expansion Factor**: 4x (standard for transformers)

### 5. Transformer Block

**Location**: `src/model.py` - `TransformerBlock`

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x))  # Attention + residual
        x = x + self.mlp(self.ln_2(x))   # MLP + residual
        return x
```

**Design Choice**: Pre-normalization (LayerNorm before attention/MLP)
- More stable training
- Better gradient flow
- Standard in modern transformers

### 6. Language Model Head

**Location**: `src/model.py` - `AIModel.__init__`

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.wte.weight = self.lm_head.weight  # Weight tying
```

**Weight Tying**: Shares weights with token embedding
- Reduces parameters
- Improves performance
- Standard practice in language models

## Model Configurations

### Tiny (Low-End PCs)
```python
config = LLMConfig(
    vocab_size=5000,
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1
)
```
- **Parameters**: ~15M
- **Memory**: 2GB GPU / 1GB CPU
- **Use case**: Testing, low-resource environments

### Small (Default)
```python
config = LLMConfig(
    vocab_size=5000,
    block_size=512,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1
)
```
- **Parameters**: ~38M
- **Memory**: 4GB GPU / 2GB CPU
- **Use case**: General purpose, balanced performance

### Medium (High-End PCs)
```python
config = LLMConfig(
    vocab_size=10000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1
)
```
- **Parameters**: ~124M
- **Memory**: 8GB GPU / 4GB CPU
- **Use case**: Best quality, research

## Parameter Count Breakdown

For default config (Small):

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Token Embedding | 1,920,000 | 5.1% |
| Position Embedding | 196,608 | 0.5% |
| Attention Layers | 17,694,720 | 46.6% |
| MLP Layers | 17,694,720 | 46.6% |
| Layer Norms | 4,608 | 0.01% |
| LM Head | 0 (tied) | 0% |
| **Total** | **~38M** | **100%** |

## Memory Requirements

### Training (FP16 with gradient accumulation)

| Config | Model | Optimizer | Gradients | Activations | Total |
|--------|-------|-----------|-----------|-------------|-------|
| Tiny | 30MB | 60MB | 30MB | 100MB | ~220MB |
| Small | 76MB | 152MB | 76MB | 300MB | ~600MB |
| Medium | 248MB | 496MB | 248MB | 1GB | ~2GB |

### Inference (FP16)

| Config | Model | KV Cache | Total |
|--------|-------|----------|-------|
| Tiny | 30MB | 50MB | ~80MB |
| Small | 76MB | 150MB | ~230MB |
| Medium | 248MB | 500MB | ~750MB |

## Optimizations

### 1. Combined QKV Projection
Instead of three separate linear layers for Q, K, V:
```python
# Efficient (current)
self.c_attn = nn.Linear(n_embd, 3 * n_embd)
qkv = self.c_attn(x)
q, k, v = qkv.split(n_embd, dim=2)

# vs. Naive
self.q_proj = nn.Linear(n_embd, n_embd)
self.k_proj = nn.Linear(n_embd, n_embd)
self.v_proj = nn.Linear(n_embd, n_embd)
```

### 2. Weight Tying
Shares weights between token embedding and output projection:
- Saves ~2M parameters (for vocab_size=5000)
- Improves generalization

### 3. Pre-Normalization
LayerNorm before attention/MLP instead of after:
- Better gradient flow
- More stable training
- Allows deeper models

### 4. Gradient Checkpointing (Optional)
Can be enabled for very large models:
```python
# Not currently implemented, but can be added
torch.utils.checkpoint.checkpoint(block, x)
```

## Forward Pass

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    
    # Get embeddings
    tok_emb = self.wte(idx)  # (B, T, n_embd)
    pos_emb = self.wpe(positions)  # (1, T, n_embd)
    x = self.drop(tok_emb + pos_emb)
    
    # Apply transformer blocks
    for block in self.blocks:
        x = block(x)  # (B, T, n_embd)
    
    # Final layer norm and projection
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)
    
    # Compute loss if targets provided
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, vocab_size), 
                               targets.view(-1))
    
    return logits, loss
```

## Generation Process

See [Inference Guide](inference.md) for detailed generation documentation.

## Related Documentation

- [Training Guide](training.md) - How to train the model
- [Configuration](configuration.md) - Model configuration options
- [Architecture Overview](architecture.md) - System design

---

**Next**: Learn about [training the model](training.md)
