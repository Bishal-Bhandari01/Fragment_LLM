# System Architecture

Overview of Fragment_LLM system design and component interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Fragment_LLM System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Preprocessing│───▶│  Tokenizer   │───▶│   Dataset    │  │
│  │              │    │              │    │              │  │
│  │ - Cleaning   │    │ - BPE Train  │    │ - Loading    │  │
│  │ - Validation │    │ - Encode     │    │ - Batching   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Training Pipeline                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │   Model    │  │  Trainer   │  │ Optimizer  │     │  │
│  │  │            │  │            │  │            │     │  │
│  │  │ - Forward  │  │ - Loop     │  │ - AdamW    │     │  │
│  │  │ - Loss     │  │ - Val      │  │ - Scheduler│     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Checkpoints  │                                           │
│  │              │                                           │
│  │ - Saving     │                                           │
│  │ - Loading    │                                           │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Inference Engine                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │  Prompt    │  │ Generation │  │  Decoding  │     │  │
│  │  │ Validation │  │            │  │            │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Preprocessing (`scripts/preprocessor.py`)

**Purpose**: Prepare raw data for training

**Responsibilities**:
- File discovery and validation
- Text cleaning and normalization
- Format conversion
- Security checks

**Inputs**: Raw text/parquet files  
**Outputs**: Clean text files in `data/processed/`

### 2. Tokenizer (`src/tokenizer.py`)

**Purpose**: Convert text to token IDs

**Responsibilities**:
- BPE algorithm implementation
- Vocabulary management
- Encoding/decoding
- Secure serialization

**Inputs**: Text strings  
**Outputs**: Token ID sequences

### 3. Dataset (`src/dataset.py`)

**Purpose**: Load and batch data for training

**Responsibilities**:
- File loading with security checks
- Tokenization integration
- Sequence creation
- Batching

**Inputs**: Text files + tokenizer  
**Outputs**: Batched (input, target) tensors

### 4. Model (`src/model.py`)

**Purpose**: Transformer neural network

**Responsibilities**:
- Forward pass computation
- Loss calculation
- Text generation
- Parameter management

**Inputs**: Token ID tensors  
**Outputs**: Logits, loss, generated sequences

### 5. Trainer (`src/trainer.py`)

**Purpose**: Training loop orchestration

**Responsibilities**:
- Training epoch execution
- Validation
- Optimization
- Checkpoint saving

**Inputs**: Model, dataloaders, config  
**Outputs**: Trained model checkpoints

### 6. Inference Engine (`src/inference.py`)

**Purpose**: Text generation interface

**Responsibilities**:
- Prompt validation
- Generation parameter management
- Interactive mode
- Safety constraints

**Inputs**: Prompts, generation parameters  
**Outputs**: Generated text

### 7. Configuration (`src/config.py`)

**Purpose**: Centralized configuration

**Responsibilities**:
- Parameter validation
- Security settings
- Default values
- Constraint enforcement

## Data Flow

### Training Flow

```
Raw Data
   ↓
[Preprocessor] → Clean Text Files
   ↓
[Tokenizer Training] → tokenizer.json
   ↓
[Dataset] → (input, target) batches
   ↓
[Model Forward] → logits, loss
   ↓
[Backward Pass] → gradients
   ↓
[Optimizer Step] → updated weights
   ↓
[Checkpoint Save] → model.pt
```

### Inference Flow

```
User Prompt
   ↓
[Validation] → sanitized prompt
   ↓
[Tokenizer Encode] → token IDs
   ↓
[Model Generate] → new token IDs
   ↓
[Tokenizer Decode] → generated text
   ↓
Output
```

## Design Decisions

### 1. Modular Architecture

**Decision**: Separate components for each concern

**Rationale**:
- Easier testing
- Better maintainability
- Clear responsibilities
- Reusable components

### 2. Security-First Design

**Decision**: Validation at every boundary

**Rationale**:
- Defense in depth
- Fail fast on invalid input
- Prevent cascading failures
- Audit trail

### 3. Configuration-Driven

**Decision**: Centralized configuration with validation

**Rationale**:
- Single source of truth
- Consistent validation
- Easy experimentation
- Prevents invalid states

### 4. Memory Efficiency

**Decision**: Optimizations for low-end hardware

**Rationale**:
- Broader accessibility
- Gradient accumulation
- Mixed precision training
- Efficient data loading

## Component Interactions

### Training Script (`train.py`)

Orchestrates the entire training pipeline:

```python
1. Load/create tokenizer
2. Create dataloaders
3. Initialize model
4. Create trainer
5. Run training loop
6. Save final model
```

### Dependencies

```
train.py
├── src/config.py (LLMConfig)
├── src/tokenizer.py (SimpleTokenizer)
├── src/dataset.py (create_dataloader)
├── src/model.py (AIModel)
└── src/trainer.py (Trainer)

inference.py
├── src/config.py (LLMConfig)
├── src/tokenizer.py (SimpleTokenizer)
└── src/model.py (AIModel)

preprocessor.py
└── (standalone)
```

## Performance Considerations

### Memory Management

1. **Gradient Accumulation**: Simulate larger batches
2. **Mixed Precision**: FP16 for 50% memory reduction
3. **Efficient Data Loading**: Streaming, no duplication

### Compute Optimization

1. **Combined QKV**: Single matrix multiplication
2. **Weight Tying**: Shared embeddings
3. **Pre-normalization**: Better gradient flow

### I/O Optimization

1. **Tokenization Caching**: Tokenize once, reuse
2. **DataLoader Workers**: Parallel data loading
3. **Pin Memory**: Faster GPU transfers

## Scalability

### Current Limitations

- Single GPU training
- In-memory datasets
- No distributed training

### Future Enhancements

- Multi-GPU support (DataParallel)
- Streaming datasets for huge data
- Distributed training (DDP)
- Model parallelism for larger models

## Related Documentation

- [Model](model.md) - Model architecture details
- [Training](training.md) - Training pipeline
- [Security](security.md) - Security features

---

**Complete**: Return to [main README](../README.md)
