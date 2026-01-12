# Training Guide

Complete guide to training Fragment_LLM models.

## Training Pipeline Overview

```
Data Preparation → Tokenizer Training → Model Training → Checkpointing → Evaluation
```

## Quick Start

Basic training command:
```bash
python train.py --epochs 10
```

## Training Script

**Location**: `train.py`

The training script handles:
- Data loading and validation
- Tokenizer training/loading
- Model initialization
- Training loop with gradient accumulation
- Validation
- Checkpoint saving

## Configuration Options

### Model Architecture

```bash
--vocab-size 10000        # Vocabulary size [256-100000]
--block-size 512          # Context length [128-2048]
--n-layer 6               # Number of transformer layers [1-48]
--n-head 6                # Number of attention heads [1-32]
--n-embd 384              # Embedding dimension [64-2048]
--dropout 0.1             # Dropout rate [0.0-0.9]
```

### Training Hyperparameters

```bash
--batch-size 16           # Batch size [1-256]
--grad-accum-steps 4      # Gradient accumulation steps
--learning-rate 3e-4      # Learning rate (0-1e-2]
--epochs 10               # Number of epochs
--max-iters 10000         # Maximum iterations
```

### System Options

```bash
--use-amp                 # Enable mixed precision (default: True)
--no-amp                  # Disable mixed precision
--retrain-tokenizer       # Force tokenizer retraining
--use-wandb               # Enable W&B logging
```

## Hardware-Specific Configurations

### Low-End PC (4-8GB RAM, No GPU)

```bash
python train.py \
    --batch-size 4 \
    --grad-accum-steps 16 \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 256 \
    --block-size 256 \
    --epochs 5
```

**Expected**: ~30 min/epoch on CPU

### Mid-Range PC (8-16GB RAM, GTX 1660)

```bash
python train.py \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --n-layer 6 \
    --n-head 6 \
    --n-embd 384 \
    --block-size 512 \
    --epochs 10 \
    --use-amp
```

**Expected**: ~3-5 min/epoch on GPU

### High-End PC (16GB+ RAM, RTX 3060+)

```bash
python train.py \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --block-size 1024 \
    --epochs 20 \
    --use-amp \
    --use-wandb
```

**Expected**: ~2-3 min/epoch on GPU

## Training Process

### 1. Tokenizer Training

If `tokenizer.json` doesn't exist:
```python
tokenizer = SimpleTokenizer(max_vocab_size=args.vocab_size)
tokenizer.train(text, vocab_size=args.vocab_size)
tokenizer.save('tokenizer.json')
```

### 2. Data Loading

```python
train_loader = create_dataloader(
    'data/processed/train_tiny.txt',
    tokenizer,
    batch_size=config.batch_size,
    block_size=config.block_size
)
```

### 3. Model Initialization

```python
model = AIModel(config)
# Weight initialization follows GPT-2 conventions
```

### 4. Training Loop

```python
for epoch in range(epochs):
    # Training
    for batch in train_loader:
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            logits, loss = model(x, y)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
    
    # Validation
    val_loss = validate()
    
    # Checkpoint saving
    if (epoch + 1) % 5 == 0:
        save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
```

## Optimization Techniques

### 1. Gradient Accumulation

Simulates larger batch sizes:
```python
effective_batch_size = batch_size × grad_accum_steps
# Example: 16 × 4 = 64
```

**Benefits**:
- Train with larger effective batch sizes
- Reduces memory usage
- Improves convergence

### 2. Mixed Precision Training

Uses FP16 for forward/backward passes:
```python
with torch.cuda.amp.autocast():
    logits, loss = model(x, y)
```

**Benefits**:
- 50% memory reduction
- 2-3x faster training
- Minimal accuracy loss

### 3. Gradient Clipping

Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Learning Rate Scheduling

Cosine annealing:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_iters
)
```

## Monitoring Training

### Console Output

```
Epoch 1/10
Training: 100%|████████| 50/50 [00:30<00:00]
Train loss: 4.5234
Val loss: 4.3210

Epoch 2/10
Training: 100%|████████| 50/50 [00:28<00:00]
Train loss: 3.8123
Val loss: 3.7456
```

### W&B Integration

Enable with `--use-wandb`:
```bash
python train.py --use-wandb
```

Tracks:
- Training loss
- Validation loss
- Learning rate
- Gradient norms

## Checkpointing

### Automatic Checkpoints

Saved every 5 epochs to `checkpoints/`:
```
checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── checkpoint_epoch_15.pt
```

### Final Model

Saved to `models/final_model.pt`

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config
}
```

## Troubleshooting

### Out of Memory

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Reduce model size: `--n-layer 4 --n-embd 256`
3. Reduce context: `--block-size 256`
4. Increase gradient accumulation: `--grad-accum-steps 8`

### Slow Training

**Symptoms**: Very slow iterations

**Solutions**:
1. Enable mixed precision: `--use-amp`
2. Reduce data loading workers
3. Use GPU if available
4. Reduce model size

### Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:
1. Check data quality
2. Reduce learning rate: `--learning-rate 1e-4`
3. Increase model capacity
4. Train longer

### NaN Loss

**Symptoms**: Loss becomes NaN

**Solutions**:
1. Reduce learning rate
2. Enable gradient clipping (already enabled)
3. Check for data issues
4. Use mixed precision carefully

## Best Practices

1. **Start Small**: Begin with a small model to verify pipeline
2. **Monitor Validation**: Watch for overfitting
3. **Save Checkpoints**: Don't lose progress
4. **Use Mixed Precision**: Faster and more memory efficient
5. **Gradient Accumulation**: Simulate larger batches
6. **Learning Rate**: Start with 3e-4, adjust as needed

## Advanced Topics

### Custom Training Loop

See `src/trainer.py` for implementation details.

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Multi-GPU Training

Not currently implemented, but can be added with:
```python
model = nn.DataParallel(model)
```

## Related Documentation

- [Model Architecture](model.md)
- [Configuration](configuration.md)
- [Dataset](dataset.md)

---

**Next**: Learn about [text generation](inference.md)
