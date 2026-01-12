# Quick Start Guide

Get up and running with Fragment_LLM in minutes.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- CUDA-capable GPU (optional but recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Fragment_LLM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch>=2.0.0 pandas>=1.5.0 tqdm>=4.65.0
```

Optional (for experiment tracking):
```bash
pip install wandb>=0.15.0
```

## First Training Run

### Step 1: Prepare Data

Create the data directories:

```bash
mkdir -p data/raw data/processed
```

Download sample data (WikiText-2) or use your own text files:

```bash
# Place your .txt or .parquet files in data/raw/
# Example: data/raw/train.txt, data/raw/val.txt
```

### Step 2: Preprocess Data

```bash
python scripts/preprocessor.py
```

This will:
- Validate and clean your text data
- Create train/validation/test splits
- Save processed files to `data/processed/`

**Expected output:**
```
Preprocessor initialized
Processing train split...
Saved train.txt: 1000 lines, 0.5 MB
Processing validation split...
Saved val.txt: 100 lines, 0.05 MB
Preprocessing complete!
```

### Step 3: Train Your First Model

For a quick test run (low-end PC friendly):

```bash
python train.py --epochs 5 --batch-size 8
```

For better results (if you have more resources):

```bash
python train.py --epochs 10 --batch-size 16 --n-layer 6 --n-embd 384
```

**Expected output:**
```
Starting training pipeline...
Device: cuda
Vocab size: 5000
Model has 38,000,000 parameters
Starting training...

Epoch 1/10
Training: 100%|████████| 50/50 [00:30<00:00]
Train loss: 4.5234
Val loss: 4.3210
```

### Step 4: Generate Text

Interactive mode (recommended for beginners):

```bash
python src/inference.py --interactive
```

Then type your prompts:
```
Prompt: Once upon a time
Generated: Once upon a time, there was a small village...
```

Single generation:

```bash
python src/inference.py --prompt "The future of AI" --max-new-tokens 100
```

## Common Issues

### Issue: "Python was not found"
**Solution**: Install Python 3.8+ and add it to your PATH

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or model size:
```bash
python train.py --batch-size 4 --n-layer 4 --n-embd 256
```

### Issue: "File not found: tokenizer.json"
**Solution**: The tokenizer is created during first training. Make sure you've run training at least once.

### Issue: "Dataset file is empty"
**Solution**: Check that your data files in `data/raw/` contain text

## Next Steps

- [Training Guide](training.md) - Learn advanced training techniques
- [Model Architecture](model.md) - Understand the model structure
- [Configuration](configuration.md) - Customize your setup
- [Inference Guide](inference.md) - Advanced text generation

## Quick Reference

### Minimal Training Command
```bash
python train.py
```

### Recommended Training Command
```bash
python train.py --epochs 10 --batch-size 16 --use-amp
```

### Generate Text
```bash
python src/inference.py --interactive
```

### Monitor Training
```bash
python train.py --use-wandb  # Requires wandb account
```

---

**Need help?** Check the [main README](../README.md) or open an issue.
