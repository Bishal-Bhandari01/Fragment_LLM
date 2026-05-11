# Training Your AI 🎓

Teaching an AI to speak is a lot like teaching a toddler. You need to give it plenty of reading material, correct its mistakes, and give it time to learn. This guide walks you through the process of training your Fragment LLM model.

## The Big Picture

Here is the journey your data takes to become a fully trained AI:

1. **Prep the Data**: Clean up your text files so they are easy to read.
2. **Teach the Alphabet**: Train the Tokenizer so the AI knows how to turn your words into numbers.
3. **Hit the Books**: Feed the numbers to the AI and let it practice guessing the next word.
4. **Save Progress**: Periodically save the AI's "brain" so you don't lose your work if your computer crashes.

## Let's Get Started!

If you just want to get up and running immediately, this is the magic command:

```bash
# This trains a 'small' model for 10 full passes (epochs) over your data
python train.py --preset small --epochs 10
```

## How Big Should My AI Be? (Hardware Presets)

We have pre-configured "presets" depending on how powerful your computer is. You don't need to be an AI engineer to figure out the settings!

### 💻 The "Tiny" Preset (Old Laptops / No Graphics Card)
Got an older computer with 4-8GB of RAM? No problem. The `tiny` preset is designed to run on almost anything. It won't be writing Shakespeare, but it will learn!
```bash
python train.py --preset tiny --batch-size 4 --epochs 5
```
*(Expect this to take about 30 minutes per pass on a standard CPU).*

### 🎮 The "Small" Preset (Gaming PCs / 8-16GB RAM)
If you have a decent graphics card (like a GTX 1660), use the default `small` preset.
```bash
python train.py --preset small --epochs 10 --use-amp
```
*(Expect this to take about 3-5 minutes per pass).*

### 🚀 The "Medium" Preset (High-End PCs / RTX 3060+)
If you have a beefy PC, let's turn the dials up. This model is much smarter.
```bash
python train.py --preset medium --epochs 20 --use-amp --use-bf16 --grad-ckpt
```

---

## Cheat Sheet: Advanced Tweaks 🎛️

Want to customize things? You can add any of these flags to the `train.py` command to change how the AI learns.

**How it Learns:**
- `--epochs 10` : How many times the AI reads your entire dataset from start to finish.
- `--batch-size 16` : How many sentences the AI tries to read at once before checking its answers. (If you get "Out of Memory" errors, lower this number!)
- `--learning-rate 3e-4` : How big of a jump the AI makes when it realizes it made a mistake. If it's too high, the AI forgets things. If it's too low, it takes forever to learn.

**Memory Savers:**
- `--use-amp` or `--use-bf16` : Does the math using half the decimal points. Cuts memory usage in half!
- `--grad-ckpt` : Gradient Checkpointing. Saves a ton of memory by re-doing some math on the fly instead of memorizing it.
- `--streaming` : Don't load the text file into RAM. Stream it directly from the hard drive!

---

## Keeping an Eye on Things 📈

As your AI trains, it will print out its progress. You are looking for the **Loss** to go down. "Loss" is basically the AI's error rate.

```
Epoch 1/10
Training: 100%|████████| 50/50 [00:30<00:00]
Train loss: 4.5234
Val loss: 4.3210
```
- **Train Loss**: How well it is doing on the text it's currently reading.
- **Val Loss**: How well it is doing on a "pop quiz" of text it hasn't seen before. (If Train Loss goes down but Val Loss goes up, the AI is just memorizing the book instead of actually learning to speak!)

### Want Beautiful Graphs?
If you want to track your AI's brain waves in real-time, you can use a free service called Weights & Biases:
```bash
python train.py --use-wandb
```

---

## Saving Your Work (Checkpoints) 💾

Every 5 epochs, the system automatically saves a copy of the AI's brain into the `checkpoints/` folder. We call these "checkpoints."

If your power goes out, you can pick right back up where you left off!
```bash
python train.py --preset small --resume checkpoints/checkpoint_epoch_5.pt
```
When training completely finishes, your shiny new AI will be saved as `models/final_model.pt`.

---

## Troubleshooting Guide 🚑

- **"CUDA out of memory"**
  Your graphics card bit off more than it could chew. Try lowering the `--batch-size` (e.g., to 4 or 8) or turn on `--grad-ckpt`.

- **"My computer is freezing up!"**
  Your text file might be too big for your RAM. Try adding `--streaming` to your command.

- **"The Loss isn't going down at all."**
  Double-check your text files. If they are filled with gibberish, the AI won't know what to learn!

- **"Training is taking days..."**
  If you don't have a graphics card (GPU), training AI is very slow. Make sure you are using the `--preset tiny` flag.

## Going Pro (Multi-GPU)

If you happen to have a server with multiple graphics cards, you can train *way* faster by splitting the work across all of them using PyTorch's `torchrun`:
```bash
torchrun --standalone --nproc_per_node=4 train.py --preset 7b
```

---
**What's next?**  
Once your AI is trained, it's time to talk to it! Check out the [Chatting (Inference) Guide](inference.md).
