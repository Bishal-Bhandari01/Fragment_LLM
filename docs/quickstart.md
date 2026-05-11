# Quick Start Guide 🚀

Ready to build your own AI? You're in the right place. This guide will walk you through getting Fragment LLM up and running on your computer in just a few minutes. 

No supercomputers needed!

## What You Need Before We Start

- **Python 3.8 or higher**: (The language the AI is written in).
- **At least 4GB of RAM**: (Though 8GB+ is highly recommended so things run smoothly).
- **A graphics card (GPU)**: Completely optional, but if you have an Nvidia GPU, things will train *much* faster.

---

## Step 1: Getting the Code

First, you'll need to grab the code from our repository and jump into the folder. Open your terminal (or Command Prompt) and run:

```bash
git clone <your-repo-url>
cd Fragment_LLM
```

## Step 2: Install the Tools

Our AI relies on a few helper libraries (like PyTorch). We can install all of them at once using:

```bash
pip install -r requirements.txt
```

*(Optional: If you want nice graphs to track how smart your AI is getting, you can also install Weights & Biases by running `pip install wandb>=0.15.0`)*

---

## Your First Training Run! 🎓

Training an AI is like teaching a toddler to speak by reading them millions of books. We just need to give it the books!

### 1. Prepare Your Data

Create some folders to hold your text files:

```bash
mkdir -p data/raw data/processed
```

Now, drop any text you want the AI to learn from into the `data/raw/` folder. It can be a `.txt` file containing Shakespeare, Wikipedia articles, or your own journal entries!

### 2. Clean Up the Data

The AI likes its reading material clean and organized. We wrote a handy script that checks your files and organizes them for the AI:

```bash
python scripts/preprocessor.py
```

This script will read your raw text and neatly save it into the `data/processed/` folder.

### 3. Start Training

This is the fun part! Let's teach the AI.

**If you are on a basic laptop:** Use our 'tiny' preset. It's perfectly sized for standard computers.
```bash
python train.py --preset tiny --epochs 5 --batch-size 8
```

**If you have a beefy gaming PC or a good GPU:** Let's crank it up and use the 'small' preset!
```bash
python train.py --preset small --epochs 10 --batch-size 16
```

Sit back and grab a coffee ☕. You'll see a progress bar showing you the AI's "Loss" going down over time. (Lower loss means it's making fewer mistakes and getting smarter!)

### 4. Talk to Your AI 💬

Once training finishes, it's time to see what your AI learned! We've included an interactive chat mode so you can type prompts directly to it.

```bash
python src/inference.py --interactive
```

Type something like "Once upon a time", hit enter, and watch your brand-new AI finish the sentence for you!

---

## Uh Oh! (Common Troubleshooting)

Hitting a roadblock? Don't worry, it happens to the best of us.

- **"Python was not found"**  
  *Fix:* Make sure Python 3.8+ is installed and checked to "Add to PATH" in the installer.
  
- **"CUDA out of memory"**  
  *Fix:* Your graphics card is overwhelmed. Try running a smaller batch size: `python train.py --preset tiny --batch-size 4`

- **"File not found: tokenizer.json"**  
  *Fix:* This file gets created automatically the very first time you train the AI. Just make sure you run the training step!

- **"Dataset file is empty"**  
  *Fix:* Double-check that the text files you dropped in `data/raw/` actually have words in them!

## What's Next?

Feeling adventurous? Check out the rest of the guides to level up:
- 🎓 [Learn advanced training tricks](training.md)
- 🧠 [See how the AI's "brain" works](model.md)
- ⚙️ [Tweak the settings to make it your own](configuration.md)
