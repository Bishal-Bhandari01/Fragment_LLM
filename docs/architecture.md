# How It All Fits Together 🧩

Ever wonder how an AI actually goes from reading a pile of text files to talking back to you? This page explains the big picture of how Fragment LLM works, piece by piece.

## The Big Picture

Imagine our AI system as a factory. Raw materials (text) come in one side, and a smart, talking AI comes out the other.

```
1. The Cleanup Crew (Preprocessor) 
   Gets rid of weird characters and organizes the text.
         ↓
2. The Translator (Tokenizer) 
   Turns words into a secret code of numbers the AI can read.
         ↓
3. The Librarian (Dataset) 
   Feeds the numbers to the AI, chunk by chunk, so it doesn't get overwhelmed.
         ↓
4. The Brain (Model)
   The actual neural network that tries to guess the next word.
         ↓
5. The Teacher (Trainer)
   Checks the AI's guesses, corrects its mistakes, and updates its brain.
         ↓
6. The Chat Interface (Inference)
   Lets you type a prompt and watch the trained AI respond!
```

## Meet the Team (The Core Files)

Here are the main files in the `src/` (source) folder that make the magic happen:

### 1. `config.py` (The Rulebook)
This file holds all the settings. It decides how big the AI's brain will be, how much RAM it's allowed to use, and what security rules it has to follow. If you ever want to tweak the system, this is where you go.

### 2. `tokenizer.py` (The Translator)
AI models don't actually understand English. They understand math. The tokenizer is a dictionary that looks at words (like "apple") and turns them into numbers (like "4021"). It also does the reverse when the AI wants to speak back to you!

### 3. `dataset.py` (The Librarian)
If you tried to hand an AI a 10-Gigabyte text file all at once, your computer would crash. The Dataset's job is to slice the book into small, manageable pages and hand them to the AI one at a time. It even supports "streaming," which means it reads the file straight from your hard drive without clogging up your RAM.

### 4. `model.py` (The Brain)
This is the Transformer neural network. It's essentially a giant math equation. When it receives a sequence of numbers (words), it runs them through layers of attention (which help it figure out the context of the sentence) and spits out its best guess for what the *next* word should be.

### 5. `trainer.py` (The Teacher)
The Trainer oversees the learning process. It takes the AI's guess, looks at the actual correct word, and calculates how "wrong" the AI was (we call this the *Loss*). It then reaches into the AI's brain and tweaks the math so it gets it right next time.

### 6. `inference.py` (The Chat Window)
Once the AI is trained, we don't need the Teacher or the Librarian anymore. We just need the Brain and the Translator! The inference script takes your prompt, translates it into numbers, asks the Brain what comes next, and translates the answer back into English.

## Why Did We Build It This Way?

We made a few specific choices to make this project special:

### 1. Security First 🔒
We don't trust any file blindly. Before the Librarian reads a file, it checks to make sure the file path isn't trying to hack the system (like pointing to your passwords). If something looks fishy, it immediately shuts it down.

### 2. Built for Normal Computers 💻
Training AI is famously expensive. We used clever tricks to make it run on your laptop:
- **Gradient Checkpointing**: A fancy way of saving memory by re-doing a bit of math instead of storing huge files in RAM.
- **Mixed Precision**: We do the math using half the normal precision (FP16). The AI barely notices, but it cuts our memory usage in half!
- **Grouped Query Attention (GQA)**: A modern shortcut that makes the AI run faster when generating text.

### 3. Modular Pieces 🧱
Every piece of code is isolated. If you want to swap out our Tokenizer for a different one, you can! If you want to change how the Teacher grades the AI, you can do that without breaking the Brain. 

## Moving Up to the Big Leagues

If you outgrow your laptop and want to train a massive model, Fragment LLM scales up:
- **Multi-GPU Support**: Have 4 graphics cards? The system will automatically split the work across all of them (using something called DDP).
- **Streaming**: You can train on datasets that are larger than your entire hard drive by streaming the data over the internet!

---

**Where to next?**  
Want to see the Brain up close? Check out [Inside the AI (Model Architecture)](model.md).
