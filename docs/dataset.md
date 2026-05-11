# How We Handle Your Data 💾

Before an AI can learn to speak, it needs something to read! This page explains how we load your text files and feed them to the AI safely and efficiently.

## The Basics

Behind the scenes, we use a class called `TextDataset`. Think of it as a super-librarian that:
- Checks if the book (your file) is safe to open.
- Makes sure the book isn't so huge that it crushes the library (your computer's RAM).
- Translates the English words into numbers (tokens) that the AI understands.

## Quick Example

If you want to poke around in Python, here's how easy it is to load your data:

```python
from src.dataset import create_dataloader
from src.tokenizer import SimpleTokenizer

# 1. Load our dictionary (Tokenizer)
tokenizer = SimpleTokenizer.load('tokenizer.json')

# 2. Tell the librarian to grab our training text
dataloader = create_dataloader(
    'data/processed/train.txt',
    tokenizer,
    batch_size=16, # How many sentences to read at once
    block_size=512 # How long each sentence can be
)
```

## How It Actually Works

When you give us a text file, here's exactly what happens:

1. **Security Check**: We make sure nobody is trying to trick the system by passing a weird file path (like `../../../passwords.txt`).
2. **Size Check**: We check the file size. If it's a 50-gigabyte file and you only have 8 gigabytes of RAM, we stop before your computer freezes!
3. **Reading**: We read the text.
4. **Translation**: We use our Tokenizer to turn the words into numbers.
5. **Chopping**: We chop the long list of numbers into bite-sized sequences the AI can digest.

### What Does the AI Actually See?

Imagine your text file says: `"Hello world! This is a test."`

The AI doesn't see those words. Instead, it sees a list of ID numbers:
`[245, 128, 67, 89, 12, 45, ...]`

We feed these numbers to the AI in pairs. We give it an "input" sequence, and the "target" is just the same sequence shifted over by one word. (We are basically asking the AI: "Given these words, what is the very next word?")

## How We Stop Your Computer From Crashing

Loading text can be surprisingly heavy on your computer. Here are three ways we fix that:

### 1. In-Memory Mode (The Fast Way)
If your text file isn't too big, we just load the whole thing into memory. It's lightning-fast, but it uses the most RAM.

### 2. Streaming Mode (The Smart Way)
If you have a massive dataset, you can turn on Streaming Mode by adding `--streaming` when you train:

```bash
python train.py --streaming
```

Instead of loading the whole book at once, the librarian just keeps their finger on the page and reads it to the AI chunk-by-chunk. This uses almost *zero* RAM!

### 3. Memory-Mapped Mode (The Heavy-Duty Way)
For gigantic, pre-processed datasets, we use memory mapping. It's a fancy way of pretending a file on your hard drive is actually in your RAM.

## Security First 🔒

We are very strict about what files can be loaded. 

- **No Sneaky Paths**: We block directory traversal attacks (CWE-22). If someone tries to load `/etc/passwd` or `../../../secrets.txt`, the system instantly rejects it.
- **Size Limits**: By default, we block text files larger than 500MB to prevent your computer from running out of memory and crashing (CWE-400).

## Troubleshooting Common Errors

- **"File not found"**  
  *Fix*: Double-check your spelling! Also, make sure you ran the `preprocessor.py` script first.

- **"Dataset too small"**  
  *Fix*: Your text file is too short! The AI needs at least enough words to fill one `block_size`. Give it a longer text file.

- **"Out of memory during data loading"**  
  *Fix*: Your file is too big for your computer. Try using `--streaming` when you run `train.py`!

## What's Next?
- 🔡 [Learn how the AI learns to read (Tokenizer)](tokenizer.md)
- 🧹 [Learn how we clean up the data (Preprocessing)](preprocessing.md)
