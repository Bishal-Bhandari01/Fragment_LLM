# Tweaking the Settings ⚙️

Fragment LLM is designed to be highly customizable. Whether you are running it on a 10-year-old laptop or a massive server, there are settings (configurations) you can tweak to make it run perfectly.

This guide explains what all those dials and knobs actually do.

---

## 🧠 Model Architecture (How Big is the Brain?)

If you open `src/config.py`, you'll see the `LLMConfig` class. This is where we decide how the AI's brain is built.

| Setting | What it actually means |
|---------|------------------------|
| `vocab_size` | **Vocabulary Size**: How many unique words/tokens the AI knows. Usually around 5,000 to 10,000. |
| `block_size` | **Memory Window**: How many words the AI can remember at one time. If you set this to 512, it can read a few paragraphs at once. If you set it to 2048, it can read a short story! |
| `n_layer` | **Layers**: How many times the AI processes the text before giving an answer. More layers = a smarter AI (but it takes longer to run). |
| `n_head` | **Attention Heads**: How many different "trains of thought" the AI can have at once while reading a sentence. |
| `n_embd` | **Brain Size**: The size of the mathematical vector used to represent a single word. Bigger numbers mean a deeper understanding of language. |

---

## 🏋️ Training Parameters (How Fast Does it Learn?)

These settings control how the AI studies the data you give it.

| Setting | What it actually means |
|---------|------------------------|
| `batch_size` | **Reading Chunk**: How many sentences the AI reads at the exact same time. If your computer crashes with an "Out of Memory" error, lower this number immediately! |
| `gradient_accumulation_steps` | **The Virtual Batch**: If your computer can only handle a small `batch_size`, this setting lets it read several small batches before it updates its brain. It's a clever hack to save RAM. |
| `learning_rate` | **Learning Speed**: How drastically the AI changes its mind when it makes a mistake. Too high, and it gets confused. Too low, and it takes years to learn anything. |
| `warmup_iters` | **Warm-up**: We start the learning rate near zero and slowly turn it up. This prevents the AI from panicking and forgetting everything in the first 5 minutes of training. |
| `use_amp` & `use_bf16` | **Math Shortcuts**: If true, the AI uses half-precision math. It saves 50% of your RAM without making the AI any dumber! |

---

## 🛡️ Security Parameters (Keeping it Safe)

We don't just want the AI to be smart; we want it to be safe. The `SecurityConfig` class acts like a bouncer for your system.

| Setting | What it actually means |
|---------|------------------------|
| `max_file_size_mb` | **File Limit**: If you try to feed the AI a text file larger than this limit, the system rejects it to prevent your computer from crashing. |
| `allowed_base_dirs` | **Safe Zones**: The AI is only allowed to read files from these specific folders (like `data/`). If a hacker tries to trick it into reading your `/passwords` folder, the system stops them. |
| `block_harmful_generation` | **Content Filter**: If turned on, the system checks the AI's output and blocks it if it starts saying nasty things. |

---

## 💡 Quick Presets

Don't want to mess with individual settings? We built presets directly into `train.py` so you can just pick a size and go!

### 💻 The "Tiny" Preset (For Laptops)
If you're testing things out or just have an old laptop, use this. It builds an AI with about 15 million parameters. It's not going to write a novel, but it will run on basically anything.
* **Usage**: `python train.py --preset tiny`

### 🎮 The "Small" Preset (Default)
A great balance of speed and smarts. It builds a 38-million parameter model. If you have an average graphics card, this is where you want to be.
* **Usage**: `python train.py --preset small`

### 🚀 The "Medium" Preset (For Good PCs)
If you have a modern GPU with 8GB+ of memory, let's turn it up. This builds a 124-million parameter model (similar to the original GPT-1).
* **Usage**: `python train.py --preset medium`

---

## 🛠️ Auto-Validation (We Catch Your Mistakes)

Don't worry about breaking anything. If you accidentally set `n_layer` to a billion, the code won't crash your computer. It checks all your settings before it starts and gives you a friendly error message if something looks wrong.

```python
config = LLMConfig(
    vocab_size=50,  # ❌ Too small! The AI needs at least 256 words.
    n_layer=100     # ❌ Too many layers! Your computer will cry.
)
# The system will immediately stop and tell you what to fix.
```

## What's Next?
Now that you know how the settings work, read our [Security Guide](security.md) to see how we protect your data.
