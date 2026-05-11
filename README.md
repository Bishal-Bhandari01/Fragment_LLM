# Fragment LLM: Your Secure AI Built from Scratch 🚀

Welcome to Fragment LLM! We built this project to be a production-ready, highly secure AI language model that you can actually run and train on an everyday computer—no massive server farm required.

Whether you're a student learning how AI works, a researcher, or just someone curious about training your own mini-ChatGPT, this project makes it easy, safe, and surprisingly fast.

## 📚 Where to Start?

We've broken our documentation down into easy-to-read guides. Don't worry if some of the terms sound complex; we've tried our best to explain things simply!

- **🔰 [Quick Start](docs/quickstart.md)** - The fastest way to get things running (Start here!)
- **🏗️ [How It Works](docs/architecture.md)** - A bird's-eye view of how all the pieces fit together.
- **🧠 [Inside the AI](docs/model.md)** - The actual brain of the AI and how it thinks.
- **🎓 [Training Guide](docs/training.md)** - How to teach your AI new things.
- **🔡 [Tokenizer](docs/tokenizer.md)** - How the AI learns to read text.
- **💾 [Data Handling](docs/dataset.md)** - How we feed data to the AI safely.
- **🧹 [Prep Work](docs/preprocessing.md)** - Getting your messy text files ready for the AI.
- **💬 [Chatting](docs/inference.md)** - How to talk to your AI once it's trained.
- **⚙️ [Settings](docs/configuration.md)** - All the dials and knobs you can turn.
- **🔒 [Security](docs/security.md)** - How we keep this safe for you to use.
- **📖 [Code Reference](docs/api-reference.md)** - For the developers who want to dig into the code.

## 🔒 Safety First

Running AI code on your computer shouldn't be risky. We take security very seriously. 
- We follow industry-standard security guidelines (like OWASP and CIS) to make sure malicious files can't trick the system.
- We never use unsafe "Pickle" files (a common way viruses sneak into AI models). Everything is saved safely as standard text/JSON.
- The system automatically limits how much memory it uses so it doesn't crash your computer.

## 🚀 Built for Regular Computers

You shouldn't need a $10,000 graphics card to play with AI. Here’s how we made it work for regular PCs:
- **Choose Your Size**: We have preset sizes ranging from "tiny" (which runs on almost anything) up to larger models.
- **Smart Memory**: We use tricks like "Mixed Precision" and "Gradient Checkpointing" which sound fancy, but basically mean the AI uses about half the memory it normally would.
- **Fast Architecture**: We use modern AI shortcuts (like RoPE and Flash Attention) to make the AI learn faster without working your computer to the bone.
- **Streaming Data**: Even if you have gigabytes of text, we stream it to the AI in bite-sized pieces so your RAM never fills up.

## 🎯 Try It Out!

Want to get it running right now? It's just a few simple commands. (Check out the [Quick Start Guide](docs/quickstart.md) for more details!)

```bash
# 1. Install the required tools
pip install torch pandas tqdm

# 2. Get your data ready
python scripts/preprocessor.py

# 3. Teach the AI! (We're using the 'small' preset here)
python train.py --preset small --epochs 10

# 4. Chat with your new AI
python src/inference.py --interactive
```

## 🤝 Want to Help?

We'd love your help! If you want to contribute code, please take a quick look at our [Security Guide](docs/security.md) first. We just want to make sure any new code is as safe as the rest of the project.

## 📄 License

This project is completely free and open source under the MIT License. (See the LICENSE file for the boring legal details).

## 🌟 A Big Thanks To...

- The original "Attention is All You Need" paper that made this whole AI wave possible.
- The open-source community for teaching us how to build these amazing tools.