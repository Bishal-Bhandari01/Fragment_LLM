# Secure LLM from Scratch

A production-ready, security-hardened GPT-style language model implementation optimized for low-end PCs and fast learning.

## � Documentation

The documentation is split into specialized guides:

- **🔰 [Quick Start](docs/quickstart.md)** - Get up and running in minutes
- **🏗️ [Architecture](docs/architecture.md)** - System design and component overview
- **🧠 [Model Architecture](docs/model.md)** - Details on the GPT implementation
- **🎓 [Training Guide](docs/training.md)** - How to train models effectively
- **🔡 [Tokenizer](docs/tokenizer.md)** - BPE tokenizer details
- **💾 [Dataset Handling](docs/dataset.md)** - Secure data loading
- **🧹 [Preprocessing](docs/preprocessing.md)** - Data preparation pipeline
- **💬 [Inference](docs/inference.md)** - Text generation parameters and guide
- **⚙️ [Configuration](docs/configuration.md)** - Full parameter reference
- **�🔒 [Security](docs/security.md)** - Detailed security features and rules
- **📖 [API Reference](docs/api-reference.md)** - Class and function documentation

## 🔒 Security Features

- **OWASP ASVS 4.0 Compliant**: Input validation, secure deserialization, path traversal prevention
- **CWE Top 25 Mitigations**: Protection against critical vulnerabilities
- **CIS Benchmark Aligned**: Follows security best practices
- **No Pickle Usage**: JSON-based serialization to prevent arbitrary code execution
- **Resource Limits**: Protection against DoS and resource exhaustion
- **Atomic Operations**: Safe checkpoint saving and loading

## 🚀 Optimizations for Low-End PCs

- **Reduced Model Size**: 6 layers, 384 embedding dimension (38M parameters)
- **Gradient Accumulation**: Effective batch size 64 with actual batch 16
- **Mixed Precision Training**: FP16 to reduce memory usage by 50%
- **Efficient Architecture**: Combined QKV projections, optimized attention
- **Memory-Efficient Data Loading**: Streaming without loading full dataset

## 🎯 Quick Start Summary

For detailed instructions, see the [Quick Start Guide](docs/quickstart.md).

```bash
# 1. Install
pip install torch pandas tqdm

# 2. Prepare Data
python scripts/preprocessor.py

# 3. Train
python train.py --epochs 10

# 4. Generate
python src/inference.py --interactive
```

## 📂 Project Structure

```
Fragment_LLM/
├── src/                    # Source code
│   ├── config.py           # Configuration
│   ├── tokenizer.py        # Tokenizer
│   ├── dataset.py          # Data handling
│   ├── model.py            # Neural network
│   ├── trainer.py          # Training loop
│   └── inference.py        # Generation
├── scripts/                # Helper scripts
│   └── preprocessor.py     # Data prep
├── docs/                   # Documentation
│   ├── quickstart.md       # Getting started
│   ├── architecture.md     # System design
│   └── ...                 # Feature docs
├── train.py                # Main entry point
├── SECURITY_RULES.md       # Security guidelines
└── README.md               # This file
```

## 🤝 Contributing

See [Security documentation](docs/security.md) before contributing. Ensure all security rules are followed:
1. Input validation for all public methods
2. No unsafe deserialization (pickle)
3. Path checks for file operations
4. Resource limits enforcement

## 📄 License

MIT License - See LICENSE file for details

## 🌟 Acknowledgments

- Based on GPT architecture from "Attention is All You Need"
- Optimized for educational and research purposes
- Security hardened following OWASP, CWE, and CIS guidelines