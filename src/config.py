"""
Configuration for Fragment_LLM — supports Tiny through 7B model sizes.
Modern LLM architecture: RoPE, GQA, SwiGLU, RMSNorm, Flash Attention.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LLMConfig:
    """
    Model + training configuration with full validation.
    Architecture aligned with modern LLMs (LLaMA-style):
      - RoPE positional embeddings (no learned wpe)
      - Grouped Query Attention (GQA) for KV-cache efficiency
      - SwiGLU feed-forward (gate * up → down)
      - RMSNorm instead of LayerNorm
    """

    # ── Architecture ───────────────────────────────────────────────────────
    vocab_size:        int   = 5_000
    block_size:        int   = 512        # context / sequence length
    n_layer:           int   = 6
    n_head:            int   = 6          # query heads
    n_kv_head:         int   = 6          # key/value heads (set < n_head for GQA)
    n_embd:            int   = 384
    # SwiGLU intermediate dim; if 0 → auto-set to int(8/3 * n_embd) rounded to 256
    intermediate_size: int   = 0
    rope_theta:        float = 10_000.0   # RoPE base frequency
    bias:              bool  = False      # No bias in modern LLMs
    dropout:           float = 0.0       # Typically 0 for LLM pre-training

    # ── Training ───────────────────────────────────────────────────────────
    batch_size:                  int   = 16
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 3e-4
    min_lr:                      float = 3e-5   # cosine decay floor
    max_iters:                   int   = 10_000
    warmup_iters:                int   = 500
    weight_decay:                float = 0.1
    grad_norm_clip:              float = 1.0

    # ── Memory / precision ─────────────────────────────────────────────────
    use_amp:                bool = True
    use_bf16:               bool = True   # bf16 > fp16 for large models
    gradient_checkpointing: bool = False  # Trades compute for memory; on for 7B
    use_flash_attention:    bool = True   # F.scaled_dot_product_attention

    # ── Multi-GPU (DDP via torchrun) ───────────────────────────────────────
    ddp: bool = False  # set automatically in train.py

    # ── Dataset streaming ──────────────────────────────────────────────────
    streaming:         bool  = False  # stream large datasets without loading all
    use_mmap:          bool  = True   # memory-map pre-tokenised .bin file
    mmap_dtype:        str   = "uint16"  # uint32 if vocab_size > 65 535

    # ── Security / resource limits ─────────────────────────────────────────
    max_file_size_mb:       int   = 0      # 0 = no limit (streaming handles it)
    max_sequence_length:    int   = 4_096
    allowed_file_extension: tuple = ('.txt', '.parquet', '.jsonl', '.json', '.csv')
    max_generation_tokens:  int   = 2_048
    temperature_min:        float = 0.1
    temperature_max:        float = 2.0

    def __post_init__(self) -> None:
        # ── auto intermediate size ─────────────────────────────────────────
        if self.intermediate_size == 0:
            raw = int(8 / 3 * self.n_embd)
            self.intermediate_size = (raw + 255) // 256 * 256  # round to 256

        # ── auto mmap dtype ───────────────────────────────────────────────
        if self.vocab_size > 65_535:
            self.mmap_dtype = "uint32"

        # ── architecture validation ───────────────────────────────────────
        if not (256 <= self.vocab_size <= 200_000):
            raise ValueError(f"vocab_size must be in [256, 200000], got {self.vocab_size}")
        if not (64 <= self.block_size <= self.max_sequence_length):
            raise ValueError(f"block_size must be in [64, {self.max_sequence_length}]")
        if not (1 <= self.n_layer <= 80):
            raise ValueError(f"n_layer must be in [1, 80], got {self.n_layer}")
        if not (1 <= self.n_head <= 128):
            raise ValueError(f"n_head must be in [1, 128], got {self.n_head}")
        if not (1 <= self.n_kv_head <= self.n_head):
            raise ValueError(f"n_kv_head must be in [1, n_head={self.n_head}]")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if not (64 <= self.n_embd <= 8_192):
            raise ValueError(f"n_embd must be in [64, 8192], got {self.n_embd}")
        if not (0.0 <= self.dropout <= 0.9):
            raise ValueError(f"dropout must be in [0.0, 0.9], got {self.dropout}")

        # ── training validation ───────────────────────────────────────────
        if not (1 <= self.batch_size <= 512):
            raise ValueError(f"batch_size must be in [1, 512], got {self.batch_size}")
        if not (0 < self.learning_rate <= 1e-2):
            raise ValueError(f"learning_rate must be in (0, 1e-2], got {self.learning_rate}")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError(f"weight_decay must be in [0, 1], got {self.weight_decay}")
        if not (0 < self.grad_norm_clip <= 10):
            raise ValueError(f"grad_norm_clip must be in (0, 10], got {self.grad_norm_clip}")

        # ── generation validation ─────────────────────────────────────────
        if not (1 <= self.max_generation_tokens <= 8_192):
            raise ValueError(f"max_generation_tokens must be in [1, 8192]")
        if self.temperature_min >= self.temperature_max:
            raise ValueError("temperature_min must be < temperature_max")

    # ── Preset factory methods ─────────────────────────────────────────────

    @classmethod
    def tiny(cls) -> "LLMConfig":
        """~15M params — CPU / 2 GB RAM."""
        return cls(
            vocab_size=5_000, block_size=256, n_layer=4, n_head=4,
            n_kv_head=4, n_embd=256, batch_size=8,
            gradient_accumulation_steps=8,
        )

    @classmethod
    def small(cls) -> "LLMConfig":
        """~38M params — default, balanced."""
        return cls(
            vocab_size=5_000, block_size=512, n_layer=6, n_head=6,
            n_kv_head=6, n_embd=384, batch_size=16,
            gradient_accumulation_steps=4,
        )

    @classmethod
    def medium(cls) -> "LLMConfig":
        """~124M params — high-end PC / 8 GB VRAM."""
        return cls(
            vocab_size=32_000, block_size=1_024, n_layer=12, n_head=12,
            n_kv_head=12, n_embd=768, batch_size=8,
            gradient_accumulation_steps=8, use_bf16=True,
        )

    @classmethod
    def model_1b(cls) -> "LLMConfig":
        """~1.1B params — 16 GB VRAM for training, 4 GB for inference (fp16)."""
        return cls(
            vocab_size=32_000, block_size=2_048, n_layer=22, n_head=16,
            n_kv_head=8, n_embd=2_048, intermediate_size=5_632,
            batch_size=4, gradient_accumulation_steps=16,
            gradient_checkpointing=True, use_bf16=True,
        )

    @classmethod
    def model_7b(cls) -> "LLMConfig":
        """
        ~6.7B params — LLaMA-style 7B.
        Requirements:
          Training  : 2× 24 GB VRAM (gradient checkpointing + bf16) or 80 GB single
          Inference : 14 GB VRAM (bf16) / 7 GB (int8)
        """
        return cls(
            vocab_size=32_000, block_size=2_048, n_layer=32, n_head=32,
            n_kv_head=32,        # full attention; GQA not needed at 7B
            n_embd=4_096, intermediate_size=11_008,
            rope_theta=10_000.0,
            batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            use_bf16=True,
            use_flash_attention=True,
            streaming=True,
            use_mmap=True,
            warmup_iters=2_000,
            max_iters=100_000,
            min_lr=3e-5,
        )


@dataclass
class SecurityConfig:
    """Security rules — path traversal, sanitisation, rate-limiting."""

    allow_absolute_path: bool  = False
    allowed_base_dirs:   tuple = ('data/', 'models/', 'checkpoints/')
    max_input_length:    int   = 10_000_000   # 10 MB text
    sanitize_inputs:     bool  = True
    max_requests_per_minute: int = 60
    enable_content_filter:   bool = True
    enable_audit_logging:    bool = True
    log_level: str = "INFO"

    def validate_file_path(self, file_path: str) -> bool:
        from pathlib import Path
        try:
            path = Path(file_path).resolve()
            if ".." in str(file_path):
                return False
            if path.is_absolute() and not self.allow_absolute_path:
                path_str = str(path)
                return any(path_str.startswith(str(Path(b).resolve()))
                           for b in self.allowed_base_dirs)
            return True
        except Exception:
            return False