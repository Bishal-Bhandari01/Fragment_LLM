"""
Fragment_LLM — Modern GPT-style decoder-only transformer.

Architecture upgrades over original (required for 7B-scale):
  - RMSNorm          : faster, no mean-subtraction (standard in LLaMA/Mistral)
  - RoPE             : rotary position embeddings; no learned wpe matrix
  - Grouped Query Attention (GQA): n_kv_head ≤ n_head → smaller KV-cache
  - Flash Attention  : F.scaled_dot_product_attention (PyTorch ≥ 2.0, no extra deps)
  - SwiGLU MLP       : SiLU(gate) × up → down (better than GELU expansion)
  - Gradient checkpointing: recompute activations to halve activation memory
  - Weight tying     : embed_tokens.weight == lm_head.weight
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


# ── Normalization ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt for numerical stability on fp16/bf16
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── Rotary Position Embeddings ─────────────────────────────────────────────────

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """
    Pre-compute complex RoPE frequencies: shape (max_seq_len, head_dim // 2).
    Stored as a buffer in AIModel; sliced per forward call.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                          # (T, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)      # complex64


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q, k: (B, T, n_head, head_dim) — float
        freqs_cis: (T, head_dim // 2) — complex
    """
    # Cast to float32 for complex ops, then back
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Broadcast over batch and head dims: (1, T, 1, head_dim//2)
    fc = freqs_cis.unsqueeze(0).unsqueeze(2)

    q_out = torch.view_as_real(q_ * fc).flatten(3).type_as(q)
    k_out = torch.view_as_real(k_ * fc).flatten(3).type_as(k)
    return q_out, k_out


# ── Grouped-Query Attention ────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
      - Grouped Query Attention (n_kv_head ≤ n_head)
      - RoPE applied to Q and K
      - Flash Attention via F.scaled_dot_product_attention
      - Falls back to manual dot-product attention when flash is disabled
    """

    def __init__(self, config) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        assert config.n_head % config.n_kv_head == 0, \
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})"

        self.n_head     = config.n_head
        self.n_kv_head  = config.n_kv_head
        self.head_dim   = config.n_embd // config.n_head
        self.n_embd     = config.n_embd
        self.flash      = config.use_flash_attention
        self.groups     = config.n_head // config.n_kv_head  # repetitions for GQA

        # Separate projections (no combined QKV) — clearer for GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd,                    bias=False)

        self.dropout = config.dropout

        # Causal mask — only needed for manual attention path
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Projections
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: expand K/V to match Q head count
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Attention
        if self.flash:
            # Uses FlashAttention kernels when available (PyTorch ≥ 2.0)
            attn_drop = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, is_causal=True)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.o_proj(y)


# ── SwiGLU Feed-Forward ────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    SwiGLU feed-forward: output = down(SiLU(gate(x)) × up(x))
    No dropout — standard for large-model pre-training.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ──────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional gradient checkpointing."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn      = CausalSelfAttention(config)
        self.ffn_norm  = RMSNorm(config.n_embd)
        self.mlp       = MLP(config)
        self._grad_ckpt = config.gradient_checkpointing

    # Separate method so checkpoint can wrap only the attention forward
    def _attn_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return self.attn(self.attn_norm(x), freqs_cis)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        if self._grad_ckpt and self.training:
            attn_out = torch.utils.checkpoint.checkpoint(
                self._attn_forward, x, freqs_cis, use_reentrant=False
            )
        else:
            attn_out = self._attn_forward(x, freqs_cis)

        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))
        return x


# ── Main Model ─────────────────────────────────────────────────────────────────

class AIModel(nn.Module):
    """
    GPT-style decoder-only transformer (modern LLaMA-style architecture).

    Key differences from original Fragment_LLM:
      - No learned position embedding (wpe); RoPE handles position.
      - RMSNorm instead of LayerNorm.
      - SwiGLU instead of GELU MLP.
      - Flash Attention + GQA.
      - Gradient checkpointing per block.
    """

    def __init__(self, config) -> None:
        super().__init__()

        if not hasattr(config, 'vocab_size'):
            raise ValueError("Config must have vocab_size")
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm   = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and output share weights → fewer params, better perf
        self.embed_tokens.weight = self.lm_head.weight

        # Pre-compute RoPE frequencies (2× block_size for safety during generation)
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.block_size * 2, config.rope_theta),
        )

        self.apply(self._init_weights)

        # Scale residual projection weights (GPT-2 convention)
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"AIModel initialised: {n_params:,} parameters "
                    f"({n_params / 1e9:.2f}B)")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        if idx.min() < 0 or idx.max() >= self.config.vocab_size:
            raise ValueError(f"Token indices out of range [0, {self.config.vocab_size})")

        # Token embeddings
        x = self.embed_tokens(idx)                    # (B, T, n_embd)

        # RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]               # (T, head_dim//2) complex

        for block in self.blocks:
            x = block(x, freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Auto-regressive token generation with top-k / nucleus sampling."""

        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be ≥ 1, got {max_new_tokens}")

        cfg = self.config
        max_new_tokens = min(max_new_tokens, cfg.max_generation_tokens)
        temperature = max(cfg.temperature_min, min(temperature, cfg.temperature_max))

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Nucleus (top-p) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits.scatter_(1, sorted_idx, remove.float() * float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

    def estimate_memory_usage(self) -> dict:
        param_mb  = sum(p.numel() * p.element_size() for p in self.parameters())  / 1024**2
        buffer_mb = sum(b.numel() * b.element_size() for b in self.buffers())     / 1024**2
        return {
            "parameters_mb": round(param_mb, 2),
            "buffers_mb":    round(buffer_mb, 2),
            "total_mb":      round(param_mb + buffer_mb, 2),
            "total_gb":      round((param_mb + buffer_mb) / 1024, 3),
        }