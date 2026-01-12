"""
Transformer model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    
    def __init__(
        self,
        features: int,
        eps: float = 1e-6,
        bias: bool = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        return self.weight * x_norm


class CausalSelfAttention(nn.Module):
    """Causal self-attention with multi-head attention."""
    
    def __init__(
        self, 
        embed_size: int, 
        num_heads: int, 
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # sanity check
        if embed_size % num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.embed_size = embed_size
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # quick validation
        if C != self.embed_size:
            raise ValueError(
                f"Input embed_size ({C}) doesn't match expected ({self.embed_size})"
            )
        
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)
        
        # reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # apply attention to values
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(
            embed_size=config.n_embd,
            num_heads=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AIModel(nn.Module):
    """GPT-style transformer model for text generation."""
    
    def __init__(self, config):
        super().__init__()
        
        # basic validation
        if not hasattr(config, 'vocab_size'):
            raise ValueError("Config must have vocab_size")
        
        self.config = config
        
        # embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # LM head with weight tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        # init weights
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Init weights like GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        
        # validate inputs
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )
        
        if idx.min() < 0 or idx.max() >= self.config.vocab_size:
            raise ValueError(
                f"Token indices out of range [0, {self.config.vocab_size})"
            )
        
        # get position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # run through transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # compute loss if we have targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text tokens autoregressively."""
        # validate inputs
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        
        if max_new_tokens > self.config.max_generation_tokens:
            logger.warning(
                f"Clamping max_new_tokens from {max_new_tokens} to "
                f"{self.config.max_generation_tokens}"
            )
            max_new_tokens = self.config.max_generation_tokens
        
        # Clamp temperature to safe range
        temperature = max(
            self.config.temperature_min,
            min(temperature, self.config.temperature_max)
        )
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
                       idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Optional nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def estimate_memory_usage(self) -> dict:
        """Estimate model memory usage for monitoring."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'parameters_mb': param_bytes / (1024 ** 2),
            'buffers_mb': buffer_bytes / (1024 ** 2),
            'total_mb': (param_bytes + buffer_bytes) / (1024 ** 2)
        }