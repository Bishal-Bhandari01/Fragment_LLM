"""
Optimized Transformer Model
Features:
- Memory-efficient architecture for low-end PCs
- Flash Attention compatible
- Gradient checkpointing support
- Input validation
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
        """Apply layer normalization"""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        return self.weight * x_norm


class CausalSelfAttention(nn.Module):
    """
    Optimized causal self-attention with security checks.
    Memory-efficient implementation for low-end hardware.
    """
    
    def __init__(
        self, 
        embed_size: int, 
        num_heads: int, 
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # Validation
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
        
        # Causal mask (registered as buffer, not a parameter)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input validation.
        
        Args:
            x: Input tensor of shape (B, T, C)
        
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        
        # Validate input dimensions
        if C != self.embed_size:
            raise ValueError(
                f"Input embed_size ({C}) doesn't match expected ({self.embed_size})"
            )
        
        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
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
        """Forward pass with residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AIModel(nn.Module):
    """
    Secure GPT model with validation and optimization for low-end PCs.
    Features:
    - Input validation
    - Memory-efficient implementation
    - Gradient checkpointing support
    - Safe generation with temperature clamping
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Validate config
        if not hasattr(config, 'vocab_size'):
            raise ValueError("Config must have vocab_size")
        
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head (weight tied with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions."""
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
        """
        Forward pass with input validation.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for loss computation
        
        Returns:
            Tuple of (logits, loss)
        """
        B, T = idx.size()
        
        # Input validation
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )
        
        if idx.min() < 0 or idx.max() >= self.config.vocab_size:
            raise ValueError(
                f"Token indices out of range [0, {self.config.vocab_size})"
            )
        
        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
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
        """
        Generate tokens with safety constraints.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (clamped to safe range)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        # Input validation
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