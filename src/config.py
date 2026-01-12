from dataclasses import dataclass

@dataclass
class LLMConfig:
    """
    Model configuration with validation.
    """
    # Model architecture
    vocab_size: int  = 5000
    block_size: int = 256  # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

    # Regularization
    dropout: float = 0.1
    bias: bool = True  # use bias in linear layers?

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # effective batch = 64
    learning_rate: float = 3e-4
    max_iters: int = 10000
    weight_decay: float = 1e-1
    grad_norm_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Security limits to prevent resource exhaustion
    max_file_size_mb: int = 500
    max_sequence_length: int = 2048
    allowed_file_extension: tuple = ('.txt', '.parquet')

    # Generation safety
    max_generation_tokens: int = 512
    temperature_min: float = 0.1
    temperature_max: float = 2.0

    def __post_init__(self):
        """Validate configuration parameters."""
        # Architecture validation
        if self.vocab_size < 256 or self.vocab_size > 100000:
            raise ValueError(f"vocab_size must be in [256, 100000], got {self.vocab_size}")
        
        if self.block_size < 128 or self.block_size > self.max_sequence_length:
            raise ValueError(f"block_size must be in [128, {self.max_sequence_length}]")
        
        if self.n_layer < 1 or self.n_layer > 48:
            raise ValueError(f"n_layer must be in [1, 48], got {self.n_layer}")
        
        if self.n_head < 1 or self.n_head > 32:
            raise ValueError(f"n_head must be in [1, 32], got {self.n_head}")
        
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        
        if self.n_embd < 64 or self.n_embd > 2048:
            raise ValueError(f"n_embd must be in [64, 2048], got {self.n_embd}")
        
        if not 0.0 <= self.dropout <= 0.9:
            raise ValueError(f"dropout must be in [0.0, 0.9], got {self.dropout}")
        
        # Training validation
        if self.batch_size < 1 or self.batch_size > 256:
            raise ValueError(f"batch_size must be in [1, 256], got {self.batch_size}")
        
        if self.learning_rate <= 0 or self.learning_rate > 1e-2:
            raise ValueError(f"learning_rate must be in (0, 1e-2), got {self.learning_rate}")
        
        if self.weight_decay < 0 or self.weight_decay > 1:
            raise ValueError(f"weight_decay must be in [0, 1], got {self.weight_decay}")
        
        if self.grad_norm_clip <= 0 or self.grad_norm_clip > 10:
            raise ValueError(f"grad_norm_clip must be in (0, 10], got {self.grad_norm_clip}")
        
        # Generation safety validation
        if self.max_generation_tokens < 1 or self.max_generation_tokens > 2048:
            raise ValueError(f"max_generation_tokens must be in [1, 2048], got {self.max_generation_tokens}")
        
        if not self.temperature_min < self.temperature_max:
            raise ValueError("temperature_min must be less than temperature_max")
        
@dataclass
class SecurityConfig:
    """
    Security rules and constraints for the system.
    """
    # File system security - prevent path traversal
    allow_absolute_path: bool = False
    allowed_base_dirs: bool = ('data/', 'models/', 'checkpoints/')

    # Input sanitization
    max_input_length: int = 10_000_000  # 10MB text
    sanitization_inputs: bool = True

    # Rate limiting
    max_requests_per_minute:int = 60

    # Model safety
    enable_content_filter: bool = True
    block_harmful_generation: bool = True
    
    # Logging
    enable_audit_logging: bool = True
    log_level: str  ="INFO"

    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path to prevent directory traversal attacks."""
        import os
        from pathlib import Path

        try:
            path = Path(file_path).resolve()

            # check if absolute paths are allowed
            if path.is_absolute() and not self.allow_absolute_path:
                return False
            
            # check for path traversal
            if ".." in str(path):
                return False
            
            # verify path is within allowed directories
            path_str = str(path)
            if not any(path_str.startswith(base) for base in self.allowed_base_dirs):
                # if not in allowed dirs, check if it's a relative safe path
                if path.is_absolute():
                    return False
            
            return True
        except Exception as e:
            return False
