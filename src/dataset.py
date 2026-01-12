"""
Text dataset for training with path validation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Text dataset with security checks."""
    
    def __init__(
        self, 
        text_file: str, 
        tokenizer, 
        block_size: int = 512,
        max_file_size_mb: int = 500,
        allowed_base_dirs: tuple = ('data/',)
    ):
        """Initialize dataset with validation."""
        # validate path
        self.text_file = self._validate_path(text_file, allowed_base_dirs)
        
        logger.info(f"Validated: {self.text_file}")
        
        # check file size
        file_size_mb = self.text_file.stat().st_size / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_file_size_mb}MB)"
            )
        
        # validate block size
        if not 1 <= block_size <= 2048:
            raise ValueError(f"block_size must be in [1, 2048], got {block_size}")
        
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # load and tokenize
        logger.info(f"Loading dataset from {self.text_file}")
        try:
            with open(self.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            raise IOError(f"Failed to read file: {e}")
        
        # validate content
        if not text:
            raise ValueError("Dataset file is empty")
        
        logger.info("Tokenizing...")
        self.tokens = tokenizer.encode(text)
        
        logger.info(f"Tokens: {len(self.tokens)}")
        
        if len(self.tokens) < block_size + 1:
            raise ValueError(
                f"Dataset too small: {len(self.tokens)} tokens < {block_size + 1}"
            )
        
        logger.info(f"Dataset loaded: {len(self.tokens)} tokens")
    
    @staticmethod
    def _validate_path(filepath: str, allowed_base_dirs: tuple) -> Path:
        """Validate file path to prevent directory traversal."""
        try:
            path = Path(filepath).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")
        
        # check file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")
        
        # check for path traversal
        if '..' in str(filepath):
            raise ValueError("Path traversal detected")
        
        # verify path is in allowed directories
        path_str = str(path)
        if allowed_base_dirs:
            is_allowed = any(
                path_str.startswith(str(Path(base).resolve())) 
                for base in allowed_base_dirs
            )
            if not is_allowed and not path.is_absolute():
                pass  # allow relative paths in current dir
            elif not is_allowed:
                raise ValueError(
                    f"Path not in allowed directories: {allowed_base_dirs}"
                )
        
        return path
    
    def __len__(self) -> int:
        """Return number of valid sequences."""
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        # validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Extract sequence
        chunk = self.tokens[idx:idx + self.block_size + 1]
        
        # Create input and target tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def create_dataloader(
    file_path: str,
    tokenizer,
    batch_size: int = 16,
    block_size: int = 512,
    num_workers: int = 0,  # 0 for low-end PCs
    max_file_size_mb: int = 500,
    allowed_base_dirs: tuple = ('data/',)
) -> DataLoader:
    """
    Create a secure DataLoader with validation.
    
    Args:
        file_path: Path to text file
        tokenizer: Trained tokenizer
        batch_size: Batch size (optimized for low-end PCs)
        block_size: Sequence length
        num_workers: Number of data loading workers
        max_file_size_mb: Maximum file size
        allowed_base_dirs: Allowed directory prefixes
    
    Returns:
        Configured DataLoader
    """
    # Parameter validation
    if not 1 <= batch_size <= 256:
        raise ValueError(f"batch_size must be in [1, 256], got {batch_size}")
    
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    
    # Create dataset
    dataset = TextDataset(
        text_file=file_path,
        tokenizer=tokenizer,
        block_size=block_size,
        max_file_size_mb=max_file_size_mb,
        allowed_base_dirs=allowed_base_dirs
    )
    logger.info(f"Loaded Datasets: {dataset}")
    
    # Create dataloader with memory efficiency
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Only if GPU available
        drop_last=True  # Ensure consistent batch sizes
    )