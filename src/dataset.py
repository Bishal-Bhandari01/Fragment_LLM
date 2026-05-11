"""
Dataset handling for Fragment_LLM — supports any size, any format.

Three loading modes (pick based on dataset size):

  MemoryMappedDataset (default / recommended)
    Tokenise once → write uint16/uint32 .bin file → numpy memmap.
    Works for datasets of any size; only a 4 KB page lives in RAM at a time.
    Fast random-access shuffling with zero memory overhead.

  StreamingTextDataset
    Read the raw file in overlapping windows — never fully loads it.
    Useful when you haven't yet pre-tokenised or disk space is tight.

  TextDataset (original, unchanged for small experiments)
    Loads everything into RAM — fine up to a few hundred MB.

Usage:
    # Recommended for large / production datasets
    bin_path = MemoryMappedDataset.build(
        "data/processed/train.txt", tokenizer, block_size=2048
    )
    dataset = MemoryMappedDataset(bin_path, block_size=2048)
"""

import io
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_path(filepath: str, allowed_base_dirs: tuple) -> Path:
    """Resolve and validate a file path (CWE-22 mitigation)."""
    if ".." in filepath:
        raise ValueError(f"Path traversal detected in: {filepath!r}")
    try:
        path = Path(filepath).resolve()
    except Exception as exc:
        raise ValueError(f"Invalid file path: {exc}") from exc
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Not a file: {filepath}")
    if allowed_base_dirs:
        allowed = [str(Path(b).resolve()) for b in allowed_base_dirs]
        if not any(str(path).startswith(a) for a in allowed):
            raise ValueError(f"Path not in allowed dirs {allowed_base_dirs}: {filepath}")
    return path


# ── Memory-Mapped Dataset (primary for large / production data) ────────────────

class MemoryMappedDataset(Dataset):
    """
    Memory-mapped binary token dataset.

    The .bin file contains raw token IDs (uint16 or uint32) written
    sequentially.  numpy.memmap gives O(1) random access with the OS
    page cache doing all the heavy lifting — the full dataset never
    lives in Python-side RAM.

    Workflow:
        bin_path = MemoryMappedDataset.build(txt_file, tokenizer, block_size)
        ds = MemoryMappedDataset(bin_path, block_size)
    """

    @staticmethod
    def build(
        source: str,
        tokenizer,
        block_size: int,
        output_bin: Optional[str] = None,
        allowed_base_dirs: tuple = ("data/",),
        chunk_chars: int = 10_000_000,   # 10 MB text chunks
    ) -> Path:
        """
        Tokenise *source* (any text file) and write a binary token file.

        Returns the path to the .bin file.
        This is idempotent: if the .bin already exists and is non-empty,
        it is reused without re-tokenising.
        """
        src = _resolve_path(source, allowed_base_dirs)

        if output_bin is None:
            output_bin = str(src.with_suffix(".bin"))
        out = Path(output_bin)

        if out.exists() and out.stat().st_size > 0:
            logger.info(f"Reusing existing token file: {out}")
            return out

        # Decide dtype
        vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 200_000
        dtype_np = np.uint16 if vocab_size <= 65_535 else np.uint32
        dtype_sz = 2 if dtype_np == np.uint16 else 4
        logger.info(f"Building mmap token file → {out}  (dtype={dtype_np.__name__})")

        total_tokens = 0
        with open(out, "wb") as fout, open(src, "r", encoding="utf-8", errors="replace") as fin:
            buf = ""
            while True:
                chunk = fin.read(chunk_chars)
                if not chunk:
                    break
                buf += chunk
                # Keep a tail so we don't cut a multi-byte character
                tokens = tokenizer.encode(buf)
                # Write all but the last block_size tokens (keep context)
                safe = max(0, len(tokens) - block_size)
                if safe:
                    arr = np.array(tokens[:safe], dtype=dtype_np)
                    fout.write(arr.tobytes())
                    total_tokens += safe
                buf_tokens = tokens[safe:]
                buf = tokenizer.decode(buf_tokens) if buf_tokens else ""

            # Flush remainder
            if buf:
                tokens = tokenizer.encode(buf)
                if tokens:
                    arr = np.array(tokens, dtype=dtype_np)
                    fout.write(arr.tobytes())
                    total_tokens += len(tokens)

        logger.info(f"Written {total_tokens:,} tokens to {out} "
                    f"({out.stat().st_size / 1024**2:.1f} MB)")
        return out

    def __init__(
        self,
        bin_file: Union[str, Path],
        block_size: int,
        dtype: str = "uint16",
    ) -> None:
        self.block_size = block_size
        path = Path(bin_file)
        if not path.exists():
            raise FileNotFoundError(f"Binary token file not found: {bin_file}")
        self.data = np.memmap(path, dtype=dtype, mode="r")
        n = len(self.data)
        if n < block_size + 1:
            raise ValueError(f"Token file too small ({n}) for block_size={block_size}")
        logger.info(f"MemoryMappedDataset: {n:,} tokens, "
                    f"{len(self):,} sequences of length {block_size}")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ── Streaming Dataset (no tokenise-ahead, pure streaming) ─────────────────────

class StreamingTextDataset(Dataset):
    """
    Streams a large text file in overlapping windows.
    Tokenises on-the-fly in fixed-size character chunks.

    Best for: one-pass training on very large files with limited disk.
    Trade-off: slower per-batch than mmap (re-tokenises each access).
    """

    def __init__(
        self,
        text_file: str,
        tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        allowed_base_dirs: tuple = ("data/",),
        chunk_chars: int = 5_000_000,
    ) -> None:
        self.path       = _resolve_path(text_file, allowed_base_dirs)
        self.tokenizer  = tokenizer
        self.block_size = block_size
        self.stride     = stride or block_size
        self.chunk_chars = chunk_chars

        # Single pass to build an index of (char_offset, token_offset) pairs
        # We store the cumulative token count per 1 MB chunk to enable __len__
        logger.info(f"StreamingTextDataset: indexing {self.path} …")
        self._total_tokens = self._count_tokens()
        logger.info(f"StreamingTextDataset: ~{self._total_tokens:,} tokens "
                    f"→ {len(self):,} sequences")

    def _count_tokens(self) -> int:
        total = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(self.chunk_chars)
                if not chunk:
                    break
                total += len(self.tokenizer.encode(chunk))
        return total

    def __len__(self) -> int:
        return max(0, (self._total_tokens - self.block_size) // self.stride)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Seek to approximate position and tokenise enough for one block
        char_approx = idx * self.stride * 4  # ~4 chars/token heuristic
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(min(char_approx, max(0, self.path.stat().st_size - 1)))
            chunk = f.read(self.chunk_chars)
        tokens = self.tokenizer.encode(chunk)
        if len(tokens) < self.block_size + 1:
            tokens = tokens + [0] * (self.block_size + 1 - len(tokens))
        x = torch.tensor(tokens[:self.block_size],    dtype=torch.long)
        y = torch.tensor(tokens[1:self.block_size+1], dtype=torch.long)
        return x, y


# ── Original In-Memory Dataset (small datasets, backward compatible) ───────────

class TextDataset(Dataset):
    """Loads an entire text file into RAM. Use for datasets < ~1 GB."""

    def __init__(
        self,
        text_file: str,
        tokenizer,
        block_size: int = 512,
        max_file_size_mb: int = 0,         # 0 = no limit
        allowed_base_dirs: tuple = ("data/",),
    ) -> None:
        path = _resolve_path(text_file, allowed_base_dirs)

        if max_file_size_mb > 0:
            mb = path.stat().st_size / 1024**2
            if mb > max_file_size_mb:
                raise ValueError(f"File {mb:.1f} MB exceeds limit {max_file_size_mb} MB")

        if not (1 <= block_size <= 8192):
            raise ValueError(f"block_size must be in [1, 8192], got {block_size}")

        self.block_size = block_size

        logger.info(f"Loading {path} into RAM …")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text:
            raise ValueError("Dataset file is empty")

        self.tokens = tokenizer.encode(text)
        if len(self.tokens) < block_size + 1:
            raise ValueError(
                f"Too few tokens ({len(self.tokens)}) for block_size={block_size}"
            )
        logger.info(f"TextDataset: {len(self.tokens):,} tokens, {len(self):,} sequences")

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


# ── DataLoader factory ─────────────────────────────────────────────────────────

def create_dataloader(
    file_path: str,
    tokenizer,
    batch_size: int        = 16,
    block_size: int        = 512,
    num_workers: int       = 0,
    allowed_base_dirs: tuple = ("data/",),
    mode: str              = "auto",  # "mmap" | "stream" | "memory" | "auto"
    max_file_size_mb: int  = 0,
    # DDP
    rank: int              = 0,
    world_size: int        = 1,
) -> DataLoader:
    """
    Create a DataLoader with automatic dataset mode selection.

    mode="auto":
        file < 512 MB  → TextDataset (in-memory)
        otherwise      → MemoryMappedDataset (mmap binary)
    """
    from torch.utils.data.distributed import DistributedSampler

    if not (1 <= batch_size <= 512):
        raise ValueError(f"batch_size must be in [1, 512], got {batch_size}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    size_mb = path.stat().st_size / 1024**2

    # Determine mode
    if mode == "auto":
        mode = "memory" if size_mb <= 512 else "mmap"

    logger.info(f"DataLoader mode={mode!r} for {path.name} ({size_mb:.1f} MB)")

    if mode == "mmap":
        dtype = "uint16" if len(getattr(tokenizer, 'vocab', range(65536))) <= 65535 else "uint32"
        bin_path = MemoryMappedDataset.build(
            file_path, tokenizer, block_size, allowed_base_dirs=allowed_base_dirs
        )
        dataset = MemoryMappedDataset(bin_path, block_size, dtype=dtype)
    elif mode == "stream":
        dataset = StreamingTextDataset(
            file_path, tokenizer, block_size, allowed_base_dirs=allowed_base_dirs
        )
    else:  # memory
        dataset = TextDataset(
            file_path, tokenizer, block_size,
            max_file_size_mb=max_file_size_mb,
            allowed_base_dirs=allowed_base_dirs,
        )

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )