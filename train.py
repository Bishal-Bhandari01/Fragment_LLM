#!/usr/bin/env python3
"""
train.py — Fragment_LLM training entry point.

Single-GPU:
    python train.py --preset small

Multi-GPU (all GPUs on one machine):
    torchrun --standalone --nproc_per_node=4 train.py --preset 7b

Resume from checkpoint:
    python train.py --preset 7b --resume checkpoints/ckpt_epoch001.pt

Presets:  tiny | small | medium | 1b | 7b
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# ── allow `from src.X import ...` regardless of cwd ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config   import LLMConfig
from src.dataset  import create_dataloader
from src.model    import AIModel
from src.tokenizer import SimpleTokenizer
from src.trainer  import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── DDP helpers ───────────────────────────────────────────────────────────────

def _setup_ddp() -> tuple[int, int, int]:
    """Initialise DDP if torchrun launched us.  Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1                          # single-GPU / CPU run
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ── Preset → config map ───────────────────────────────────────────────────────

_PRESETS = {
    "tiny":   LLMConfig.tiny,
    "small":  LLMConfig.small,
    "medium": LLMConfig.medium,
    "1b":     LLMConfig.model_1b,
    "7b":     LLMConfig.model_7b,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
        if is_main:
            logger.warning("CUDA not available — training on CPU (very slow for large models)")

    # ── Config ────────────────────────────────────────────────────────────
    if args.preset:
        config = _PRESETS[args.preset]()
        if is_main:
            logger.info(f"Using preset: {args.preset!r}")
    else:
        config = LLMConfig(
            vocab_size        = args.vocab_size,
            block_size        = args.block_size,
            n_layer           = args.n_layer,
            n_head            = args.n_head,
            n_kv_head         = args.n_kv_head,
            n_embd            = args.n_embd,
            dropout           = args.dropout,
            batch_size        = args.batch_size,
            gradient_accumulation_steps = args.grad_accum_steps,
            learning_rate     = args.learning_rate,
            warmup_iters      = args.warmup_iters,
            max_iters         = args.max_iters,
            use_amp           = args.use_amp and device != "cpu",
            use_bf16          = args.use_bf16,
            gradient_checkpointing = args.grad_ckpt,
            streaming         = args.streaming,
        )

    config.ddp = (world_size > 1)

    # ── Directories ───────────────────────────────────────────────────────
    for d in ("data/processed", "models", "checkpoints"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer_path = Path(args.tokenizer_path)
    if tokenizer_path.exists() and not args.retrain_tokenizer:
        if is_main:
            logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    else:
        if not is_main:
            dist.barrier()          # wait for rank 0 to train tokenizer
        else:
            logger.info("Training new tokenizer …")
            train_file = Path(args.train_file)
            if not train_file.exists():
                raise FileNotFoundError(f"Train file not found: {train_file}")

            # Stream a sample for tokenizer training (max 300 MB)
            MAX_CHARS = 300_000_000
            chars = []
            total = 0
            with open(train_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    chars.append(line)
                    total += len(line)
                    if total >= MAX_CHARS:
                        break
            sample_text = "".join(chars)[:MAX_CHARS]

            tokenizer = SimpleTokenizer(max_vocab_size=config.vocab_size)
            tokenizer.train(sample_text, vocab_size=config.vocab_size, max_text_length=MAX_CHARS)
            tokenizer.save(str(tokenizer_path))
            logger.info(f"Tokenizer saved → {tokenizer_path}")
            dist.barrier() if world_size > 1 else None

        if not is_main:
            tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    config.vocab_size = len(tokenizer.vocab)
    if is_main:
        logger.info(f"Vocabulary size: {config.vocab_size:,}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    dataset_mode = "stream" if config.streaming else ("mmap" if config.use_mmap else "memory")

    if is_main:
        logger.info(f"Dataset mode: {dataset_mode!r}")

    train_loader = create_dataloader(
        args.train_file, tokenizer,
        batch_size        = config.batch_size,
        block_size        = config.block_size,
        num_workers       = args.num_workers,
        allowed_base_dirs = (str(Path(args.train_file).parent),),
        mode              = dataset_mode,
        rank              = rank,
        world_size        = world_size,
    )
    val_loader = create_dataloader(
        args.val_file, tokenizer,
        batch_size        = config.batch_size,
        block_size        = config.block_size,
        num_workers       = args.num_workers,
        allowed_base_dirs = (str(Path(args.val_file).parent),),
        mode              = dataset_mode,
        rank              = rank,
        world_size        = world_size,
    )

    if is_main:
        logger.info(f"Train batches: {len(train_loader):,}  Val batches: {len(val_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Initialising model …")
    model = AIModel(config)

    if is_main:
        mem = model.estimate_memory_usage()
        logger.info(f"Model memory estimate: {mem['total_gb']:.2f} GB (weights only)")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model              = model,
        train_dataloader   = train_loader,
        val_loader         = val_loader,
        config             = config,
        device             = device,
        checkpoint_dir     = args.checkpoint_dir,
        save_every_n_steps = args.save_every,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── W&B (optional) ────────────────────────────────────────────────────
    if args.use_wandb and is_main:
        try:
            import wandb
            wandb.init(project="fragment-llm", config=vars(config))
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")

    # ── Train ─────────────────────────────────────────────────────────────
    if is_main:
        logger.info(f"Starting training for {args.epochs} epoch(s) …")
    trainer.train(epochs=args.epochs)

    # ── Save final model ──────────────────────────────────────────────────
    if is_main:
        out = Path("models/final_model.pt")
        torch.save(model.state_dict(), out)
        logger.info(f"Final model saved → {out}")

    _cleanup_ddp()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fragment_LLM trainer — supports tiny through 7B models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Preset (overrides all architecture flags) ──────────────────────────
    p.add_argument("--preset", choices=list(_PRESETS), default=None,
                   help="Named model preset (tiny / small / medium / 1b / 7b)")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--train-file",  default="data/processed/train.txt")
    p.add_argument("--val-file",    default="data/processed/val.txt")
    p.add_argument("--tokenizer-path", default="models/tokenizer.json")
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safest for most OSes)")
    p.add_argument("--streaming",   action="store_true",
                   help="Stream dataset instead of memory-mapping (use for very large data)")

    # ── Architecture (ignored if --preset given) ───────────────────────────
    p.add_argument("--vocab-size",  type=int,   default=10_000)
    p.add_argument("--block-size",  type=int,   default=512)
    p.add_argument("--n-layer",     type=int,   default=6)
    p.add_argument("--n-head",      type=int,   default=6)
    p.add_argument("--n-kv-head",   type=int,   default=6)
    p.add_argument("--n-embd",      type=int,   default=384)
    p.add_argument("--dropout",     type=float, default=0.0)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--grad-accum-steps", type=int,   default=4)
    p.add_argument("--learning-rate",    type=float, default=3e-4)
    p.add_argument("--warmup-iters",     type=int,   default=500)
    p.add_argument("--max-iters",        type=int,   default=10_000)
    p.add_argument("--use-amp",    action="store_true",  default=True)
    p.add_argument("--no-amp",     dest="use_amp",  action="store_false")
    p.add_argument("--use-bf16",   action="store_true",  default=True)
    p.add_argument("--grad-ckpt",  action="store_true",
                   help="Gradient checkpointing (saves memory, slower per step)")

    # ── Checkpointing ─────────────────────────────────────────────────────
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--save-every",     type=int, default=1_000,
                   help="Save checkpoint every N gradient steps")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint to resume from")

    # ── Logging ───────────────────────────────────────────────────────────
    p.add_argument("--use-wandb", action="store_true")

    args = p.parse_args()
    main(args)