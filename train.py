#!/usr/bin/env python3
"""
Secure Training Script - Main Entry Point
"""
import torch
from pathlib import Path
import logging
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import LLMConfig
from src.tokenizer import SimpleTokenizer
from src.dataset import create_dataloader
from src.model import AIModel
from src.trainer import Trainer  # Note: Using Trainer, not SecureTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    logger.info("="*60)
    logger.info("SECURE LLM TRAINING PIPELINE")
    logger.info("="*60)
    
    # Setup directories
    Path('data/processed').mkdir(exist_ok=True, parents=True)
    Path('models').mkdir(exist_ok=True, parents=True)
    Path('checkpoints').mkdir(exist_ok=True, parents=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Config
    config = LLMConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        use_amp=args.use_amp and device == 'cuda'
    )
    
    # Tokenizer
    tokenizer_path = Path('tokenizer.json')
    if tokenizer_path.exists() and not args.retrain_tokenizer:
        logger.info("Loading tokenizer...")
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    else:
        logger.info("Training tokenizer...")
        with open('data/processed/train_tiny.txt', 'r') as f:
            text = f.read()
        tokenizer = SimpleTokenizer(max_vocab_size=args.vocab_size)
        tokenizer.train(text, vocab_size=args.vocab_size)
        tokenizer.save(str(tokenizer_path))
    
    config.vocab_size = len(tokenizer.vocab)
    logger.info(f"Vocabulary size: {config.vocab_size}")
    
    # Data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        'data/processed/train_tiny.txt',
        tokenizer,
        batch_size=config.batch_size,
        block_size=config.block_size,
        allowed_base_dirs=('data/',)
    )
    
    val_loader = create_dataloader(
        'data/processed/val_tiny.txt',
        tokenizer,
        batch_size=config.batch_size,
        block_size=config.block_size,
        allowed_base_dirs=('data/',)
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Model
    logger.info("Initializing model...")
    model = AIModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    logger.info("\nStarting training...")
    trainer.train(epochs=args.epochs)
    
    # Save final model
    torch.save(model.state_dict(), 'models/final_model.pt')
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=6)
    parser.add_argument('--n-embd', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-iters', type=int, default=10000)
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.add_argument('--retrain-tokenizer', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    
    args = parser.parse_args()
    main(args)
