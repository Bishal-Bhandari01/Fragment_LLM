import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, train_dataloader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr = config.learning_rate,
            weight_decay = config.weight_decay
        )

        # cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config.max_iters
        )

        # fradiwnt scaler for mixed precision  # TODO: fix typo
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x,y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # forward with mixed precision
            with torch.cuda.amp.autocast():
                logits, loss = self.model(x, y)
            
            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)

            # update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            # wandb logging if enabled
            if wandb.run is not None:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for x, y in tqdm(self.val_loader, desc="Validation"):
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        if wandb.run is not None:
            wandb.log({'val_loss': avg_loss})
        
        return avg_loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            # train one epoch
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")

            # validate
            val_loss = self.validate()
            print(f"Val loss: {val_loss:.4f}")

            # update lr
            self.scheduler.step()

            # checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")