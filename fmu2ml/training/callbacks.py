import torch
from pathlib import Path


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    """Save model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True, 
                 monitor: str = 'val_loss', mode: str = 'min', verbose: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model, epoch: int, metrics: dict):
        """Save checkpoint if conditions are met"""
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            return
        
        is_best = False
        if self.mode == 'min':
            if metric_value < self.best_value:
                self.best_value = metric_value
                is_best = True
        else:
            if metric_value > self.best_value:
                self.best_value = metric_value
                is_best = True
        
        if self.save_best_only and not is_best:
            return
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        
        if self.verbose:
            print(f'Saved checkpoint to {checkpoint_path}')
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            if self.verbose:
                print(f'Saved best model to {best_path}')

