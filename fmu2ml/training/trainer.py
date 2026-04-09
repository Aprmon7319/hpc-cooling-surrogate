import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.amp as amp
from typing import Optional
import logging

from ..config import ModelConfig, TrainingConfig
from .loss_functions import PhysicsLoss, CombinedLoss

import torch.serialization

# Add ModelConfig to safe globals
torch.serialization.add_safe_globals([ModelConfig])

# Set up logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, args):
        self.model = model
        self.config = config
        self.args = args

        # Get normalization handler from train_loader's dataset
        self.norm_handler = None
        if hasattr(args, 'train_loader') and hasattr(args.train_loader.dataset, 'norm_handler'):
            self.norm_handler = args.train_loader.dataset.norm_handler
        
        # Setup device and DDP if needed
        self.device = torch.device(f'cuda:{args.local_rank}' if args.distributed 
                                 else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        
        if args.distributed:
            self.model = DDP(self.model, device_ids=[args.local_rank])
        
        # Loss functions - Use the refactored PhysicsLoss
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.physics_loss = PhysicsLoss(norm_handler=self.norm_handler)
        
        # Alternative: Use combined loss
        # self.combined_loss = CombinedLoss(norm_handler=self.norm_handler, physics_weight=0.3)
        
        # Get physics weights from PhysicsLoss module
        self.physics_weights = self.physics_loss.physics_weights
        
        # Add AMP scaler
        self.scaler = amp.GradScaler(enabled=True)
        self.use_amp = True

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                         lr=config.learning_rate,
                                         weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs)
        
        # Set up checkpoint directory
        if args.local_rank == 0:
            self.checkpoint_dir = Path(args.checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
        # Training state tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Try to load checkpoint if exists
        if args.resume:
            self.load_checkpoint()
            
        # Logging
        if args.local_rank == 0:
            os.environ['WANDB_API_KEY'] = config.wandb_api_key
            wandb.init(project="datacenter-cooling-neural-operators",
                      config=vars(config), name=f"{args.model}_run" +
                      f"_{np.random.randint(1000)}" )
            wandb.config.update(vars(args))
            
            logger.info(f"Trainer initialized with device: {self.device}")
            logger.info(f"Using physics loss with weights: {self.physics_weights}")
    
    def set_norm_handler(self, train_loader):
        """Set normalization handler from train_loader"""
        if hasattr(train_loader, 'dataset'):
            if hasattr(train_loader.dataset, 'norm_handler'):
                self.norm_handler = train_loader.dataset.norm_handler
                self.physics_loss.norm_handler = self.norm_handler  # Update physics loss handler
                logger.info("Normalization handler set successfully")
            elif hasattr(train_loader.dataset, 'dataset'):  # For Subset datasets
                if hasattr(train_loader.dataset.dataset, 'norm_handler'):
                    self.norm_handler = train_loader.dataset.dataset.norm_handler
                    self.physics_loss.norm_handler = self.norm_handler  # Update physics loss handler
                    logger.info("Normalization handler set successfully (from subset)")
        
        if self.norm_handler is None:
            logger.warning("Could not find normalization handler in dataset")
    
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_physics_loss = 0
        num_batches = 0
        
        physics_loss_components = {k: 0.0 for k in self.physics_weights.keys()}
        
        progress_bar = tqdm(train_loader, desc="Training", 
                        disable=self.args.local_rank != 0)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                
                # MSE loss
                mse_loss = self.mse_loss(outputs, targets)
                
                # Physics losses (denormalize first)
                if self.norm_handler is not None:
                    outputs_denorm = self.norm_handler.denormalize_output(outputs)
                    inputs_denorm = self.norm_handler.denormalize_input(inputs)
                else:
                    outputs_denorm = outputs
                    inputs_denorm = inputs
                
                physics_losses = self.physics_loss(outputs_denorm, inputs_denorm)
                
                # Combined loss
                total_physics = physics_losses['total']
                loss = mse_loss + 0.1 * total_physics  # Physics weight
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_physics_loss += total_physics.item()
            num_batches += 1
            
            # Track individual physics components
            for k in physics_loss_components:
                if k in physics_losses and k != 'total':
                    physics_loss_components[k] += physics_losses[k].item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}',
                    'physics': f'{total_physics.item():.4f}'
                })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_physics = total_physics_loss / num_batches
        
        for k in physics_loss_components:
            physics_loss_components[k] /= num_batches
        
        # Log to wandb
        if self.args.local_rank == 0 and hasattr(self, 'wandb_run'):
            self.wandb_run.log({
                'train/loss': avg_loss,
                'train/mse': avg_mse,
                'train/physics_loss': avg_physics,
                **{f'train/physics_{k}': v for k, v in physics_loss_components.items()}
            })
        
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        # Track physics violations during validation
        physics_violations = {k: 0.0 for k in self.physics_weights.keys()}
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation",
                                      disable=self.args.local_rank != 0):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                loss = self.mse_loss(outputs, targets)
                mae = self.mae_loss(outputs, targets)
                
                # Compute physics violations using PhysicsLoss module
                physics_losses_dict = self.physics_loss(outputs, inputs)
                
                for k in physics_violations:
                    if k in physics_losses_dict:
                        value = physics_losses_dict[k]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        physics_violations[k] += value
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        # Average physics violations
        for k in physics_violations:
            physics_violations[k] /= num_batches
        
        # Log validation physics violations
        if self.args.local_rank == 0:
            wandb.log({
                f"val/physics/{k}": v for k, v in physics_violations.items()
            })
            
            # Log validation metrics
            wandb.log({
                'val/loss': total_loss / num_batches,
                'val/mae': total_mae / num_batches
            })
        
        return total_loss / num_batches, total_mae / num_batches
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint"""
        if self.args.local_rank != 0:
            return
                
        checkpoint_path = self.checkpoint_dir / filename
        
        # If this is a DDP model, save the internal module
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        
        # Save normalization stats if available
        stats_path = None
        if self.norm_handler is not None:
            stats_path = self.checkpoint_dir / f'norm_stats_{self.args.model}.npz'
            self.norm_handler.save_stats(str(stats_path))
            logger.info(f"Normalization stats saved to {stats_path}")
        else:
            logger.warning("No normalization handler available to save!")
            
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'args': self.args,
            'norm_stats_path': str(stats_path) if stats_path is not None else None,
            'physics_weights': self.physics_weights
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model, save a separate copy
        if is_best:
            best_path = self.checkpoint_dir / f'best_model_{self.args.model}.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self):
        """Load the latest checkpoint if it exists"""
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = Path(self.args.checkpoint_dir)
                
        # Find the latest checkpoint
        checkpoints = list(self.checkpoint_dir.glob(f'checkpoint_{self.args.model}_epoch_*.pth'))
        if not checkpoints:
            if self.args.local_rank == 0:
                logger.info("No checkpoint found, starting from scratch")
            return False
                
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(str(x).split('epoch_')[1].split('.')[0]))
        
        try:
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            # If this is a DDP model, load to the module
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                    
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            
            # Load physics weights if available
            if 'physics_weights' in checkpoint:
                self.physics_weights = checkpoint['physics_weights']
                self.physics_loss.physics_weights = self.physics_weights
            
            if self.args.local_rank == 0:
                logger.info(f"Resuming from checkpoint {latest_checkpoint} at epoch {self.start_epoch}")
            return True
                
        except Exception as e:
            if self.args.local_rank == 0:
                logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def train(self, train_loader, val_loader):
        """Train model with checkpoint support"""
        # Set normalization handler from train_loader
        self.set_norm_handler(train_loader)

        # Initialize current epoch
        self.current_epoch = self.start_epoch
        
        # Loop from start_epoch to account for resumed training
        for epoch in range(self.start_epoch, self.config.epochs):
            # Store current epoch
            self.current_epoch = epoch
            
            # Training
            if self.args.distributed:
                if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
            
            # Training epoch with timing
            epoch_start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - epoch_start_time
            
            # Validation
            val_loss, val_mae = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            checkpoint_frequency = getattr(self.args, 'checkpoint_freq', 5)
            if (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == self.config.epochs:
                self.save_checkpoint(f'checkpoint_{self.args.model}_epoch_{epoch+1}.pth')
            
            # Logging
            if self.args.local_rank == 0:
                # Main metrics
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/mae': val_mae,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch_time': epoch_time
                })
                
                # Log improvement metrics
                if hasattr(self, 'last_val_loss'):
                    improvement = (self.last_val_loss - val_loss) / self.last_val_loss * 100
                    wandb.log({'val/improvement': improvement})
                self.last_val_loss = val_loss
                
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, Val MAE = {val_mae:.4f}, "
                          f"LR = {self.scheduler.get_last_lr()[0]:.6f}, "
                          f"Time = {epoch_time:.1f}s")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f'checkpoint_{self.args.model}_epoch_{epoch+1}.pth', is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
        
        return self.best_val_loss
    
    def test(self, test_loader):
        """Evaluate model on test set with detailed physics analysis"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        # Track detailed physics performance
        physics_performance = {k: {'violations': 0, 'severity': 0} for k in self.physics_weights.keys()}
        
        # Store sample predictions for visualization
        sample_predictions = {
            'inputs': [],
            'outputs': [],
            'targets': []
        }
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing",
                                                             disable=self.args.local_rank != 0)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                loss = self.mse_loss(outputs, targets)
                mae = self.mae_loss(outputs, targets)
                
                # Compute physics violations using PhysicsLoss module
                physics_losses_dict = self.physics_loss(outputs, inputs)
                
                # Track physics violations
                for key in physics_performance:
                    if key in physics_losses_dict:
                        value = physics_losses_dict[key]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        
                        if value > 0.01:  # Threshold for considering a violation
                            physics_performance[key]['violations'] += 1
                        physics_performance[key]['severity'] += value
                
                # Store first few batches for visualization
                if batch_idx < 5:
                    sample_predictions['inputs'].append(inputs.cpu().numpy())
                    sample_predictions['outputs'].append(outputs.cpu().numpy())
                    sample_predictions['targets'].append(targets.cpu().numpy())
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        test_loss = total_loss / num_batches
        test_mae = total_mae / num_batches
        
        # Calculate physics violation rates
        for key in physics_performance:
            physics_performance[key]['violation_rate'] = physics_performance[key]['violations'] / num_batches
            physics_performance[key]['avg_severity'] = physics_performance[key]['severity'] / num_batches
        
        # Logging
        if self.args.local_rank == 0:
            # Log test metrics
            wandb.log({
                'test/loss': test_loss,
                'test/mae': test_mae
            })
            
            # Log physics performance
            for key, metrics in physics_performance.items():
                wandb.log({
                    f'test/physics/{key}_violation_rate': metrics['violation_rate'],
                    f'test/physics/{key}_severity': metrics['avg_severity']
                })
            
            # Create physics performance table
            physics_data = []
            for key, metrics in physics_performance.items():
                physics_data.append([
                    key,
                    f"{metrics['violation_rate']:.2%}",
                    f"{metrics['avg_severity']:.4f}"
                ])
            
            physics_table = wandb.Table(
                columns=["Physics Constraint", "Violation Rate", "Avg Severity"],
                data=physics_data
            )
            wandb.log({"test/physics_summary": physics_table})
            
            # Print detailed results
            logger.info("\n" + "="*60)
            logger.info("TEST RESULTS")
            logger.info("="*60)
            logger.info(f"Test Loss (MSE): {test_loss:.4f}")
            logger.info(f"Test MAE: {test_mae:.4f}")
            logger.info("\nPhysics Constraint Performance:")
            logger.info("-"*60)
            logger.info(f"{'Constraint':<30} {'Violation Rate':>15} {'Avg Severity':>15}")
            logger.info("-"*60)
            for key, metrics in physics_performance.items():
                logger.info(f"{key:<30} {metrics['violation_rate']:>14.2%} {metrics['avg_severity']:>15.4f}")
            logger.info("="*60)
            
            # Create visualization plots if sample predictions were collected
            if sample_predictions['inputs']:
                self._create_test_visualizations(sample_predictions)
        
        return test_loss, test_mae
    
    def _create_test_visualizations(self, sample_predictions):
        """Create visualization plots from test predictions"""
        import matplotlib.pyplot as plt
        
        # Concatenate samples
        inputs = np.concatenate(sample_predictions['inputs'], axis=0)
        outputs = np.concatenate(sample_predictions['outputs'], axis=0)
        targets = np.concatenate(sample_predictions['targets'], axis=0)
        
        # Select a few random samples to visualize
        n_samples = min(5, inputs.shape[0])
        indices = np.random.choice(inputs.shape[0], n_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Plot predictions vs targets for first few outputs
            n_outputs_to_plot = min(49, outputs.shape[1])
            
            # Temperature comparison
            ax = axes[i, 0]
            ax.plot(targets[idx, :n_outputs_to_plot], 'b-', label='Target', alpha=0.7)
            ax.plot(outputs[idx, :n_outputs_to_plot], 'r--', label='Prediction', alpha=0.7)
            ax.set_title(f'Sample {idx}: First {n_outputs_to_plot} Outputs')
            ax.set_xlabel('Output Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Error plot
            ax = axes[i, 1]
            errors = np.abs(outputs[idx, :n_outputs_to_plot] - targets[idx, :n_outputs_to_plot])
            ax.bar(range(n_outputs_to_plot), errors, alpha=0.7)
            ax.set_title(f'Sample {idx}: Absolute Errors')
            ax.set_xlabel('Output Index')
            ax.set_ylabel('|Prediction - Target|')
            ax.grid(True, alpha=0.3)
            
            # Scatter plot
            ax = axes[i, 2]
            ax.scatter(targets[idx, :], outputs[idx, :], alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val = min(targets[idx, :].min(), outputs[idx, :].min())
            max_val = max(targets[idx, :].max(), outputs[idx, :].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_title(f'Sample {idx}: Prediction vs Target')
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
            
            # Add R² score
            ss_tot = np.sum((targets[idx, :] - targets[idx, :].mean())**2)
            ss_res = np.sum((targets[idx, :] - outputs[idx, :])**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"test/sample_predictions": wandb.Image(fig)})
        plt.close(fig)
        
        logger.info("Test visualizations created and logged to wandb")