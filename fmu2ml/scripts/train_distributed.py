import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmu2ml.config import get_system_config, FNOConfig, HybridFNOConfig, DeepONetConfig, TrainingConfig
from fmu2ml.models import create_model
from fmu2ml.data.processors import create_data_loaders
from fmu2ml.training import PhysicsLoss
from fmu2ml.utils import setup_logger
import warnings
warnings.filterwarnings('ignore')


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DistributedTrainer:
    """Distributed trainer with DDP support optimized for SLURM clusters"""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_logging()
        self.setup_device()
        
        # Load configurations
        self.system_config = get_system_config(args.system)
        self.model_config = self.create_model_config()
        self.train_config = self.create_train_config()
        
        # Create model
        self.model = self.create_model()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        
        # Setup training components
        self.setup_optimizer()
        self.setup_loss()
        self.setup_scheduler()
        self.scaler = GradScaler(enabled=args.mixed_precision)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.global_step = 0
        
        # Setup W&B
        if self.is_main_process and WANDB_AVAILABLE and args.use_wandb:
            self.setup_wandb()
    
    def setup_distributed(self):
        """Setup distributed training with SLURM support"""
        # Check if running under SLURM
        if 'SLURM_PROCID' in os.environ:
            # SLURM environment
            self.rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.node_id = int(os.environ.get('SLURM_NODEID', 0))
            
            # Get master address and port
            hostnames = os.environ.get('SLURM_JOB_NODELIST', 'localhost')
            master_addr = hostnames.split(',')[0] if ',' in hostnames else hostnames
            master_port = os.environ.get('MASTER_PORT', '12355')
            
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            
            self.distributed = True
            
        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Standard DDP environment
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.node_id = 0
            self.distributed = True
            
        else:
            # Single process
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.node_id = 0
            self.distributed = False
        
        self.is_main_process = (self.rank == 0)
        
        # Initialize process group
        if self.distributed:
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            if self.is_main_process:
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"Initialized DDP: world_size={self.world_size}, rank={self.rank}")
                self.logger.info(f"SLURM job: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
                self.logger.info(f"Node: {self.node_id}, Local rank: {self.local_rank}")
    
    def setup_logging(self):
        """Setup logging"""
        log_level = logging.INFO if self.is_main_process else logging.WARNING
        log_file = None
        if self.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
            log_file = f'{self.args.output_dir}/train_rank_{self.rank}.log'
        
        self.logger = setup_logger(
            f'train_rank_{self.rank}',
            log_file=log_file,
            level=log_level
        )
        
        if self.is_main_process:
            self.logger.info("=" * 80)
            self.logger.info("FMU2ML Distributed Training")
            self.logger.info("=" * 80)
            self.logger.info(f"System: {self.args.system}")
            self.logger.info(f"Model: {self.args.model}")
            self.logger.info(f"Data path: {self.args.data_path}")
            self.logger.info(f"Output dir: {self.args.output_dir}")
            self.logger.info(f"Batch size per GPU: {self.args.batch_size}")
            self.logger.info(f"Effective batch size: {self.args.batch_size * self.world_size}")
            self.logger.info(f"GPUs: {self.world_size}")
            self.logger.info(f"CPUs per task: {self.args.cpus_per_task}")
    
    def setup_device(self):
        """Setup CUDA device for this process."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Get number of GPUs available on this node
        num_gpus = torch.cuda.device_count()
        
        # Map local rank to available GPU
        # This handles cases where local_rank >= num_gpus
        device_id = self.local_rank % num_gpus
        
        if self.local_rank >= num_gpus:
            self.logger.warning(
                f"Local rank {self.local_rank} >= num GPUs {num_gpus}. "
                f"Mapping to GPU {device_id}"
            )
        
        torch.cuda.set_device(device_id)
        self.device = torch.device(f"cuda:{device_id}")
        
        if self.rank == 0:
            gpu_name = torch.cuda.get_device_name(device_id)
            self.logger.info(f"Rank {self.rank} using GPU {device_id}: {gpu_name}")
    
    def create_model_config(self):
        """Create model-specific configuration"""
        base_config = {
            'num_cdus': 49,
            'sequence_length': self.args.sequence_length,
        }
        
        if self.args.model == 'fno':
            return FNOConfig(
                **base_config,
                fno_modes=self.args.fno_modes,
                fno_width=self.args.fno_width,
                fno_layers=self.args.fno_layers
            )
        elif self.args.model == 'hybrid_fno':
            return HybridFNOConfig(
                **base_config,
                hidden_dim=self.args.hidden_dim,
                num_gru_layers=self.args.num_gru_layers,
                fno_modes=self.args.fno_modes,
                fno_width=self.args.fno_width,
                fno_layers=self.args.fno_layers
            )
        elif self.args.model == 'deeponet':
            return DeepONetConfig(
                **base_config,
                deeponet_basis_dim=self.args.deeponet_basis_dim
            )
        else:
            raise ValueError(f"Unknown model: {self.args.model}")
    
    def create_train_config(self):
        """Create training configuration"""
        return TrainingConfig(
            batch_size=self.args.batch_size,
            learning_rate=self.args.lr,
            epochs=self.args.epochs,
            weight_decay=self.args.weight_decay,
            data_path=self.args.data_path,
            checkpoint_dir=self.args.output_dir,
            distributed=self.distributed,
            world_size=self.world_size,
            local_rank=self.local_rank,
            val_split=self.args.val_split,
            test_split=self.args.test_split,
            num_workers=self.args.cpus_per_task
        )
    
    def create_model(self):
        """Create and setup model"""
        model = create_model(self.args.model, self.model_config)
        model = model.to(self.device)
        
        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Model: {model.__class__.__name__}")
            self.logger.info(f"Parameters: {n_params:,}")
        
        # Wrap with DDP
        if self.distributed:
            if torch.cuda.is_available():
                # GPU: use device_ids and output_device
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False
                )
            else:
                # CPU: don't use device_ids and output_device
                model = DDP(
                    model,
                    find_unused_parameters=False
                )
        
        return model
    
    def create_dataloaders(self):
        """Create data loaders with chunk specification"""
        # Parse chunk specifications
        train_chunks = self._parse_chunks(self.args.train_chunks)
        val_chunks = self._parse_chunks(self.args.val_chunks)
        test_chunks = self._parse_chunks(self.args.test_chunks)
        
        if self.is_main_process:
            self.logger.info(f"Data chunks:")
            self.logger.info(f"  Train: {train_chunks}")
            self.logger.info(f"  Val: {val_chunks}")
            self.logger.info(f"  Test: {test_chunks}")
        
        # Override train_config with chunk info
        self.train_config.specific_chunks = None  # Don't filter in create_data_loaders
        self.train_config.sample_based_split = self.args.sample_based_split
        
        train_loader, val_loader, test_loader = create_data_loaders(
            self.model_config,
            self.train_config
        )
        
        if self.is_main_process:
            self.logger.info(f"Data loaders created:")
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Val batches: {len(val_loader)}")
            self.logger.info(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _parse_chunks(self, chunk_spec):
        """Parse chunk specification (e.g., '0,1,2' or '0-5')"""
        if chunk_spec is None:
            return None
        
        chunks = []
        for part in chunk_spec.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                chunks.extend(range(start, end + 1))
            else:
                chunks.append(int(part))
        
        return chunks
    
    def setup_optimizer(self):
        """Setup optimizer"""
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
    
    def setup_loss(self):
        """Setup loss functions"""
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Get normalization handler from train_loader
        norm_handler = None
        if hasattr(self.train_loader.dataset, 'norm_handler'):
            norm_handler = self.train_loader.dataset.norm_handler
        
        self.physics_loss = PhysicsLoss(norm_handler=norm_handler)
        self.physics_weight = self.args.physics_weight
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.scheduler_step_size,
                gamma=self.args.scheduler_gamma
            )
        elif self.args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.args.scheduler_gamma,
                patience=self.args.scheduler_patience
            )
        else:
            self.scheduler = None
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        # Check if API key is configured
        if not os.environ.get('WANDB_API_KEY'):
            self.logger.warning(
                "WANDB_API_KEY not found in environment. "
                "Skipping W&B logging. Set it with: export WANDB_API_KEY=your_key"
            )
            return
        
        try:
            # Set offline mode if no network or API key issues
            wandb_mode = os.environ.get('WANDB_MODE', 'online')
            
            wandb.init(
                project=self.args.wandb_project,
                name=f"{self.args.model}_{self.args.system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    **vars(self.args),
                    'world_size': self.world_size,
                    'effective_batch_size': self.args.batch_size * self.world_size
                },
                mode=wandb_mode  # Can be 'online', 'offline', or 'disabled'
            )
            wandb.watch(self.model, log='all', log_freq=100)
            self.logger.info(f"W&B initialized in {wandb_mode} mode")
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
            # Disable W&B for this run
            self.args.use_wandb = False
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_mse = 0
        total_physics = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(self.device.type,enabled=self.args.mixed_precision):
                outputs = self.model(inputs)
                
                # Calculate losses
                mse_loss = self.mse_loss(outputs, targets)
                physics_losses = self.physics_loss(outputs, inputs)
                
                loss = mse_loss + self.physics_weight * physics_losses['total']
            
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
            total_physics += physics_losses['total'].item()
            num_batches += 1
            self.global_step += 1
            
            # Log
            if self.is_main_process and batch_idx % self.args.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f}, MSE: {mse_loss.item():.6f}, "
                    f"Physics: {physics_losses['total'].item():.6f}"
                )
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_physics = total_physics / num_batches
        
        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'physics': avg_physics
        }
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_mse = 0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            
            mse_loss = self.mse_loss(outputs, targets)
            total_loss += mse_loss.item()
            total_mse += mse_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        
        return {
            'loss': avg_loss,
            'mse': avg_mse
        }
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        if not self.is_main_process:
            return
        
        checkpoint_dir = Path(self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model_config,
            'args': self.args
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Starting training")
            self.logger.info("=" * 80)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log
            if self.is_main_process:
                self.logger.info(
                    f"\nEpoch {epoch} Summary:"
                    f"\n  Train Loss: {train_metrics['loss']:.6f}"
                    f"\n  Val Loss: {val_metrics['loss']:.6f}"
                    f"\n  LR: {self.optimizer.param_groups[0]['lr']:.6e}"
                )
                
                if WANDB_AVAILABLE and self.args.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train/loss': train_metrics['loss'],
                        'train/mse': train_metrics['mse'],
                        'train/physics': train_metrics['physics'],
                        'val/loss': val_metrics['loss'],
                        'val/mse': val_metrics['mse'],
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics['loss'], is_best)
        
        if self.is_main_process:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Training complete!")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
            self.logger.info("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed training for datacenter cooling models')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['fno', 'hybrid_fno', 'deeponet'],
                       help='Model architecture')
    parser.add_argument('--system', type=str, default='marconi100',
                       help='System name')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--train_chunks', type=str, default=None,
                       help='Training chunks (e.g., "0,1,2" or "0-5")')
    parser.add_argument('--val_chunks', type=str, default=None,
                       help='Validation chunks')
    parser.add_argument('--test_chunks', type=str, default=None,
                       help='Test chunks')
    parser.add_argument('--sample_based_split', action='store_true',
                       help='Use sample-based split for single chunk')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--sequence_length', type=int, default=12,
                       help='Input sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=30,
                       help='Step size for StepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                       help='Gamma for scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                       help='Patience for ReduceLROnPlateau')
    
    # Model-specific arguments
    parser.add_argument('--fno_modes', type=int, default=16,
                       help='FNO modes')
    parser.add_argument('--fno_width', type=int, default=64,
                       help='FNO width')
    parser.add_argument('--fno_layers', type=int, default=4,
                       help='FNO layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for Hybrid-FNO')
    parser.add_argument('--num_gru_layers', type=int, default=2,
                       help='Number of GRU layers for Hybrid-FNO')
    parser.add_argument('--deeponet_basis_dim', type=int, default=100,
                       help='DeepONet basis dimension')
    
    # Physics loss arguments
    parser.add_argument('--physics_weight', type=float, default=0.3,
                       help='Physics loss weight')
    
    # Hardware arguments (SLURM-aware)
    parser.add_argument('--cpus_per_task', type=int, 
                       default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
                       help='CPUs per task (default from SLURM)')
    
    # Training features
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint save frequency')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory')
    
    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='fmu2ml',
                       help='W&B project name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save args
    if int(os.environ.get('RANK', 0)) == 0:
        import json
        with open(f'{args.output_dir}/train_args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Create trainer and train
    trainer = DistributedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()