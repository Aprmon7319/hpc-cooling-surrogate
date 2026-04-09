"""
TNO Training Example with Multiple Specialized Operators
========================================================

Train multiple specialized TNO operators for datacenter cooling prediction:
- TNO_forward: Complete forward model (all outputs)
- TNO_thermal: Thermal subsystem (temperatures only)
- TNO_hydraulic: Hydraulic subsystem (flows and pressures)
- TNO_power: Power consumption (pump power)
- TNO_htc: Heat transfer coefficient
- TNO_inverse: Inverse model (estimate inputs from outputs)
"""
import gc
import psutil
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

# Import from tno_architecture
from fmu2ml.data.processors.tno_architecture import (
    TNOConfig,
    TemporalNeuralOperator,
    TNOLoss,
    create_tno_model,
    get_activation,
)

# Import from Dask-based data loader
from fmu2ml.data.processors.tno_data_loader_dask import (
    TNOSequenceConfig,
    LazyDataConfig,
    SplitConfig,
    CDUDataManager,
    CDUDataView,
    create_tno_data_loaders,
    create_data_manager,
    tno_collate_fn,
    get_sample_batch,
    print_batch_shapes,
    inspect_manager,
)


# =============================================================================
# Distributed Training Setup for Frontier
# =============================================================================

def setup_distributed():
    """
    Initialize distributed training for Frontier (AMD MI250X with ROCm).
    Uses RCCL backend (AMD's NCCL equivalent).
    """
    # Slurm/srun sets these environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    # On Frontier with ROCm, 'nccl' maps to RCCL
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only from rank 0 to avoid duplicate output"""
    if is_main_process():
        print(*args, **kwargs)


# =============================================================================
# Output Variable Definitions
# =============================================================================

OUTPUT_VAR_INDICES = {
    'V_flow_prim_GPM': 0, 
    'V_flow_sec_GPM': 1, 
    'W_flow_CDUP_kW': 2,
    'T_prim_s_C': 3, 
    'T_prim_r_C': 4, 
    'T_sec_s_C': 5, 
    'T_sec_r_C': 6,
    'p_prim_s_psig': 7, 
    'p_prim_r_psig': 8, 
    'p_sec_s_psig': 9, 
    'p_sec_r_psig': 10,
    'htc': 11
}


@dataclass
class OperatorConfig:
    """Configuration for a specialized TNO operator"""
    name: str
    description: str
    input_type: str
    output_indices: List[int]
    output_names: List[str]
    lambda_weights: Dict[str, float]
    
    def get_output_dim(self) -> int:
        return len(self.output_indices)


# Physics loss weights exclude cooling tower effectiveness
OPERATOR_CONFIGS = {
    'forward': OperatorConfig(
        name='TNO_forward',
        description='Complete forward model (all outputs)',
        input_type='forward',
        output_indices=list(range(12)),
        output_names=list(OUTPUT_VAR_INDICES.keys()),
        lambda_weights={
            'lambda_data': 1.0, 
            'lambda_physics': 0.1,
            'lambda_temp_order': 0.5, 
            'lambda_positivity': 0.1,
            'lambda_smoothness': 0.05, 
            'lambda_energy': 0.1,
        }
    ),
    'thermal': OperatorConfig(
        name='TNO_thermal',
        description='Thermal subsystem (temperatures)',
        input_type='forward',
        output_indices=[3, 4, 5, 6],
        output_names=['T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C'],
        lambda_weights={
            'lambda_data': 1.0, 
            'lambda_physics': 0.5,
            'lambda_temp_order': 1.0, 
            'lambda_positivity': 0.0,  # Temperatures can be any value
            'lambda_smoothness': 0.1, 
            'lambda_energy': 0.0,  # Need flows for energy balance
        }
    ),
    'hydraulic': OperatorConfig(
        name='TNO_hydraulic',
        description='Hydraulic subsystem (flows and pressures)',
        input_type='forward',
        output_indices=[0, 1, 7, 8, 9, 10],
        output_names=['V_flow_prim_GPM', 'V_flow_sec_GPM', 'p_prim_s_psig', 
                      'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'],
        lambda_weights={
            'lambda_data': 1.0, 
            'lambda_physics': 0.1,
            'lambda_temp_order': 0.0,  # No temperatures
            'lambda_positivity': 0.3,
            'lambda_smoothness': 0.08, 
            'lambda_energy': 0.0,  # Need temperatures for energy
        }
    ),
    'power': OperatorConfig(
        name='TNO_power',
        description='Power consumption (pump power)',
        input_type='forward',
        output_indices=[2],
        output_names=['W_flow_CDUP_kW'],
        lambda_weights={
            'lambda_data': 1.0, 
            'lambda_physics': 0.1,
            'lambda_temp_order': 0.0, 
            'lambda_positivity': 0.5,
            'lambda_smoothness': 0.1, 
            'lambda_energy': 0.0,
        }
    ),
    'htc': OperatorConfig(
        name='TNO_htc',
        description='Heat transfer coefficient',
        input_type='forward',
        output_indices=[11],
        output_names=['htc'],
        lambda_weights={
            'lambda_data': 1.0, 
            'lambda_physics': 0.1,
            'lambda_temp_order': 0.0, 
            'lambda_positivity': 0.3,
            'lambda_smoothness': 0.15, 
            'lambda_energy': 0.0,
        }
    ),
}


# =============================================================================
# Custom TNO Loss without Cooling Tower Effectiveness
# =============================================================================

class TNOPhysicsLoss(TNOLoss):
    """
    Extended TNO Loss that integrates with PhysicsValidator.
    Excludes cooling tower effectiveness loss.
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_temp_order: float = 0.5,
        lambda_positivity: float = 0.1,
        lambda_smoothness: float = 0.05,
        lambda_energy: float = 0.1,
        denormalize_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        output_indices: Optional[List[int]] = None,
        use_physics_validator: bool = False,
        system_name: str = 'summit'
    ):
        """Initialize with physics validator integration."""
        super().__init__(
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
            lambda_temp_order=lambda_temp_order,
            lambda_positivity=lambda_positivity,
            lambda_smoothness=lambda_smoothness,
            lambda_energy=lambda_energy,
            denormalize_fn=denormalize_fn,
            use_physics_validator=False,  # We handle physics separately
            physics_validator_config=None,
            device=device,
            output_indices=output_indices
        )
        
        self.use_external_physics = use_physics_validator
        self.physics_validator = None
        
        if use_physics_validator:
            try:
                from fmu2ml.data.processors.physics_loss import PhysicsValidator
                self.physics_validator = PhysicsValidator(
                    device=self.device,
                    system_name=system_name
                )
                # Disable cooling tower effectiveness in physics weights
                self.physics_validator.physics_weights['cooling_tower_effectiveness'] = 0.0
            except ImportError:
                print("Warning: PhysicsValidator not available")
                self.use_external_physics = False
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        Q_flow_raw: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with physics constraints.
        Excludes cooling tower effectiveness.
        """
        # Call parent forward
        total_loss, losses = super().forward(
            y_pred, y_true, Q_flow_raw, return_components=True
        )
        
        if return_components:
            return total_loss, losses
        return total_loss


# =============================================================================
# Denormalization Helper
# =============================================================================

def create_denormalize_fn_for_operator(
    data_manager: CDUDataManager,
    output_indices: List[int]
) -> Callable:
    """
    Create a denormalization function for a specific operator's outputs.
    
    Args:
        data_manager: The CDU data manager with normalization stats
        output_indices: Indices of outputs this operator predicts
        
    Returns:
        Function to denormalize predictions
    """
    norm_handler = data_manager.norm_handler
    
    if norm_handler is None or norm_handler.mean_out is None:
        # No normalization applied
        def identity_denorm(y: torch.Tensor) -> torch.Tensor:
            return y
        return identity_denorm
    
    # Get mean/std for the specific output indices
    # Note: Need to map from operator indices to full output indices
    n_outputs = len(output_indices)
    
    def denormalize_fn(y_pred: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions using stored statistics."""
        device = y_pred.device
        dtype = y_pred.dtype
        
        # Get the relevant normalization parameters
        mean_vals = []
        std_vals = []
        
        for idx in output_indices:
            if idx < len(norm_handler.mean_out):
                mean_vals.append(norm_handler.mean_out[idx])
                std_vals.append(norm_handler.std_out[idx])
            else:
                mean_vals.append(0.0)
                std_vals.append(1.0)
        
        mean = torch.tensor(mean_vals, device=device, dtype=dtype)
        std = torch.tensor(std_vals, device=device, dtype=dtype)
        
        # Handle potential zero std
        std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
        
        # Denormalize: y = y_normalized * std + mean
        return y_pred * std + mean
    
    return denormalize_fn


# =============================================================================
# Distributed Multi-Operator Trainer
# =============================================================================

class DistributedMultiOperatorTrainer:
    """Trainer for multiple specialized TNO operators with DDP support"""
    
    def __init__(
        self,
        data_path: str,
        checkpoint_base_dir: str,
        operators_to_train: List[str],
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        history_length: int = 30,
        prediction_horizon: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        patience: int = 20,
        use_wandb: bool = True,
        train_chunks: Optional[List[int]] = None,
        val_chunks: Optional[List[int]] = None,
        test_chunks: Optional[List[int]] = None,
        cdu_lim: Optional[int] = 50,
        num_workers: int = 4,
        wandb_project: str = "tno-cooling",
        wandb_entity: Optional[str] = None,
        use_bf16: bool = True,
        system_name: str = 'summit',
        cache_dir: Optional[str] = None,
        use_physics_validator: bool = True,
    ):
        self.data_path = data_path
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.operators_to_train = operators_to_train
        
        # Distributed settings
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Training settings
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.effective_batch_size = batch_size * world_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.use_wandb = use_wandb and WANDB_AVAILABLE and is_main_process()
        self.train_chunks = train_chunks
        self.val_chunks = val_chunks
        self.test_chunks = test_chunks
        self.cdu_lim = cdu_lim
        self.num_workers = num_workers
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.use_bf16 = use_bf16
        self.system_name = system_name
        self.cache_dir = cache_dir
        self.use_physics_validator = use_physics_validator
        
        # BF16 Automatic Mixed Precision
        if self.use_bf16 and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
            print_rank0("Using BFloat16 mixed precision")
        else:
            self.amp_dtype = torch.float32
        
        if is_main_process():
            self.checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronize before continuing
        if dist.is_initialized():
            dist.barrier()
        
        self.results = {}
        self.data_manager = None
        
        print_rank0(f"Distributed Trainer initialized:")
        print_rank0(f"  World size: {world_size}")
        print_rank0(f"  Per-GPU batch size: {batch_size}")
        print_rank0(f"  Effective batch size: {self.effective_batch_size}")
        print_rank0(f"  Operators to train: {', '.join(operators_to_train)}")
        print_rank0(f"  Device: {self.device}")
    
    def _create_data_loaders(self):
        """Create data loaders using Dask-based loader"""
        print_rank0("\n" + "="*80)
        print_rank0("Creating Data Loaders (Dask Backend)")
        print_rank0("="*80)
        
        tno_seq_config = TNOSequenceConfig(
            history_length=self.history_length,
            prediction_horizon=self.prediction_horizon,
            stride=1,
            pool_cdus=True
        )
        
        # Use the Dask-based data loader
        train_loader, val_loader, test_loader, data_manager = create_tno_data_loaders(
            data_path=self.data_path,
            tno_config=tno_seq_config,
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            num_workers=self.num_workers,
            pin_memory=True,
            distributed=(self.world_size > 1),
            system_name=self.system_name,
            cdu_lim=self.cdu_lim,
            chunk_indices=self.train_chunks,
            cache_dir=self.cache_dir,
            use_mmap=True,
            verbose=is_main_process()
        )
        
        self.data_manager = data_manager
        
        if is_main_process():
            print("\nData Manager Info:")
            info = inspect_manager(data_manager)
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            print("\nSample batch shapes:")
            sample_batch = get_sample_batch(train_loader)
            print_batch_shapes(sample_batch)
        
        return train_loader, val_loader, test_loader
    
    def _create_output_selector(self, output_indices: List[int]):
        """Create function to select specific outputs"""
        indices = torch.tensor(output_indices)
        
        def select_outputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            selected_batch = {}
            for key, value in batch.items():
                if key in ['y_hist', 'y_future']:
                    selected_batch[key] = value[..., output_indices]
                else:
                    selected_batch[key] = value
            return selected_batch
        return select_outputs
    
    def _train_operator(
        self,
        operator_name: str,
        train_loader,
        val_loader,
        test_loader,
    ) -> Dict:
        """Train a single specialized operator with DDP"""
        config = OPERATOR_CONFIGS[operator_name]
        
        print_rank0("\n" + "="*80)
        print_rank0(f"Training {config.name}: {config.description}")
        print_rank0("="*80)
        print_rank0(f"Output variables: {', '.join(config.output_names)}")
        print_rank0(f"Output dimension: {config.get_output_dim()}")
        
        # Create model using tno_architecture
        tno_config = TNOConfig(
            input_dim=3,  # [Q_flow, T_Air, T_ext]
            output_dim=config.get_output_dim(),
            history_length=self.history_length,
            prediction_horizon=self.prediction_horizon,
            latent_dim=64,
            d_model=32,
            branch_type='conv1d',
            tbranch_type='conv1d',
            dropout=0.3,
            activation='gelu',
            use_layer_norm=True,
            use_batch_norm=True
        )
        
        model = TemporalNeuralOperator(tno_config)
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            model = DDP(
                model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            base_model = model.module
        else:
            base_model = model
        
        if is_main_process():
            param_counts = base_model.get_num_parameters()
            print(f"\nModel Parameters:")
            for name, count in param_counts.items():
                print(f"  {name}: {count:,}")
        
        # Create denormalization function for this operator
        denormalize_fn = create_denormalize_fn_for_operator(
            self.data_manager, 
            config.output_indices
        )
        
        # Create loss function with physics constraints (excluding cooling tower)
        loss_fn = TNOPhysicsLoss(
            lambda_data=config.lambda_weights.get('lambda_data', 1.0),
            lambda_physics=config.lambda_weights.get('lambda_physics', 0.1),
            lambda_temp_order=config.lambda_weights.get('lambda_temp_order', 0.5),
            lambda_positivity=config.lambda_weights.get('lambda_positivity', 0.1),
            lambda_smoothness=config.lambda_weights.get('lambda_smoothness', 0.05),
            lambda_energy=config.lambda_weights.get('lambda_energy', 0.1),
            denormalize_fn=denormalize_fn,
            device=self.device,
            output_indices=config.output_indices,
            use_physics_validator=self.use_physics_validator,
            system_name=self.system_name
        )
        
        # Create output selector
        output_selector = self._create_output_selector(config.output_indices)
        
        # Create checkpoint directory
        checkpoint_dir = self.checkpoint_base_dir / config.name
        if is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        
        # Create optimizer with scaled learning rate
        scaled_lr = self.learning_rate * self.world_size if self.world_size > 1 else self.learning_rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs
        )
        
        # Setup wandb
        if self.use_wandb:
            wandb_config = {
                'operator': config.name,
                'description': config.description,
                'output_variables': config.output_names,
                'output_dim': config.get_output_dim(),
                'world_size': self.world_size,
                'effective_batch_size': self.effective_batch_size,
                'scaled_lr': scaled_lr,
                'use_bf16': self.use_bf16,
                'sequence': {
                    'history_length': self.history_length,
                    'prediction_horizon': self.prediction_horizon,
                },
                'training': {
                    'batch_size_per_gpu': self.batch_size,
                    'epochs': self.num_epochs,
                    'base_lr': self.learning_rate,
                },
                'loss_weights': config.lambda_weights,
                'physics_validator': self.use_physics_validator,
            }
            
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=wandb_config,
                name=f"{config.name}_{'ddp_' + str(self.world_size) + 'gpu' if self.world_size > 1 else 'single'}"
            )
        
        # Training loop
        print_rank0(f"\nStarting training for {config.name}...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = defaultdict(list)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            train_losses = defaultdict(float)
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Select outputs for this operator
                batch = output_selector(batch)
                
                # Move to device
                u_hist = batch['u_hist'].to(self.device)
                y_hist = batch['y_hist'].to(self.device)
                y_future = batch['y_future'].to(self.device)
                Q_flow_raw = batch['Q_flow_raw'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_bf16):
                    y_pred = model(u_hist, y_hist)
                    loss, loss_components = loss_fn(
                        y_pred, y_future, Q_flow_raw, return_components=True
                    )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Accumulate losses
                for key, value in loss_components.items():
                    train_losses[key] += value
                num_batches += 1
                
                # Log batch progress
                if batch_idx % 100 == 0 and is_main_process():
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            # Average training losses
            for key in train_losses:
                train_losses[key] /= num_batches
            
            # All-reduce losses across processes
            if dist.is_initialized():
                for key in train_losses:
                    loss_tensor = torch.tensor(train_losses[key], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    train_losses[key] = loss_tensor.item() / self.world_size
            
            # Validation phase
            model.eval()
            val_losses = defaultdict(float)
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = output_selector(batch)
                    
                    u_hist = batch['u_hist'].to(self.device)
                    y_hist = batch['y_hist'].to(self.device)
                    y_future = batch['y_future'].to(self.device)
                    Q_flow_raw = batch['Q_flow_raw'].to(self.device)
                    
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_bf16):
                        y_pred = model(u_hist, y_hist)
                        _, loss_components = loss_fn(
                            y_pred, y_future, Q_flow_raw, return_components=True
                        )
                    
                    for key, value in loss_components.items():
                        val_losses[key] += value
                    num_val_batches += 1
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= max(num_val_batches, 1)
            
            # All-reduce validation losses
            if dist.is_initialized():
                for key in val_losses:
                    loss_tensor = torch.tensor(val_losses[key], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    val_losses[key] = loss_tensor.item() / self.world_size
            
            scheduler.step()
            
            # Record history
            for key, value in train_losses.items():
                history[f'train_{key}'].append(value)
            for key, value in val_losses.items():
                history[f'val_{key}'].append(value)
            
            elapsed = time.time() - start_time
            val_total = val_losses.get('total', 0.0)
            
            # Early stopping and checkpointing
            if is_main_process():
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    patience_counter = 0
                    
                    # Save best model
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_total,
                        'config': tno_config,
                        'operator_config': config,
                    }
                    torch.save(save_dict, checkpoint_dir / 'best_model.pt')
                else:
                    patience_counter += 1
                
                # Logging
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{self.num_epochs} ({elapsed:.1f}s) | "
                      f"LR: {lr:.2e} | Train: {train_losses['total']:.6f} | "
                      f"Val: {val_total:.6f} | Best: {best_val_loss:.6f} | "
                      f"Patience: {patience_counter}/{self.patience}")
                
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/total_loss': train_losses['total'],
                        'train/data_loss': train_losses.get('data', 0),
                        'train/temp_order_loss': train_losses.get('temp_order', 0),
                        'train/positivity_loss': train_losses.get('positivity', 0),
                        'train/smoothness_loss': train_losses.get('smoothness', 0),
                        'train/energy_loss': train_losses.get('energy', 0),
                        'val/total_loss': val_total,
                        'val/data_loss': val_losses.get('data', 0),
                        'training/learning_rate': lr,
                        'training/epoch_time': elapsed,
                        'training/best_val_loss': best_val_loss,
                        'training/patience_counter': patience_counter,
                    }
                    wandb.log(log_dict, step=epoch + 1)
            
            # Broadcast patience counter
            if dist.is_initialized():
                patience_tensor = torch.tensor(patience_counter, device=self.device)
                dist.broadcast(patience_tensor, src=0)
                patience_counter = patience_tensor.item()
            
            # Check early stopping
            if patience_counter >= self.patience:
                print_rank0(f"Early stopping at epoch {epoch+1}")
                break
            
            if dist.is_initialized():
                dist.barrier()
        
        # Load best model for evaluation
        if is_main_process() and (checkpoint_dir / 'best_model.pt').exists():
            checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=self.device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Finish wandb
        if self.use_wandb:
            wandb.finish()
        
        return {
            'config': config,
            'history': dict(history),
            'best_val_loss': best_val_loss,
            'checkpoint_dir': checkpoint_dir,
            'model': base_model
        }
    
    def train_all(self):
        """Train all specified operators"""
        print_rank0("\n" + "="*80)
        print_rank0("DISTRIBUTED MULTI-OPERATOR TNO TRAINING")
        print_rank0("="*80)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()
        
        # Train each operator
        for i, operator_name in enumerate(self.operators_to_train, 1):
            print_rank0(f"\n\n{'='*80}")
            print_rank0(f"OPERATOR {i}/{len(self.operators_to_train)}: {operator_name}")
            print_rank0(f"{'='*80}")
            
            result = self._train_operator(
                operator_name,
                train_loader,
                val_loader,
                test_loader,
            )
            
            self.results[operator_name] = result
            
            # Clear memory between operators
            gc.collect()
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
        
        if is_main_process():
            self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"{'Operator':<20} {'Best Val Loss':<15} {'Checkpoint':<50}")
        print("-"*80)
        for name, result in self.results.items():
            print(f"{name:<20} {result['best_val_loss']:<15.6f} {str(result['checkpoint_dir']):<50}")
        print("="*80)


# =============================================================================
# Memory Utilities
# =============================================================================

def check_memory_status(threshold_percent=90, label=""):
    if is_main_process():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = psutil.virtual_memory().percent
        mem_used_gb = mem_info.rss / (1024 ** 3)
        print(f"[{label}] Memory: {mem_used_gb:.2f}GB used | {mem_percent:.1f}% system")


def check_gpu_memory(threshold_percent=90, label=""):
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        allocated = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
        total = torch.cuda.get_device_properties(local_rank).total_memory / (1024 ** 3)
        print(f"[{label}] GPU {local_rank}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved / {total:.2f}GB total")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Check if running distributed
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    if world_size > 1:
        # Initialize distributed training
        rank, world_size, local_rank = setup_distributed()
        print(f"Rank {rank}/{world_size}, Local Rank {local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        print("Running in single-GPU mode")
    
    print_rank0(f"Initialized training: {world_size} process(es)")
    
    # Configuration
    DATA_PATH = "../summit/data"
    CHECKPOINT_BASE_DIR = "checkpoints/multi_operator_tno"
    
    HISTORY_LENGTH = 30
    PREDICTION_HORIZON = 1
    
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 20
    NUM_WORKERS = 4
    CDU_LIM = 257
    TRAIN_CHUNKS = [0, 1, 2, 3]
    SYSTEM_NAME = 'summit'
    
    USE_WANDB = True
    WANDB_PROJECT = "tno-cooling-frontier"
    WANDB_KEY = os.environ.get('WANDB_API_KEY', None)
    
    OPERATORS = ['thermal', 'hydraulic', 'power']
    
    # Initialize wandb (rank 0 only)
    if USE_WANDB and WANDB_AVAILABLE and is_main_process() and WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    
    try:
        check_memory_status(90, "INIT")
        if torch.cuda.is_available():
            check_gpu_memory(90, "INIT")
        
        trainer = DistributedMultiOperatorTrainer(
            data_path=DATA_PATH,
            checkpoint_base_dir=CHECKPOINT_BASE_DIR,
            operators_to_train=OPERATORS,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            history_length=HISTORY_LENGTH,
            prediction_horizon=PREDICTION_HORIZON,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            use_wandb=USE_WANDB,
            train_chunks=TRAIN_CHUNKS,
            val_chunks=None,  # Use temporal split
            test_chunks=None,
            cdu_lim=CDU_LIM,
            num_workers=NUM_WORKERS,
            wandb_project=WANDB_PROJECT,
            use_bf16=True,
            system_name=SYSTEM_NAME,
            use_physics_validator=True,
        )
        
        results = trainer.train_all()
        
        # Cleanup data manager
        if trainer.data_manager is not None:
            trainer.data_manager.cleanup()
        
        return results
        
    except Exception as e:
        print(f"Rank {rank} error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()