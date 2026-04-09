from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    device: str = 'cuda'
    
    # Optimizer settings
    weight_decay: float = 1e-4
    
    # Data splits
    val_split: float = 0.15
    test_split: float = 0.15

    # Data path
    data_path: str = 'data/'

    # ADD: CPU workers for data loading
    num_workers: int = 8
    
    # Checkpoint settings
    checkpoint_dir: str = 'checkpoints'
    checkpoint_freq: int = 5
    save_best_only: bool = True
    
    # Logging
    wandb_api_key: str = "200b677eeeb86f1b039af5886bc26db7e59d9be2"
    log_interval: int = 10
    
    # Physics loss weights
    physics_weights: Dict[str, float] = field(default_factory=lambda: {
        'temp_ordering': 2.0,
        'approach_temp': 1.5,
        'pue_physics': 1.0,
        'mass_conservation': 1.5,
        'monotonicity': 0.5,
        'thermodynamic_cop': 1.0,
        'cooling_tower_effectiveness': 1.0
    })
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    
    def __post_init__(self):
        """Validate and setup derived parameters."""
        
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert 0 < self.test_split < 1, "test_split must be between 0 and 1"
        assert self.val_split + self.test_split < 1, "val_split + test_split must be < 1"