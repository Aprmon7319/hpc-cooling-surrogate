from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Base model configuration."""
    num_cdus: int = 49
    input_dim: int = 99  # 2 variables per CDU + 1 global variable
    output_dim_per_cdu: int = 11
    sequence_length: int = 12
    
    def __post_init__(self):
        """Validate and auto-calculate dimensions."""
        # Auto-calculate input_dim if not explicitly set
        if not hasattr(self, '_input_dim_set'):
            self.input_dim = self.num_cdus * 2 + 1
        
        assert self.num_cdus > 0, "num_cdus must be positive"
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.output_dim_per_cdu > 0, "output_dim_per_cdu must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"

@dataclass
class FNOConfig(ModelConfig):
    """FNO model configuration."""
    num_cdus: int = 49
    fno_modes: int = 16
    fno_width: int = 64
    fno_layers: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        assert self.fno_modes > 0, "fno_modes must be positive"
        assert self.fno_width > 0, "fno_width must be positive"
        assert self.fno_layers > 0, "fno_layers must be positive"


@dataclass
class HybridFNOConfig(ModelConfig):
    """Hybrid FNO model configuration."""
    num_cdus: int = 49
    hidden_dim: int = 128
    num_gru_layers: int = 2
    fno_modes: int = 3
    fno_width: int = 32
    fno_layers: int = 2
    
    def __post_init__(self):
        super().__post_init__()
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_gru_layers > 0, "num_gru_layers must be positive"


@dataclass
class DeepONetConfig(ModelConfig):
    """DeepONet model configuration."""
    num_cdus: int = 49
    deeponet_branch_layers: Optional[List[int]] = None
    deeponet_trunk_layers: Optional[List[int]] = None
    deeponet_basis_dim: int = 100
    
    def __post_init__(self):
        super().__post_init__()
        if self.deeponet_branch_layers is None:
            self.deeponet_branch_layers = [128, 256, 512, 256]
        if self.deeponet_trunk_layers is None:
            self.deeponet_trunk_layers = [128, 256, 512, 256]
        assert self.deeponet_basis_dim > 0, "deeponet_basis_dim must be positive"