"""
Temporal Neural Operator (TNO) for CDU Cooling Model
=====================================================

Implementation of TNO architecture for learning temporal dynamics
of datacenter cooling systems. Designed for use with CDUPooledSequenceDataset.

Architecture:
- Branch Network: Encodes input function history u(t-L:t)
- Temporal Branch (T-Branch): Encodes state history y(t-L:t)
- Trunk Network: Encodes temporal position (optional for bundling)
- Decoder: Projects fused representation to K future timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union
from collections import defaultdict
from pathlib import Path
import time
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .normalization import NormalizationHandler

# Import Dask data loader components
from .tno_data_loader_dask import (
    TNOSequenceConfig,
    LazyDataConfig,
    SplitConfig,
    CDUDataManager,
    CDUDataView,
    create_data_manager,
    create_tno_data_loaders,
    auto_create_data_loaders,
    tno_collate_fn
)

# Import PhysicsValidator for physics-informed losses
from .physics_loss import PhysicsValidator


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TNOConfig:
    """Configuration for Temporal Neural Operator."""
    # Input/Output dimensions
    input_dim: int = 3      # [Q_flow, T_Air, T_ext]
    output_dim: int = 12    # CDU output variables
    history_length: int = 30
    prediction_horizon: int = 10
    
    # Latent space
    latent_dim: int = 128
    d_model: int = 64
    
    # Branch network
    branch_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    branch_type: str = 'conv1d'  # 'conv1d', 'lstm', 'transformer', 'mlp'
    
    # T-Branch network
    tbranch_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    tbranch_type: str = 'conv1d'
    
    # Trunk network
    use_trunk: bool = False
    
    # Decoder
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    
    # Regularization
    dropout: float = 0.1
    activation: str = 'gelu'
    use_layer_norm: bool = True
    use_batch_norm: bool = True


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU()
    }
    return activations.get(name, nn.GELU())


# =============================================================================
# ENCODER MODULE
# =============================================================================

class LinearEncoder(nn.Module):
    """Linear lift from input dimension to model dimension."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 use_layer_norm: bool = True, activation: str = 'gelu'):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        self.activation = get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(x)))


# =============================================================================
# PROCESSOR MODULES
# =============================================================================

class Conv1DProcessor(nn.Module):
    """1D Causal Convolution processor for temporal features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 kernel_size: int = 3, dropout: float = 0.1, 
                 activation: str = 'gelu', use_batch_norm: bool = True):
        super().__init__()
        
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            padding = kernel_size - 1  # Causal padding
            layers.append(nn.Conv1d(dims[i], dims[i+1], kernel_size, padding=padding))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.ModuleList(layers)
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim) -> Conv1d expects (batch, dim, seq_len)
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
                # Remove future padding for causal conv
                if self.kernel_size > 1:
                    x = x[:, :, :-(self.kernel_size - 1)]
            else:
                x = layer(x)
        
        return x.transpose(1, 2)


class LSTMProcessor(nn.Module):
    """LSTM processor for temporal features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(lstm_out_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.output_proj(output)


class TransformerProcessor(nn.Module):
    """Transformer encoder processor for temporal features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 max_seq_len: int = 100):
        super().__init__()
        
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, input_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        x = self.transformer(x, mask=mask)
        return self.output_proj(x)


class MLPProcessor(nn.Module):
    """MLP processor (processes each timestep independently or flattens)."""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dims: List[int],
                 output_dim: int, dropout: float = 0.1, activation: str = 'gelu',
                 flatten_temporal: bool = False):
        super().__init__()
        self.flatten_temporal = flatten_temporal
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        if flatten_temporal:
            flat_dim = input_dim * seq_len
            dims = [flat_dim] + list(hidden_dims)
        else:
            dims = [input_dim] + list(hidden_dims)
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        if flatten_temporal:
            self.output_proj = nn.Linear(dims[-1], output_dim * seq_len)
        else:
            self.output_proj = nn.Linear(dims[-1], output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        if self.flatten_temporal:
            x = x.reshape(batch_size, -1)
            x = self.mlp(x)
            x = self.output_proj(x)
            return x.reshape(batch_size, self.seq_len, self.output_dim)
        else:
            x = self.mlp(x)
            return self.output_proj(x)


# =============================================================================
# BRANCH NETWORK
# =============================================================================

class BranchNetwork(nn.Module):
    """Branch network for encoding input function history."""
    
    def __init__(self, config: TNOConfig):
        super().__init__()
        self.config = config
        
        self.encoder = LinearEncoder(
            config.input_dim, config.d_model,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation
        )
        
        processor_out_dim = config.branch_hidden_dims[-1] if config.branch_hidden_dims else config.d_model
        self.processor = self._build_processor(config, config.d_model, processor_out_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(processor_out_dim, config.latent_dim),
            get_activation(config.activation)
        )
    
    def _build_processor(self, config: TNOConfig, input_dim: int, output_dim: int) -> nn.Module:
        if config.branch_type == 'conv1d':
            return Conv1DProcessor(
                input_dim, config.branch_hidden_dims[:-1] if len(config.branch_hidden_dims) > 1 else [], 
                output_dim, dropout=config.dropout, activation=config.activation,
                use_batch_norm=config.use_batch_norm
            )
        elif config.branch_type == 'lstm':
            return LSTMProcessor(
                input_dim, output_dim // 2, output_dim,
                num_layers=2, dropout=config.dropout
            )
        elif config.branch_type == 'transformer':
            return TransformerProcessor(
                input_dim, output_dim * 2, output_dim,
                num_heads=4, num_layers=2, dropout=config.dropout,
                max_seq_len=config.history_length + 10
            )
        elif config.branch_type == 'mlp':
            return MLPProcessor(
                input_dim, config.history_length, config.branch_hidden_dims,
                output_dim, dropout=config.dropout, activation=config.activation
            )
        else:
            raise ValueError(f"Unknown branch type: {config.branch_type}")
    
    def forward(self, u_hist: torch.Tensor) -> torch.Tensor:
        x = self.encoder(u_hist)
        x = self.processor(x)
        x = x[:, -1, :]  # Take last timestep
        return self.output_proj(x)


# =============================================================================
# TEMPORAL BRANCH NETWORK
# =============================================================================

class TemporalBranchNetwork(nn.Module):
    """T-Branch network for encoding state history."""
    
    def __init__(self, config: TNOConfig):
        super().__init__()
        self.config = config
        
        self.encoder = LinearEncoder(
            config.output_dim, config.d_model,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation
        )
        
        processor_out_dim = config.tbranch_hidden_dims[-1] if config.tbranch_hidden_dims else config.d_model
        self.processor = self._build_processor(config, config.d_model, processor_out_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(processor_out_dim, config.latent_dim),
            get_activation(config.activation)
        )
    
    def _build_processor(self, config: TNOConfig, input_dim: int, output_dim: int) -> nn.Module:
        if config.tbranch_type == 'conv1d':
            return Conv1DProcessor(
                input_dim, config.tbranch_hidden_dims[:-1] if len(config.tbranch_hidden_dims) > 1 else [],
                output_dim, dropout=config.dropout, activation=config.activation,
                use_batch_norm=config.use_batch_norm
            )
        elif config.tbranch_type == 'lstm':
            return LSTMProcessor(
                input_dim, output_dim // 2, output_dim,
                num_layers=2, dropout=config.dropout
            )
        elif config.tbranch_type == 'transformer':
            return TransformerProcessor(
                input_dim, output_dim * 2, output_dim,
                num_heads=4, num_layers=2, dropout=config.dropout,
                max_seq_len=config.history_length + 10
            )
        elif config.tbranch_type == 'mlp':
            return MLPProcessor(
                input_dim, config.history_length, config.tbranch_hidden_dims,
                output_dim, dropout=config.dropout, activation=config.activation
            )
        else:
            raise ValueError(f"Unknown tbranch type: {config.tbranch_type}")
    
    def forward(self, y_hist: torch.Tensor) -> torch.Tensor:
        x = self.encoder(y_hist)
        x = self.processor(x)
        x = x[:, -1, :]
        return self.output_proj(x)


# =============================================================================
# DECODER NETWORK
# =============================================================================

class DecoderNetwork(nn.Module):
    """Decoder to project fused latent representation to output space."""
    
    def __init__(self, config: TNOConfig):
        super().__init__()
        self.config = config
        
        dims = [config.latent_dim] + list(config.decoder_hidden_dims)
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        final_dim = dims[-1]
        self.output_proj = nn.Linear(final_dim, config.prediction_horizon * config.output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mlp(z)
        x = self.output_proj(x)
        return x.view(-1, self.config.prediction_horizon, self.config.output_dim)


# =============================================================================
# MAIN TNO MODEL
# =============================================================================

class TemporalNeuralOperator(nn.Module):
    """
    Temporal Neural Operator (TNO) for CDU Cooling Model.
    
    Architecture:
        u_hist → BranchNetwork → b
        y_hist → TemporalBranchNetwork → tb
        Fusion: z = b ⊙ tb (Hadamard product)
        z → DecoderNetwork → ŷ_fut
    """
    
    def __init__(self, config: TNOConfig):
        super().__init__()
        self.config = config
        
        self.branch = BranchNetwork(config)
        self.tbranch = TemporalBranchNetwork(config)
        self.decoder = DecoderNetwork(config)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, u_hist: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        b = self.branch(u_hist)
        tb = self.tbranch(y_hist)
        z = b * tb  # Hadamard product
        return self.decoder(z)
    
    def predict(self, u_hist: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(u_hist, y_hist)
    
    def get_num_parameters(self) -> Dict[str, int]:
        def count_params(m):
            return sum(p.numel() for p in m.parameters())
        
        return {
            'branch': count_params(self.branch),
            'tbranch': count_params(self.tbranch),
            'decoder': count_params(self.decoder),
            'total': count_params(self)
        }


# =============================================================================
# TNO LOSS FUNCTION (integrated with existing PhysicsValidator)
# =============================================================================

class TNOLoss(nn.Module):
    """
    Combined loss function for TNO training.
    Integrates with PhysicsValidator for physics-informed losses.
    Excludes cooling tower effectiveness loss.
    
    L_total = λ_data × L_data + λ_physics × L_physics
    
    Where L_physics includes:
    - Temperature ordering constraints
    - Approach temperature constraints  
    - Positivity constraints (flows, pressures)
    - Temporal smoothness
    - Energy balance
    - Mass conservation
    - PUE physics
    - Monotonicity
    - Thermodynamic COP
    """
    
    # Output variable indices (consistent with CDUPooledSequenceDataset)
    OUTPUT_INDICES = {
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
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_temp_order: float = 0.5,
        lambda_positivity: float = 0.1,
        lambda_smoothness: float = 0.05,
        lambda_energy: float = 0.1,
        lambda_mass_conservation: float = 0.1,
        lambda_monotonicity: float = 0.05,
        lambda_thermodynamic_cop: float = 0.1,
        denormalize_fn: Optional[Callable] = None,
        use_physics_validator: bool = True,
        physics_validator_config: Optional[Dict] = None,
        system_name: str = 'marconi100',
        device: Optional[torch.device] = None,
        output_indices: Optional[List[int]] = None,
        num_cdus: int = 49
    ):
        """
        Initialize TNO Loss function with PhysicsValidator integration.
        
        Args:
            lambda_data: Weight for data reconstruction loss (MSE)
            lambda_physics: Weight for combined physics loss (from PhysicsValidator)
            lambda_temp_order: Weight for temperature ordering constraint
            lambda_positivity: Weight for positivity constraints
            lambda_smoothness: Weight for temporal smoothness
            lambda_energy: Weight for energy balance constraint
            lambda_mass_conservation: Weight for mass conservation
            lambda_monotonicity: Weight for monotonicity constraint
            lambda_thermodynamic_cop: Weight for COP constraint
            denormalize_fn: Function to denormalize predictions for physics losses
            use_physics_validator: Whether to use full PhysicsValidator
            physics_validator_config: Config dict for PhysicsValidator
            system_name: System name for PhysicsValidator
            device: Device for computations
            output_indices: For specialized operators with reduced output space
            num_cdus: Number of CDUs in the system
        """
        super().__init__()
        
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_temp_order = lambda_temp_order
        self.lambda_positivity = lambda_positivity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_energy = lambda_energy
        self.lambda_mass_conservation = lambda_mass_conservation
        self.lambda_monotonicity = lambda_monotonicity
        self.lambda_thermodynamic_cop = lambda_thermodynamic_cop
        self.denormalize_fn = denormalize_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cdus = num_cdus
        
        # Indices for positive constraints (flows, power, pressures)
        self.positive_indices = [0, 1, 2, 7, 8, 9, 10]
        
        # Physical constants for energy balance
        self.WATER_DENSITY = 997  # kg/m³
        self.WATER_SPECIFIC_HEAT = 4186  # J/(kg·K)
        self.GPM_TO_M3_S = 6.30902e-5
        self.MIN_APPROACH_TEMP = 2.0  # °C
        self.epsilon = 1e-6
        self.MIN_COP = 2.0
        
        # Initialize PhysicsValidator
        self.use_physics_validator = use_physics_validator
        self.physics_validator = None
        
        if use_physics_validator:
            try:
                self.physics_validator = PhysicsValidator(
                    config=physics_validator_config,
                    device=self.device,
                    system_name=system_name
                )
                # Override physics weights to exclude cooling tower effectiveness
                self.physics_validator.physics_weights['cooling_tower_effectiveness'] = 0.0
                logging.info("PhysicsValidator initialized (cooling tower effectiveness disabled)")
            except Exception as e:
                logging.warning(f"PhysicsValidator initialization failed: {e}, using built-in physics losses")
                self.use_physics_validator = False
        
        # Handle specialized operators with reduced output space
        self.output_indices = output_indices or list(range(12))
        self.is_specialized = output_indices is not None and len(output_indices) < 12
        
        # Create mapping from reduced indices to original OUTPUT_INDICES
        self._create_index_mapping()
    
    def _create_index_mapping(self):
        """Create mapping from reduced output space to original indices."""
        # Reverse lookup: original_index -> name
        idx_to_name = {v: k for k, v in self.OUTPUT_INDICES.items()}
        
        # Create mapping: reduced_index -> original_name (if it exists)
        self.reduced_to_name = {}
        for reduced_idx, original_idx in enumerate(self.output_indices):
            if original_idx in idx_to_name:
                self.reduced_to_name[reduced_idx] = idx_to_name[original_idx]
        
        # Track which physics losses are applicable
        self.has_temperatures = any(
            name in self.reduced_to_name.values() 
            for name in ['T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C']
        )
        self.has_flows = any(
            name in self.reduced_to_name.values()
            for name in ['V_flow_prim_GPM', 'V_flow_sec_GPM']
        )
        self.has_pressures = any(
            name in self.reduced_to_name.values()
            for name in ['p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        )
    
    def _get_reduced_index(self, var_name: str) -> Optional[int]:
        """Get reduced index for a variable name, or None if not present."""
        for reduced_idx, name in self.reduced_to_name.items():
            if name == var_name:
                return reduced_idx
        return None
    
    def _prepare_physics_validator_inputs(
        self,
        y_pred: torch.Tensor,
        u_hist: torch.Tensor,
        Q_flow_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tensors in the format expected by PhysicsValidator.
        
        PhysicsValidator expects:
        - outputs: [batch, 590] = 11*49 CDU outputs + 2 datacenter + 49 HTC
        - inputs: [batch, seq_len, 99] = 49*2 (power, temp) + 1 (T_ext)
        
        Args:
            y_pred: (batch, K, output_dim) - Model predictions
            u_hist: (batch, L, 3) - Input history [Q_flow, T_Air, T_ext]
            Q_flow_raw: (batch, L) - Raw Q_flow values
            
        Returns:
            Tuple of (outputs, inputs) formatted for PhysicsValidator
        """
        batch_size = y_pred.shape[0]
        K = y_pred.shape[1]
        
        # Use mean prediction across horizon for physics validation
        y_mean = y_pred.mean(dim=1)  # (batch, output_dim)
        
        # Expand single CDU prediction to all CDUs (simplified assumption)
        # In practice, you might have per-CDU predictions
        num_cdus = self.num_cdus
        
        # Build outputs tensor [batch, 11*num_cdus + 2 + num_cdus]
        # For single CDU model, replicate predictions across CDUs
        outputs = torch.zeros(batch_size, 11 * num_cdus + 2 + num_cdus, device=y_pred.device)
        
        # Fill CDU outputs (11 vars per CDU)
        for cdu_idx in range(num_cdus):
            base_idx = cdu_idx * 11
            # Map from our output format to PhysicsValidator format
            v_flow_prim_idx = self._get_reduced_index('V_flow_prim_GPM')
            v_flow_sec_idx = self._get_reduced_index('V_flow_sec_GPM')
            pump_power_idx = self._get_reduced_index('W_flow_CDUP_kW')
            t_prim_s_idx = self._get_reduced_index('T_prim_s_C')
            t_prim_r_idx = self._get_reduced_index('T_prim_r_C')
            t_sec_s_idx = self._get_reduced_index('T_sec_s_C')
            t_sec_r_idx = self._get_reduced_index('T_sec_r_C')
            p_prim_s_idx = self._get_reduced_index('p_prim_s_psig')
            p_prim_r_idx = self._get_reduced_index('p_prim_r_psig')
            p_sec_s_idx = self._get_reduced_index('p_sec_s_psig')
            p_sec_r_idx = self._get_reduced_index('p_sec_r_psig')
            
            if v_flow_prim_idx is not None:
                outputs[:, base_idx + 0] = y_mean[:, v_flow_prim_idx]
            if v_flow_sec_idx is not None:
                outputs[:, base_idx + 1] = y_mean[:, v_flow_sec_idx]
            if pump_power_idx is not None:
                outputs[:, base_idx + 2] = y_mean[:, pump_power_idx]
            if t_prim_s_idx is not None:
                outputs[:, base_idx + 3] = y_mean[:, t_prim_s_idx]
            if t_prim_r_idx is not None:
                outputs[:, base_idx + 4] = y_mean[:, t_prim_r_idx]
            if t_sec_s_idx is not None:
                outputs[:, base_idx + 5] = y_mean[:, t_sec_s_idx]
            if t_sec_r_idx is not None:
                outputs[:, base_idx + 6] = y_mean[:, t_sec_r_idx]
            if p_prim_s_idx is not None:
                outputs[:, base_idx + 7] = y_mean[:, p_prim_s_idx]
            if p_prim_r_idx is not None:
                outputs[:, base_idx + 8] = y_mean[:, p_prim_r_idx]
            if p_sec_s_idx is not None:
                outputs[:, base_idx + 9] = y_mean[:, p_sec_s_idx]
            if p_sec_r_idx is not None:
                outputs[:, base_idx + 10] = y_mean[:, p_sec_r_idx]
        
        # Datacenter-level outputs
        dc_base = 11 * num_cdus
        if v_flow_prim_idx is not None:
            outputs[:, dc_base] = y_mean[:, v_flow_prim_idx] * num_cdus  # Total flow
        outputs[:, dc_base + 1] = 1.2  # Default PUE estimate
        
        # HTC values
        htc_idx = self._get_reduced_index('htc')
        if htc_idx is not None:
            outputs[:, dc_base + 2:dc_base + 2 + num_cdus] = y_mean[:, htc_idx:htc_idx+1].expand(-1, num_cdus)
        
        # Build inputs tensor [batch, seq_len, 99]
        L = u_hist.shape[1]
        inputs = torch.zeros(batch_size, L, 2 * num_cdus + 1, device=y_pred.device)
        
        # Fill power and temperature for each CDU (using single CDU values)
        for cdu_idx in range(num_cdus):
            # Q_flow (power)
            inputs[:, :, cdu_idx * 2] = Q_flow_raw
            # T_Air (from u_hist[:, :, 1])
            inputs[:, :, cdu_idx * 2 + 1] = u_hist[:, :, 1]
        
        # External temperature (from u_hist[:, :, 2])
        inputs[:, :, -1] = u_hist[:, :, 2]
        
        return outputs, inputs
    
    def _compute_physics_validator_loss(
        self,
        y_pred: torch.Tensor,
        u_hist: torch.Tensor,
        Q_flow_raw: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute physics losses using PhysicsValidator.
        Excludes cooling tower effectiveness.
        
        Returns:
            Dict with physics loss components
        """
        if self.physics_validator is None:
            return {'physics_total': 0.0}
        
        try:
            # Prepare inputs for PhysicsValidator
            outputs, inputs = self._prepare_physics_validator_inputs(y_pred, u_hist, Q_flow_raw)
            
            # Compute physics losses
            physics_losses = self.physics_validator.compute_physics_loss(outputs, inputs)
            
            # Exclude cooling tower effectiveness (already set to 0 weight, but ensure it's not counted)
            physics_losses.pop('cooling_tower_effectiveness', None)
            
            return physics_losses
            
        except Exception as e:
            logging.warning(f"PhysicsValidator loss computation failed: {e}")
            return {'physics_total': 0.0}
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        Q_flow_raw: Optional[torch.Tensor] = None,
        u_hist: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute combined loss.
        
        Args:
            y_pred: (batch, K, output_dim) - Predictions (normalized)
            y_true: (batch, K, output_dim) - Targets (normalized)
            Q_flow_raw: (batch, L) - Raw Q_flow history for physics losses
            u_hist: (batch, L, 3) - Input history for PhysicsValidator
            return_components: Whether to return individual loss components
            
        Returns:
            total_loss or (total_loss, loss_components_dict)
        """
        losses = {}
        
        # 1. Data loss (MSE on normalized predictions)
        losses['data'] = F.mse_loss(y_pred, y_true)
        
        # Denormalize for physics losses if function provided
        if self.denormalize_fn is not None:
            y_pred_denorm = self.denormalize_fn(y_pred)
        else:
            y_pred_denorm = y_pred
        
        # 2. Use PhysicsValidator if available and inputs provided
        if self.use_physics_validator and self.physics_validator is not None and u_hist is not None and Q_flow_raw is not None:
            physics_losses = self._compute_physics_validator_loss(y_pred_denorm, u_hist, Q_flow_raw)
            
            # Extract individual physics components
            losses['temp_ordering'] = physics_losses.get('temp_ordering', 0.0)
            losses['approach_temp'] = physics_losses.get('approach_temp', 0.0)
            losses['pue_physics'] = physics_losses.get('pue_physics', 0.0)
            losses['mass_conservation'] = physics_losses.get('mass_conservation', 0.0)
            losses['energy_balance'] = physics_losses.get('energy_balance', 0.0)
            losses['monotonicity'] = physics_losses.get('monotonicity', 0.0)
            losses['thermodynamic_cop'] = physics_losses.get('thermodynamic_cop', 0.0)
            losses['physics_total'] = physics_losses.get('total', 0.0)
        else:
            # Fallback to built-in physics losses
            # 2a. Temperature ordering loss
            if self.lambda_temp_order > 0:
                losses['temp_ordering'] = self._compute_temp_order_loss(y_pred_denorm)
            else:
                losses['temp_ordering'] = torch.tensor(0.0, device=y_pred.device)
            
            # 2b. Approach temperature loss
            if self.lambda_physics > 0:
                losses['approach_temp'] = self._compute_approach_temp_loss(y_pred_denorm)
            else:
                losses['approach_temp'] = torch.tensor(0.0, device=y_pred.device)
            
            # 2c. Energy balance loss
            if self.lambda_energy > 0 and Q_flow_raw is not None:
                losses['energy_balance'] = self._compute_energy_loss(y_pred_denorm, Q_flow_raw)
            else:
                losses['energy_balance'] = torch.tensor(0.0, device=y_pred.device)
            
            losses['pue_physics'] = torch.tensor(0.0, device=y_pred.device)
            losses['mass_conservation'] = torch.tensor(0.0, device=y_pred.device)
            losses['monotonicity'] = torch.tensor(0.0, device=y_pred.device)
            losses['thermodynamic_cop'] = torch.tensor(0.0, device=y_pred.device)
            losses['physics_total'] = torch.tensor(0.0, device=y_pred.device)
        
        # 3. Positivity loss (always computed locally)
        if self.lambda_positivity > 0:
            losses['positivity'] = self._compute_positivity_loss(y_pred_denorm)
        else:
            losses['positivity'] = torch.tensor(0.0, device=y_pred.device)
        
        # 4. Smoothness loss (on normalized for stability)
        if self.lambda_smoothness > 0:
            losses['smoothness'] = self._compute_smoothness_loss(y_pred)
        else:
            losses['smoothness'] = torch.tensor(0.0, device=y_pred.device)
        
        # Calculate total weighted loss
        def to_tensor(x):
            if isinstance(x, (int, float)):
                return torch.tensor(x, device=y_pred.device)
            return x
        
        total_loss = (
            self.lambda_data * losses['data'] +
            self.lambda_temp_order * to_tensor(losses['temp_ordering']) +
            self.lambda_positivity * to_tensor(losses['positivity']) +
            self.lambda_smoothness * to_tensor(losses['smoothness']) +
            self.lambda_energy * to_tensor(losses['energy_balance']) +
            self.lambda_physics * to_tensor(losses['approach_temp']) +
            self.lambda_mass_conservation * to_tensor(losses['mass_conservation']) +
            self.lambda_monotonicity * to_tensor(losses['monotonicity']) +
            self.lambda_thermodynamic_cop * to_tensor(losses['thermodynamic_cop'])
        )
        
        losses['total'] = total_loss
        
        if return_components:
            return total_loss, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        return total_loss
    
    def _compute_temp_order_loss(self, y: torch.Tensor) -> torch.Tensor:
        """
        Temperature ordering constraints:
        - T_prim_r > T_prim_s (primary return > supply)
        - T_sec_r > T_sec_s (secondary return > supply)
        """
        if not self.has_temperatures:
            return torch.tensor(0.0, device=y.device)
        
        violations = []
        
        # Primary loop
        t_prim_s_idx = self._get_reduced_index('T_prim_s_C')
        t_prim_r_idx = self._get_reduced_index('T_prim_r_C')
        if t_prim_s_idx is not None and t_prim_r_idx is not None:
            prim_violation = F.relu(y[..., t_prim_s_idx] - y[..., t_prim_r_idx])
            violations.append(prim_violation.mean())
        
        # Secondary loop
        t_sec_s_idx = self._get_reduced_index('T_sec_s_C')
        t_sec_r_idx = self._get_reduced_index('T_sec_r_C')
        if t_sec_s_idx is not None and t_sec_r_idx is not None:
            sec_violation = F.relu(y[..., t_sec_s_idx] - y[..., t_sec_r_idx])
            violations.append(sec_violation.mean())
        
        if violations:
            return torch.stack(violations).mean()
        return torch.tensor(0.0, device=y.device)
    
    def _compute_positivity_loss(self, y: torch.Tensor) -> torch.Tensor:
        """
        Positivity constraints for flows, power, and pressures.
        These physical quantities should be non-negative.
        """
        violations = []
        
        # Check each variable in the reduced output space
        for reduced_idx, var_name in self.reduced_to_name.items():
            # Variables that should be positive
            positive_vars = [
                'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
            ]
            if var_name in positive_vars:
                violations.append(F.relu(-y[..., reduced_idx]).mean())
        
        if violations:
            return torch.stack(violations).mean()
        return torch.tensor(0.0, device=y.device)
    
    def _compute_smoothness_loss(self, y: torch.Tensor) -> torch.Tensor:
        """
        Temporal smoothness penalty.
        Penalizes large jumps between consecutive prediction timesteps.
        """
        if y.dim() < 3 or y.size(1) < 2:
            return torch.tensor(0.0, device=y.device)
        
        # First-order difference
        diff = y[:, 1:, :] - y[:, :-1, :]
        return (diff ** 2).mean()
    
    def _compute_approach_temp_loss(self, y: torch.Tensor) -> torch.Tensor:
        """
        Approach temperature constraints for heat exchanger.
        Minimum temperature difference required between loops.
        """
        if not self.has_temperatures:
            return torch.tensor(0.0, device=y.device)
        
        violations = []
        
        # Get indices
        t_sec_r_idx = self._get_reduced_index('T_sec_r_C')
        t_prim_s_idx = self._get_reduced_index('T_prim_s_C')
        t_sec_s_idx = self._get_reduced_index('T_sec_s_C')
        t_prim_r_idx = self._get_reduced_index('T_prim_r_C')
        
        # Approach 1: T_sec_r - T_prim_s should be >= MIN_APPROACH_TEMP
        if t_sec_r_idx is not None and t_prim_s_idx is not None:
            approach1 = y[..., t_sec_r_idx] - y[..., t_prim_s_idx]
            violation1 = F.relu(self.MIN_APPROACH_TEMP - approach1)
            violations.append(violation1.mean())
        
        # Approach 2: T_sec_s - T_prim_r should be >= MIN_APPROACH_TEMP  
        if t_sec_s_idx is not None and t_prim_r_idx is not None:
            approach2 = y[..., t_sec_s_idx] - y[..., t_prim_r_idx]
            violation2 = F.relu(self.MIN_APPROACH_TEMP - approach2)
            violations.append(violation2.mean())
        
        if violations:
            return torch.stack(violations).mean()
        return torch.tensor(0.0, device=y.device)
    
    def _compute_energy_loss(self, y: torch.Tensor, Q_flow_raw: torch.Tensor) -> torch.Tensor:
        """
        Simplified energy balance check.
        Heat removed by cooling should be consistent with heat load.
        """
        # Need flow and temperature data
        v_flow_idx = self._get_reduced_index('V_flow_sec_GPM')
        t_sec_r_idx = self._get_reduced_index('T_sec_r_C')
        t_sec_s_idx = self._get_reduced_index('T_sec_s_C')
        
        if v_flow_idx is None or t_sec_r_idx is None or t_sec_s_idx is None:
            return torch.tensor(0.0, device=y.device)
        
        # Use last Q_flow value from history
        Q_load = Q_flow_raw[:, -1] if Q_flow_raw.dim() > 1 else Q_flow_raw
        
        # Average over prediction horizon
        if y.dim() > 2:
            V_flow_sec = y[..., v_flow_idx].mean(dim=1)  # GPM
            delta_T = (y[..., t_sec_r_idx] - y[..., t_sec_s_idx]).mean(dim=1)
        else:
            V_flow_sec = y[..., v_flow_idx]
            delta_T = y[..., t_sec_r_idx] - y[..., t_sec_s_idx]
        
        # Convert flow to m³/s
        V_flow_m3s = V_flow_sec * self.GPM_TO_M3_S
        
        # Heat removed (W)
        Q_removed = self.WATER_DENSITY * V_flow_m3s * self.WATER_SPECIFIC_HEAT * torch.abs(delta_T)
        
        # Relative error (with epsilon for numerical stability)
        rel_error = torch.abs(Q_load - Q_removed) / (Q_load + self.epsilon)
        
        # Clamp to prevent extreme values
        return torch.clamp(rel_error, 0, 1).mean()


# =============================================================================
# TRAINER (updated for Dask data loader)
# =============================================================================

class TNOTrainer:
    """Trainer class for TNO model with wandb logging and Dask data loader support."""
    
    def __init__(
        self,
        model: TemporalNeuralOperator,
        loss_fn: TNOLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 100,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None,
        project_name: str = "tno-cooling"
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
        
        # Initialize wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        
        if self.use_wandb:
            self._init_wandb(wandb_config, project_name)
    
    def _init_wandb(self, config: Optional[Dict], project_name: str):
        """Initialize Weights & Biases logging."""
        try:
            import os
            
            if config and 'wandb_api_key' in config:
                os.environ['WANDB_API_KEY'] = config['wandb_api_key']
            
            wandb_config = config or {}
            wandb_config['model'] = {
                'input_dim': self.model.config.input_dim,
                'output_dim': self.model.config.output_dim,
                'history_length': self.model.config.history_length,
                'prediction_horizon': self.model.config.prediction_horizon,
                'latent_dim': self.model.config.latent_dim,
                'branch_type': self.model.config.branch_type,
                'tbranch_type': self.model.config.tbranch_type,
                'dropout': self.model.config.dropout,
                'activation': self.model.config.activation,
            }
            
            wandb_config['optimizer'] = {
                'type': self.optimizer.__class__.__name__,
                'lr': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0),
            }
            
            if self.scheduler:
                wandb_config['scheduler'] = {'type': self.scheduler.__class__.__name__}
            
            wandb_config['loss_weights'] = {
                'lambda_data': self.loss_fn.lambda_data,
                'lambda_physics': self.loss_fn.lambda_physics,
                'lambda_temp_order': self.loss_fn.lambda_temp_order,
                'lambda_positivity': self.loss_fn.lambda_positivity,
                'lambda_smoothness': self.loss_fn.lambda_smoothness,
                'lambda_energy': self.loss_fn.lambda_energy,
                'lambda_mass_conservation': self.loss_fn.lambda_mass_conservation,
                'lambda_monotonicity': self.loss_fn.lambda_monotonicity,
                'lambda_thermodynamic_cop': self.loss_fn.lambda_thermodynamic_cop,
            }
            
            run_name = f"tno_run_{np.random.randint(10000)}"
            
            init_kwargs = {
                'project': project_name,
                'config': wandb_config,
                'name': run_name,
                'save_code': True
            }
            
            if config and 'wandb_entity' in config:
                init_kwargs['entity'] = config['wandb_entity']
            
            self.wandb_run = wandb.init(**init_kwargs)
            self.wandb_run.watch(self.model, log_freq=100)
            
            self.logger.info(f"Initialized wandb run: {self.wandb_run.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with Dask data loader support."""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle both dict-style (from tno_data_loader_dask) and tuple-style batches
            if isinstance(batch, dict):
                u_hist = batch['u_hist'].to(self.device)
                y_hist = batch['y_hist'].to(self.device)
                y_future = batch['y_future'].to(self.device)
                Q_flow_raw = batch['Q_flow_raw'].to(self.device)
            else:
                # Fallback for tuple-style batches
                u_hist, y_hist, y_future = batch[:3]
                u_hist = u_hist.to(self.device)
                y_hist = y_hist.to(self.device)
                y_future = y_future.to(self.device)
                Q_flow_raw = batch[3].to(self.device) if len(batch) > 3 else None
            
            self.optimizer.zero_grad()
            y_pred = self.model(u_hist, y_hist)
            
            # Pass u_hist to loss function for PhysicsValidator
            loss, loss_components = self.loss_fn(
                y_pred, y_future, Q_flow_raw, u_hist=u_hist, return_components=True
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key, value in loss_components.items():
                epoch_losses[key] += value
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % self.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/global_step': self.global_step,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    }, step=self.global_step)
                    
                    for key, value in loss_components.items():
                        wandb.log({f'train/batch_{key}': value}, step=self.global_step)
        
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model with Dask data loader support."""
        self.model.eval()
        val_losses = defaultdict(float)
        num_batches = 0
        
        for batch in val_loader:
            if isinstance(batch, dict):
                u_hist = batch['u_hist'].to(self.device)
                y_hist = batch['y_hist'].to(self.device)
                y_future = batch['y_future'].to(self.device)
                Q_flow_raw = batch['Q_flow_raw'].to(self.device)
            else:
                u_hist, y_hist, y_future = batch[:3]
                u_hist = u_hist.to(self.device)
                y_hist = y_hist.to(self.device)
                y_future = y_future.to(self.device)
                Q_flow_raw = batch[3].to(self.device) if len(batch) > 3 else None
            
            y_pred = self.model(u_hist, y_hist)
            _, loss_components = self.loss_fn(
                y_pred, y_future, Q_flow_raw, u_hist=u_hist, return_components=True
            )
            
            for key, value in loss_components.items():
                val_losses[key] += value
            num_batches += 1
        
        return {k: v / num_batches for k, v in val_losses.items()}
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        patience: int = 20,
        min_delta: float = 1e-5
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters()}")
        
        if self.use_wandb:
            wandb.log({
                'model/total_parameters': self.model.get_num_parameters()['total'],
                'model/branch_parameters': self.model.get_num_parameters()['branch'],
                'model/tbranch_parameters': self.model.get_num_parameters()['tbranch'],
                'model/decoder_parameters': self.model.get_num_parameters()['decoder'],
            })
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            train_losses = self.train_epoch(train_loader, epoch)
            val_losses = self.validate(val_loader)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            for key, value in train_losses.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_losses.items():
                self.history[f'val_{key}'].append(value)
            
            val_total = val_losses['total']
            if val_total < self.best_val_loss - min_delta:
                self.best_val_loss = val_total
                patience_counter = 0
                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': train_losses['total'],
                'train/data_loss': train_losses.get('data', 0),
                'train/temp_ordering_loss': train_losses.get('temp_ordering', 0),
                'train/positivity_loss': train_losses.get('positivity', 0),
                'train/smoothness_loss': train_losses.get('smoothness', 0),
                'train/energy_balance_loss': train_losses.get('energy_balance', 0),
                'train/approach_temp_loss': train_losses.get('approach_temp', 0),
                'train/mass_conservation_loss': train_losses.get('mass_conservation', 0),
                'train/monotonicity_loss': train_losses.get('monotonicity', 0),
                'train/thermodynamic_cop_loss': train_losses.get('thermodynamic_cop', 0),
                
                'val/total_loss': val_losses['total'],
                'val/data_loss': val_losses.get('data', 0),
                'val/temp_ordering_loss': val_losses.get('temp_ordering', 0),
                
                'training/learning_rate': lr,
                'training/epoch_time': elapsed,
                'training/best_val_loss': self.best_val_loss,
                'training/patience_counter': patience_counter,
            }
            
            if self.use_wandb:
                wandb.log(log_dict, step=epoch + 1)
            
            print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
                  f"LR: {lr:.2e} | Train: {train_losses['total']:.6f} | "
                  f"Val: {val_total:.6f} | Best: {self.best_val_loss:.6f} | "
                  f"Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                if self.use_wandb:
                    wandb.log({'training/early_stopping_epoch': epoch + 1})
                break
            
            if self.checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt')
        
        if self.checkpoint_dir and (self.checkpoint_dir / 'best_model.pt').exists():
            self.load_checkpoint('best_model.pt')
            print("Loaded best model checkpoint")
        
        return dict(self.history)
    
    def save_checkpoint(self, filename: str):
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.model.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        if self.checkpoint_dir is None:
            return
        
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def finish_wandb(self):
        """Finish wandb logging."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
            self.logger.info("Finished wandb run")


# =============================================================================
# FACTORY FUNCTIONS (updated for Dask data loader)
# =============================================================================

def create_tno_model(
    input_dim: int = 3,
    output_dim: int = 12,
    history_length: int = 30,
    prediction_horizon: int = 10,
    latent_dim: int = 128,
    branch_type: str = 'conv1d',
    tbranch_type: str = 'conv1d',
    **kwargs
) -> TemporalNeuralOperator:
    """Factory function to create TNO model."""
    config = TNOConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        latent_dim=latent_dim,
        branch_type=branch_type,
        tbranch_type=tbranch_type,
        **kwargs
    )
    return TemporalNeuralOperator(config)


def create_tno_trainer(
    model: TemporalNeuralOperator,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_data: float = 1.0,
    lambda_physics: float = 0.1,
    lambda_temp_order: float = 0.5,
    lambda_positivity: float = 0.1,
    lambda_smoothness: float = 0.05,
    lambda_energy: float = 0.1,
    lambda_mass_conservation: float = 0.1,
    lambda_monotonicity: float = 0.05,
    lambda_thermodynamic_cop: float = 0.1,
    num_epochs: int = 100,
    checkpoint_dir: Optional[str] = None,
    denormalize_fn: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    output_indices: Optional[List[int]] = None,
    use_physics_validator: bool = True,
    system_name: str = 'marconi100',
    num_cdus: int = 49,
    use_wandb: bool = False,
    wandb_config: Optional[Dict] = None,
    project_name: str = "tno-cooling"
) -> TNOTrainer:
    """
    Factory function to create TNO trainer with physics-informed loss.
    Integrates with PhysicsValidator (excluding cooling tower effectiveness).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    loss_fn = TNOLoss(
        lambda_data=lambda_data,
        lambda_physics=lambda_physics,
        lambda_temp_order=lambda_temp_order,
        lambda_positivity=lambda_positivity,
        lambda_smoothness=lambda_smoothness,
        lambda_energy=lambda_energy,
        lambda_mass_conservation=lambda_mass_conservation,
        lambda_monotonicity=lambda_monotonicity,
        lambda_thermodynamic_cop=lambda_thermodynamic_cop,
        denormalize_fn=denormalize_fn,
        device=device,
        output_indices=output_indices,
        use_physics_validator=use_physics_validator,
        system_name=system_name,
        num_cdus=num_cdus
    )
    
    return TNOTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        project_name=project_name
    )


def create_denormalize_fn(norm_handler: NormalizationHandler) -> Callable:
    """
    Create a denormalization function from NormalizationHandler.
    
    Args:
        norm_handler: Fitted NormalizationHandler instance
        
    Returns:
        Callable that denormalizes output tensors
    """
    def denormalize_fn(y_normalized: torch.Tensor) -> torch.Tensor:
        return norm_handler.denormalize_output(y_normalized)    
    return denormalize_fn


# =============================================================================
# END-TO-END TRAINING FUNCTION WITH DASK DATA LOADER
# =============================================================================

def train_tno_with_dask_loader(
    data_path: str,
    # Model config
    input_dim: int = 3,
    output_dim: int = 12,
    history_length: int = 30,
    prediction_horizon: int = 10,
    latent_dim: int = 128,
    branch_type: str = 'conv1d',
    tbranch_type: str = 'conv1d',
    # Data config
    batch_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    cdu_lim: Optional[int] = None,
    chunk_indices: Optional[List[int]] = None,
    system_name: str = 'marconi100',
    # Training config
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    num_epochs: int = 100,
    patience: int = 20,
    # Physics loss config
    lambda_data: float = 1.0,
    lambda_physics: float = 0.1,
    lambda_temp_order: float = 0.5,
    lambda_positivity: float = 0.1,
    lambda_smoothness: float = 0.05,
    lambda_energy: float = 0.1,
    lambda_mass_conservation: float = 0.1,
    lambda_monotonicity: float = 0.05,
    lambda_thermodynamic_cop: float = 0.1,
    use_physics_validator: bool = True,
    # Other config
    checkpoint_dir: Optional[str] = None,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
    wandb_config: Optional[Dict] = None,
    project_name: str = "tno-cooling",
    verbose: bool = True
) -> Tuple[TemporalNeuralOperator, TNOTrainer, Dict[str, List[float]]]:
    """
    End-to-end training function using Dask data loader and physics-informed loss.
    
    Args:
        data_path: Path to data directory
        input_dim: Input dimension per CDU
        output_dim: Output dimension per CDU
        history_length: Length of history window (L)
        prediction_horizon: Prediction horizon (K)
        latent_dim: Latent dimension for TNO
        branch_type: Type of branch network
        tbranch_type: Type of temporal branch network
        batch_size: Training batch size
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        cdu_lim: Maximum CDUs to load
        chunk_indices: Specific chunks to load
        system_name: System name for config
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        lambda_*: Weights for various loss components
        use_physics_validator: Whether to use PhysicsValidator
        checkpoint_dir: Directory for checkpoints
        num_workers: Data loading workers
        device: Training device
        use_wandb: Whether to use wandb logging
        wandb_config: Wandb configuration
        project_name: Wandb project name
        verbose: Print progress
        
    Returns:
        Tuple of (trained_model, trainer, training_history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print("=" * 60)
        print("TNO Training with Dask Data Loader")
        print("=" * 60)
    
    # Create TNO sequence config
    tno_seq_config = TNOSequenceConfig(
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        stride=1,
        pool_cdus=True,
        include_cdu_id=True
    )
    
    # Create data loaders using Dask backend
    train_loader, val_loader, test_loader, data_manager = create_tno_data_loaders(
        data_path=data_path,
        tno_config=tno_seq_config,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers,
        system_name=system_name,
        cdu_lim=cdu_lim,
        chunk_indices=chunk_indices,
        verbose=verbose
    )
    
    # Get number of CDUs from data manager
    num_cdus = data_manager.num_cdus
    
    # Create denormalization function
    denormalize_fn = None
    if data_manager.norm_handler is not None:
        denormalize_fn = create_denormalize_fn(data_manager.norm_handler)
    
    # Create model
    model = create_tno_model(
        input_dim=input_dim,
        output_dim=output_dim,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        latent_dim=latent_dim,
        branch_type=branch_type,
        tbranch_type=tbranch_type
    )
    
    if verbose:
        print(f"\nModel created with {model.get_num_parameters()['total']} parameters")
    
    # Create trainer with physics-informed loss
    trainer = create_tno_trainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lambda_data=lambda_data,
        lambda_physics=lambda_physics,
        lambda_temp_order=lambda_temp_order,
        lambda_positivity=lambda_positivity,
        lambda_smoothness=lambda_smoothness,
        lambda_energy=lambda_energy,
        lambda_mass_conservation=lambda_mass_conservation,
        lambda_monotonicity=lambda_monotonicity,
        lambda_thermodynamic_cop=lambda_thermodynamic_cop,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        denormalize_fn=denormalize_fn,
        device=device,
        use_physics_validator=use_physics_validator,
        system_name=system_name,
        num_cdus=num_cdus,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        project_name=project_name
    )
    
    # Train model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=patience
    )
    
    # Finish wandb if used
    if use_wandb:
        trainer.finish_wandb()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print("=" * 60)
    
    return model, trainer, history

