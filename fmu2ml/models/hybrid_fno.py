"""
Hybrid FNO model combining GRU and Fourier Neural Operators for datacenter cooling.
Refactored from deep_learning/models/hybrid_fno.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseModel
from ..config import ModelConfig


class HybridFNOCooling(BaseModel):
    """Hybrid model combining GRU temporal processing with FNO spatial coupling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sequence_length = config.sequence_length
        
        # Shared dimensions
        self.input_dim_per_cdu = 2  # Each CDU has 2 input variables
        self.hidden_dim = 128
        self.num_gru_layers = 2
        
        # Batch-parallel processing for all CDUs
        self.cdu_gru = nn.GRU(
            input_size=self.input_dim_per_cdu,
            hidden_size=self.hidden_dim,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=0.1 if self.num_gru_layers > 1 else 0
        )
        
        # Global context GRU for global variables
        self.global_gru = nn.GRU(
            input_size=1,  # Single global variable
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Spatial coupling network (lightweight FNO for CDU interactions)
        self.spatial_coupling = SimpleFNO2d(
            in_channels=128,
            out_channels=64,
            modes=3,  # Lower modes for interaction patterns
            width=32,
            n_layers=2
        )
        
        # Physics-informed layer for energy balance and flow constraints
        self.physics_informed_layer = PhysicsInformedLayer(64)
        
        # CDU-specific decoder (shared weights)
        self.cdu_decoder = self._build_cdu_decoder()
        
        # Global outputs head
        self.global_head = nn.Sequential(
            nn.Linear(64 * 49 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 51)  # V_flow, PUE, 49 HTC values
        )
        
        # Initialize positional encodings buffer
        self.register_buffer('cdu_positions', torch.randn(49, 128) * 0.02)
        
        # Layer normalization for residual connections
        self.layer_norm_cdu = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_coupled = nn.LayerNorm(64)
        
        # Input projection for residual connection
        self.input_projection = nn.Linear(49 * 2, 49 * self.hidden_dim)
        
        # Hidden state buffers (for stateful prediction)
        self.cdu_hidden_states: Optional[torch.Tensor] = None
        self.global_hidden_state: Optional[torch.Tensor] = None
        self.stateful_mode = False
    
    def _build_cdu_decoder(self):
        """Decoder for single CDU outputs"""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 11)  # 11 outputs per CDU
        )
    
    def enable_stateful_prediction(self, batch_size: int = 1):
        """Enable stateful mode for sequential prediction"""
        self.stateful_mode = True
        device = next(self.parameters()).device
        self.cdu_hidden_states = torch.zeros(
            self.num_gru_layers, batch_size, 49, self.hidden_dim, 
            device=device
        )
        self.global_hidden_state = torch.zeros(
            2, batch_size, 128, 
            device=device
        )
    
    def disable_stateful_prediction(self):
        """Disable stateful mode"""
        self.stateful_mode = False
        self.cdu_hidden_states = None
        self.global_hidden_state = None
    
    def reset_states(self, batch_size: int = 1):
        """Reset hidden states"""
        if self.stateful_mode:
            device = next(self.parameters()).device
            self.cdu_hidden_states = torch.zeros(
                self.num_gru_layers, batch_size, 49, self.hidden_dim, 
                device=device
            )
            self.global_hidden_state = torch.zeros(
                2, batch_size, 128, 
                device=device
            )
    
    def process_cdus_batch(self, cdu_data):
        """Process all CDUs in parallel for efficiency"""
        batch_size, seq_len, num_cdus, features = cdu_data.shape
        
        # Reshape to process all CDUs in parallel batches
        # [batch, seq, num_cdus, features] -> [batch * num_cdus, seq, features]
        cdu_data_flat = cdu_data.permute(0, 2, 1, 3).reshape(
            batch_size * num_cdus, seq_len, features
        )
        
        # Get hidden state if in stateful mode
        h0 = None
        if self.stateful_mode and self.cdu_hidden_states is not None:
            # Reshape: [layers, batch, num_cdus, hidden] -> [layers, batch * num_cdus, hidden]
            h0 = self.cdu_hidden_states.view(
                self.num_gru_layers, batch_size * num_cdus, self.hidden_dim
            )
        
        # Process through GRU
        output, hn = self.cdu_gru(cdu_data_flat, h0)
        
        # Store hidden state for next call if in stateful mode
        if self.stateful_mode:
            # Reshape back: [layers, batch * num_cdus, hidden] -> [layers, batch, num_cdus, hidden]
            self.cdu_hidden_states = hn.view(
                self.num_gru_layers, batch_size, num_cdus, self.hidden_dim
            ).detach()
        
        # Use the last output as features
        features = output[:, -1, :]  # [batch * num_cdus, hidden]
        
        # Reshape back: [batch * num_cdus, hidden] -> [batch, num_cdus, hidden]
        return features.view(batch_size, num_cdus, self.hidden_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Split inputs
        cdu_data = x[:, :, :98].reshape(batch_size, seq_len, 49, 2)
        global_data = x[:, :, 98:99]
        
        # Extract heat load for physics-informed processing
        cdu_heat_load = cdu_data[:, -1, :, 0]  # Last timestep, first feature
        
        # Process all CDUs in parallel
        cdu_features = self.process_cdus_batch(cdu_data)
        
        # Add residual connection from input
        cdu_data_flat = cdu_data[:, -1, :, :].reshape(batch_size, -1)
        residual = self.input_projection(cdu_data_flat).view(
            batch_size, 49, self.hidden_dim
        )
        cdu_features = cdu_features + residual
        
        # Apply layer normalization
        cdu_features = self.layer_norm_cdu(cdu_features)
        
        # Process global data with GRU
        h0_global = self.global_hidden_state if self.stateful_mode else None
        global_output, hn_global = self.global_gru(global_data, h0_global)
        global_features = global_output[:, -1, :]  # (batch, hidden_dim)
        
        # Store global hidden state if in stateful mode
        if self.stateful_mode:
            self.global_hidden_state = hn_global.detach()
        
        # Add positional encoding for CDU locations
        positions = self.cdu_positions.unsqueeze(0).expand(batch_size, -1, -1)
        cdu_features = cdu_features + positions
        
        # Reshape for spatial processing: (batch, hidden_dim, 7, 7)
        cdu_features_grid = cdu_features.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, 7, 7
        )
        
        # Apply spatial coupling (FNO captures CDU interactions)
        coupled_features = self.spatial_coupling(cdu_features_grid)
        
        # Reshape back: (batch, 64, 49)
        coupled_features = coupled_features.reshape(batch_size, 64, 49)
        coupled_features = coupled_features.transpose(1, 2)  # (batch, 49, 64)
        
        # Apply physics-informed layer
        coupled_features = self.physics_informed_layer(coupled_features, cdu_heat_load)
        
        # Apply layer normalization
        coupled_features = self.layer_norm_coupled(coupled_features)
        
        # Decode each CDU with shared weights using batch operations
        cdu_features_flat = coupled_features.reshape(batch_size * 49, 64)
        cdu_outputs_flat = self.cdu_decoder(cdu_features_flat)
        cdu_outputs = cdu_outputs_flat.reshape(batch_size, 49 * 11)
        
        # Combine for global outputs
        all_features = torch.cat([
            coupled_features.reshape(batch_size, -1),
            global_features
        ], dim=1)
        
        global_outputs = self.global_head(all_features)
        
        # Final output
        return torch.cat([cdu_outputs, global_outputs], dim=1)


class PhysicsInformedLayer(nn.Module):
    """Physics-informed layer for energy balance and flow constraints"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.energy_balance = nn.Linear(hidden_dim, hidden_dim)
        self.flow_conservation = nn.Linear(hidden_dim, hidden_dim)
        self.heat_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(self, x, heat_load):
        # Encode physics constraints into features
        energy_features = self.energy_balance(x)
        flow_features = self.flow_conservation(x)
        
        # Heat load aware processing
        heat_context = self.heat_encoder(heat_load.unsqueeze(-1))
        
        # Apply heat load context to energy features
        physics_features = energy_features * torch.sigmoid(heat_context)
        
        # Combine with flow conservation features
        physics_features = physics_features + flow_features
        
        return x + 0.1 * physics_features  # Add as residual with small scaling


class SimpleFNO2d(nn.Module):
    """Lightweight FNO for spatial coupling"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int, 
                 width: int, n_layers: int):
        super().__init__()
        self.modes = modes
        self.width = width
        
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        
        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes)
            for _ in range(n_layers)
        ])
        
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(n_layers)
        ])
        
        # Add layer normalization after each spectral convolution
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(8, width), width)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Conv2d(width, out_channels, 1)
    
    def forward(self, x):
        x = self.fc0(x)
        
        for i, (conv, w, norm) in enumerate(zip(self.convs, self.ws, self.norms)):
            # Store for residual
            x_in = x
            
            # Apply spectral convolution and linear layer
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)
            
            # Apply normalization
            x = norm(x)
            
            # Add residual connection except for first layer
            if i > 0:
                x = x + x_in
        
        x = self.fc1(x)
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Better initialization scale
        self.scale = (1 / (in_channels * out_channels))**0.5
        
        # Store weights as real tensors with last dimension for real/imag
        self.weights1 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, 2))
    
    def compl_mul2d(self, input, weights):
        """Complex multiplication for 2D tensors"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        H, W = x.size(-2), x.size(-1)
        
        # Safety check
        assert self.modes1 <= H // 2, f"modes1 {self.modes1} too large for height {H}"
        assert self.modes2 <= W // 2 + 1, f"modes2 {self.modes2} too large for width {W}"
        
        # Store original dtype
        orig_dtype = x.dtype
        
        # Force float32 for FFT operations
        x_float32 = x.to(torch.float32)
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x_float32, norm='ortho')
        
        # Initialize output
        out_ft = torch.zeros(batchsize, self.out_channels, H, W//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        # Convert weights to complex
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        x = x.contiguous()
        
        # Convert back to original dtype if needed
        if x.dtype != orig_dtype:
            x = x.to(orig_dtype)
        
        return x
