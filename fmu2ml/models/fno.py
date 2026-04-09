"""
Fourier Neural Operator (FNO) for datacenter cooling
Refactored from deep_learning/models/fno.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fmu2ml.config import ModelConfig
from fmu2ml.models.base_model import BaseModel


class SpectralConv2d(nn.Module):
    """Spectral convolution layer for FNO"""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))**0.5
        
        self.weights1 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(
            in_channels, out_channels, self.modes1, self.modes2, 2))
    
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        H, W = x.size(-2), x.size(-1)
        
        assert self.modes1 <= H // 2, f"modes1 {self.modes1} too large for height {H}"
        assert self.modes2 <= W // 2 + 1, f"modes2 {self.modes2} too large for width {W}"
        
        x_float32 = x.to(torch.float32)
        x_ft = torch.fft.rfft2(x_float32, norm='ortho')
        
        out_ft = torch.zeros(batchsize, self.out_channels, H, W//2 + 1, 
                        dtype=torch.cfloat, device=x.device)
        
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)
        
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        x = x.contiguous()
        
        if x.dtype != x_float32.dtype:
            x = x.to(x.dtype)
            
        return x


class FourierBlock(nn.Module):
    """Fourier block combining spectral and spatial operations"""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(FourierBlock, self).__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.norm(x)
        return x


class FNOCooling(BaseModel):
    """
    Fourier Neural Operator for datacenter cooling prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Grid configuration
        self.grid_h, self.grid_w = 7, 7  # 7x7 grid for 49 CDUs
        self.modes1 = min(config.fno_modes, self.grid_h // 2)
        self.modes2 = min(config.fno_modes, self.grid_w // 2)
        self.width = config.fno_width
        
        # Create CDU position indices
        self.register_buffer('cdu_indices', self._create_cdu_indices())
        
        # Input channels: seq_len * 2 (CDU vars) + seq_len (global) + 2 (coordinates)
        input_channels = config.sequence_length * 3 + 2
        
        # Input projection
        self.fc0 = nn.Conv2d(input_channels, self.width, 1)
        
        # Fourier layers
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(self.width, self.width, self.modes1, self.modes2)
            for _ in range(config.fno_layers)
        ])
        
        # Output projection for CDUs
        self.fc1 = nn.Conv2d(self.width, 128, 1)
        self.fc2 = nn.Conv2d(128, config.output_dim_per_cdu, 1)
        
        # Global outputs (datacenter V_flow, PUE, 49 HTC values)
        self.global_head = nn.Sequential(
            nn.Linear(self.width * self.grid_h * self.grid_w, 256),
            nn.ReLU(),
            nn.Linear(256, 51)
        )
    
    def _create_cdu_indices(self):
        """Create indices for CDU positions in the grid"""
        indices = torch.zeros(49, 2, dtype=torch.long)
        for i in range(49):
            indices[i, 0] = i // 7  # row
            indices[i, 1] = i % 7   # col
        return indices
    
    def _create_coordinate_channels(self, x):
        """Create normalized coordinate channels"""
        batch_size = x.shape[0]
        y_coords = torch.linspace(0, 1, self.grid_h, device=x.device)
        x_coords = torch.linspace(0, 1, self.grid_w, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
        coords = coords.expand(batch_size, -1, -1, -1)
        return coords
    

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        expected_features = self.config.num_cdus * 2 + 1  # 2 vars per CDU + 1 global
        
        # Add shape validation
        if x.shape[2] != expected_features:
            raise ValueError(
                f"Expected {expected_features} input features, got {x.shape[2]}. "
                f"Shape: {x.shape}, Expected: (batch, seq={self.config.sequence_length}, features={expected_features})"
            )
        
        # Reshape input: (batch, seq, 99) -> (batch, seq, 49, 3)
        x_cdu = x[:, :, :98].reshape(batch_size, seq_len, 49, 2)  # 49 CDUs * 2 vars
        x_global = x[:, :, 98:99].unsqueeze(2).expand(-1, -1, 49, -1)  # Broadcast to all CDUs
        x_combined = torch.cat([x_cdu, x_global], dim=-1)  # (batch, seq, 49, 3)
        
        # Create grid tensor (batch, channels, H, W)
        # We need seq*3 channels: seq timesteps * 3 features per CDU
        x_grid = torch.zeros(
            batch_size, 
            seq_len * 3, 
            self.grid_h, 
            self.grid_w, 
            device=x.device,
            dtype=x.dtype
        )
        
        # Get row and column indices for all 49 CDUs
        rows = self.cdu_indices[:, 0]  # [49] - row indices
        cols = self.cdu_indices[:, 1]  # [49] - col indices
        
        # Reshape combined data: (batch, seq, 49, 3) -> (batch, 49, seq*3)
        x_combined_flat = x_combined.permute(0, 2, 1, 3).reshape(batch_size, 49, seq_len * 3)
        
        # Assign to grid using advanced indexing
        # x_grid shape: (batch, seq*3, 7, 7)
        # need to assign x_combined_flat[:, i, :] to x_grid[:, :, rows[i], cols[i]]
        for i in range(49):
            x_grid[:, :, rows[i], cols[i]] = x_combined_flat[:, i, :]
        
        # Add coordinate channels
        coords = self._create_coordinate_channels(x)  # (batch, 2, 7, 7)
        x_grid = torch.cat([x_grid, coords], dim=1)  # (batch, seq*3+2, 7, 7)
        
        # Apply input projection
        x_grid = self.fc0(x_grid)  # (batch, width, 7, 7)
        
        # Apply Fourier layers
        for block in self.fourier_blocks:
            x_grid = block(x_grid)
        
        # Extract CDU features at their grid positions
        # x_grid: (batch, width, 7, 7)
        # We want features at positions (rows[i], cols[i]) for each CDU
        cdu_features = torch.zeros(batch_size, 49, self.width, device=x.device, dtype=x.dtype)
        for i in range(49):
            cdu_features[:, i, :] = x_grid[:, :, rows[i], cols[i]]
        
        # Project to CDU outputs: (batch, 49, width) -> (batch, 49, 11)
        # Reshape for conv layers: (batch, 49, width) -> (batch, width, 49, 1)
        cdu_features = cdu_features.permute(0, 2, 1).unsqueeze(-1)  # (batch, width, 49, 1)
        cdu_features = self.fc1(cdu_features)  # (batch, 128, 49, 1)
        cdu_features = F.relu(cdu_features)
        cdu_outputs = self.fc2(cdu_features)  # (batch, 11, 49, 1)
        cdu_outputs = cdu_outputs.squeeze(-1).permute(0, 2, 1)  # (batch, 49, 11)
        
        # Global outputs
        x_flat = x_grid.reshape(batch_size, -1)  # (batch, width*7*7)
        global_outputs = self.global_head(x_flat)  # (batch, 51)
        
        # Combine outputs
        outputs = torch.cat([
            cdu_outputs.reshape(batch_size, -1),  # 49 * 11 = 539
            global_outputs  # 51
        ], dim=1)  # Total: 590
        
        return outputs
