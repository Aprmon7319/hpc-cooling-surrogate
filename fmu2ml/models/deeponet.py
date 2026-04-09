"""
DeepONet model for datacenter cooling prediction.
Refactored from deep_learning/models/deeponet.py
"""
import torch
import torch.nn as nn
from typing import List

from .base_model import BaseModel
from ..config import ModelConfig


class DeepONetCooling(BaseModel):
    """DeepONet model with branch and trunk networks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Branch network - processes input functions
        self.branch_net = self._build_branch_network(config)
        
        # Trunk network - learns basis functions
        self.trunk_nets = self._build_trunk_network(config)  
    
    
    def _build_branch_network(self, config: ModelConfig):
        """Build branch network for processing inputs"""
        layers = []
        input_dim = config.sequence_length * config.input_dim
        
        for i, hidden_dim in enumerate(config.deeponet_branch_layers):
            in_dim = input_dim if i == 0 else config.deeponet_branch_layers[i-1]
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(
            config.deeponet_branch_layers[-1], 
            config.deeponet_basis_dim
        ))
        
        return nn.Sequential(*layers)
    
    def _build_trunk_network(self, config: ModelConfig):
        """Build trunk network for basis functions"""
        # For each output location (CDU + global)
        trunk_nets = nn.ModuleList()
        
        # CDU trunk nets (49 CDUs * 11 outputs each)
        for _ in range(config.num_cdus):  # FIX: Use num_cdus
            net = nn.Sequential(
                nn.Linear(2, 64),  # CDU location encoding
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, config.deeponet_basis_dim * config.output_dim_per_cdu)
            )
            trunk_nets.append(net)
        
        # Global outputs trunk net
        global_net = nn.Sequential(
            nn.Linear(1, 64),  # Global encoding
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, config.deeponet_basis_dim * 51)  # 51 global outputs
        )
        trunk_nets.append(global_net)
        
        return trunk_nets  
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Flatten input for branch network
        x_flat = x.reshape(batch_size, -1)
        
        # Get branch outputs (coefficients)
        branch_out = self.branch_net(x_flat)  # (batch, basis_dim)
        
        outputs = []
        
        # Process each CDU
        for i in range(self.config.num_cdus):  # FIX: Use num_cdus
            # CDU location encoding
            cdu_loc = torch.tensor(
                [i // 7, i % 7], 
                dtype=torch.float32, 
                device=x.device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # Get trunk basis functions
            trunk_out = self.trunk_nets[i](cdu_loc)
            trunk_out = trunk_out.reshape(
                batch_size, 
                self.config.deeponet_basis_dim, 
                self.config.output_dim_per_cdu
            )
            
            # Compute output as inner product
            cdu_out = torch.einsum('bi,bio->bo', branch_out, trunk_out)
            outputs.append(cdu_out)
        
        # Process global outputs
        global_loc = torch.ones(batch_size, 1, device=x.device)
        global_trunk = self.trunk_nets[-1](global_loc)
        global_trunk = global_trunk.reshape(
            batch_size, 
            self.config.deeponet_basis_dim, 
            51
        )
        global_out = torch.einsum('bi,bio->bo', branch_out, global_trunk)
        
        # Combine all outputs
        all_outputs = torch.cat(outputs + [global_out], dim=1)
        
        return all_outputs
