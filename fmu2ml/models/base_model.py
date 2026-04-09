"""
Base model class for all neural network models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from fmu2ml.config import ModelConfig


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model
        
        Parameters:
        -----------
        config : ModelConfig
            Model configuration
        """
        super().__init__()
        self.config = config
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters
        
        Returns:
        --------
        int
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        """
        Save model checkpoint
        
        Parameters:
        -----------
        filepath : str
            Path to save checkpoint
        optimizer : torch.optim.Optimizer, optional
            Optimizer state to save
        epoch : int, optional
            Current epoch number
        **kwargs : dict
            Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'model_type': self.model_type,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        checkpoint.update(kwargs)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Parameters:
        -----------
        filepath : str
            Path to checkpoint file
        optimizer : torch.optim.Optimizer, optional
            Optimizer to load state into
        device : str, optional
            Device to map checkpoint to
        
        Returns:
        --------
        Dict[str, Any]
            Checkpoint data
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(filepath, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_model_summary(self) -> str:
        """
        Get model summary string
        
        Returns:
        --------
        str
            Model summary
        """
        summary = [
            f"Model: {self.model_type}",
            f"Parameters: {self.get_num_parameters():,}",
            f"Config: {self.config.to_dict()}",
        ]
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        return f"{self.model_type}(params={self.get_num_parameters():,})"
