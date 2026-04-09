"""
Normalization handler for consistent data normalization/denormalization
"""

import torch
import numpy as np
import os
from pathlib import Path
import json
from typing import Union, Tuple, Optional
import pandas as pd


class NormalizationHandler:
    """
    Centralized handler for consistent normalization/denormalization
    Works with both PyTorch tensors and NumPy arrays
    """
    
    def __init__(self, stats_path: Optional[str] = None):
        """
        Initialize with optional path to existing normalization stats
        
        Parameters:
        -----------
        stats_path : str, optional
            Path to saved normalization statistics
        """
        self.mean_in = None
        self.std_in = None
        self.mean_out = None
        self.std_out = None
        self.epsilon = 1e-8  # Small value to prevent division by zero
        
        if stats_path is not None and os.path.exists(stats_path):
            self.load_stats(stats_path)
    
    def compute_stats(
        self,
        train_data: pd.DataFrame,
        input_cols: list,
        output_cols: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute normalization statistics from training data
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        input_cols : list
            Input column names
        output_cols : list
            Output column names
        
        Returns:
        --------
        Tuple of (mean_in, std_in, mean_out, std_out)
        """
        # Extract input and output data
        input_data = train_data[input_cols].values.astype(np.float64)
        output_data = train_data[output_cols].values.astype(np.float64)
        
        # Compute statistics
        self.mean_in = input_data.mean(axis=0)
        self.std_in = input_data.std(axis=0, ddof=1)  # Use sample std
        self.mean_out = output_data.mean(axis=0)
        self.std_out = output_data.std(axis=0, ddof=1)
        
        # Replace zero or very small std with 1 to avoid division by zero
        self.std_in = np.where(np.abs(self.std_in) < self.epsilon, 1.0, self.std_in)
        self.std_out = np.where(np.abs(self.std_out) < self.epsilon, 1.0, self.std_out)
        
        # Convert to float32 to save memory
        self.mean_in = self.mean_in.astype(np.float32)
        self.std_in = self.std_in.astype(np.float32)
        self.mean_out = self.mean_out.astype(np.float32)
        self.std_out = self.std_out.astype(np.float32)
        
        return self.mean_in, self.std_in, self.mean_out, self.std_out
    
    def normalize_input(
        self,
        input_data: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Normalize input data consistently
        
        Parameters:
        -----------
        input_data : torch.Tensor or np.ndarray
            Input data to normalize
        
        Returns:
        --------
        Normalized input data (same type as input)
        """
        if self.mean_in is None or self.std_in is None:
         raise ValueError("Normalization statistics not computed. Call compute_stats() first.")
    
        if isinstance(input_data, torch.Tensor):
            device = input_data.device
            dtype = input_data.dtype
            
            mean_in = torch.tensor(self.mean_in, device=device, dtype=dtype)
            std_in = torch.tensor(self.std_in, device=device, dtype=dtype)
            
            # ADD: Handle zero std
            std_in = torch.where(torch.abs(std_in) < self.epsilon, 
                                torch.ones_like(std_in), std_in)
            
            return (input_data - mean_in) / std_in
        else:
            input_float64 = input_data.astype(np.float64)
            
            # ADD: Handle zero std
            std_safe = np.where(np.abs(self.std_in) < self.epsilon, 1.0, self.std_in)
            
            normalized = (input_float64 - self.mean_in.astype(np.float64)) / std_safe.astype(np.float64)
            return normalized.astype(input_data.dtype)
    
    
    def denormalize_input(
        self,
        normalized_input: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize model inputs consistently"""
        if isinstance(normalized_input, torch.Tensor):
            device = normalized_input.device
            dtype = normalized_input.dtype
            
            mean_in = torch.tensor(self.mean_in, device=device, dtype=dtype)
            std_in = torch.tensor(self.std_in, device=device, dtype=dtype)
            
            return normalized_input * std_in + mean_in
        else:
            normalized_float64 = normalized_input.astype(np.float64)
            denormalized = normalized_float64 * self.std_in.astype(np.float64) + self.mean_in.astype(np.float64)
            return denormalized.astype(normalized_input.dtype)
    
    def normalize_output(
        self,
        output_data: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Normalize output data consistently"""
        if isinstance(output_data, torch.Tensor):
            device = output_data.device
            dtype = output_data.dtype
            
            mean_out = torch.tensor(self.mean_out, device=device, dtype=dtype)
            std_out = torch.tensor(self.std_out, device=device, dtype=dtype)
            
            return (output_data - mean_out) / std_out
        else:
            output_float64 = output_data.astype(np.float64)
            normalized = (output_float64 - self.mean_out.astype(np.float64)) / self.std_out.astype(np.float64)
            return normalized.astype(output_data.dtype)
    
    def denormalize_output(
        self,
        normalized_output: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize model outputs consistently"""
        if isinstance(normalized_output, torch.Tensor):
            device = normalized_output.device
            dtype = normalized_output.dtype
            
            mean_out = torch.tensor(self.mean_out, device=device, dtype=dtype)
            std_out = torch.tensor(self.std_out, device=device, dtype=dtype)
            
            return normalized_output * std_out + mean_out
        else:
            normalized_float64 = normalized_output.astype(np.float64)
            denormalized = normalized_float64 * self.std_out.astype(np.float64) + self.mean_out.astype(np.float64)
            return denormalized.astype(normalized_output.dtype)
    
    def save_stats(self, output_path: str):
        """
        Save normalization statistics to file
        
        Parameters:
        -----------
        output_path : str
            Path to save statistics (.npz or .json)
        """
        stats_dict = {
            'mean_in': self.mean_in.tolist() if hasattr(self.mean_in, 'tolist') else self.mean_in,
            'std_in': self.std_in.tolist() if hasattr(self.std_in, 'tolist') else self.std_in,
            'mean_out': self.mean_out.tolist() if hasattr(self.mean_out, 'tolist') else self.mean_out,
            'std_out': self.std_out.tolist() if hasattr(self.std_out, 'tolist') else self.std_out
        }
        
        # Save as both NPZ (for numpy compatibility) and JSON (for readability)
        np.savez(output_path, **{k: np.array(v, dtype=np.float32) for k, v in stats_dict.items()})
        
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def load_stats(self, stats_path: str):
        """
        Load normalization statistics from file
        
        Parameters:
        -----------
        stats_path : str
            Path to load statistics from
        """
        stats_path = Path(stats_path)
        
        if stats_path.suffix == '.json':
            with open(stats_path, 'r') as f:
                stats_dict = json.load(f)
            self.mean_in = np.array(stats_dict['mean_in'], dtype=np.float32)
            self.std_in = np.array(stats_dict['std_in'], dtype=np.float32)
            self.mean_out = np.array(stats_dict['mean_out'], dtype=np.float32)
            self.std_out = np.array(stats_dict['std_out'], dtype=np.float32)
        else:
            stats = np.load(stats_path)
            self.mean_in = stats['mean_in'].astype(np.float32)
            self.std_in = stats['std_in'].astype(np.float32)
            self.mean_out = stats['mean_out'].astype(np.float32)
            self.std_out = stats['std_out'].astype(np.float32)
        
        # Safety check for zero std
        self.std_in = np.where(np.abs(self.std_in) < self.epsilon, 1.0, self.std_in)
        self.std_out = np.where(np.abs(self.std_out) < self.epsilon, 1.0, self.std_out)
