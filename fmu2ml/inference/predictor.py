"""
Model inference and prediction.
Refactored from deep_learning/inference.py
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

from ..data.processors import NormalizationHandler
from ..models import create_model


class CoolingModelPredictor:
    """Main inference class for cooling models"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.model_type = checkpoint.get('args', {}).get('model', 'fno')
        
        # Create and load model
        self.model = create_model(self.model_type, self.config)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization handler
        self.norm_handler = self._load_normalization(checkpoint)
        
        print(f"✓ Model {self.model_type} loaded successfully")
        print(f"✓ Device: {self.device}")
    
    def _load_normalization(self, checkpoint):
        """Load normalization stats"""
        if 'norm_stats_path' in checkpoint and checkpoint['norm_stats_path']:
            stats_path = checkpoint['norm_stats_path']
            if Path(stats_path).exists():
                return NormalizationHandler(stats_path)
        
        # Try to find in checkpoint directory
        checkpoint_dir = Path(self.checkpoint_path).parent
        stats_path = checkpoint_dir / f'norm_stats_{self.model_type}.npz'
        if stats_path.exists():
            return NormalizationHandler(stats_path)
        
        print("WARNING: No normalization stats found")
        return NormalizationHandler()
    
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Make prediction
        
        Parameters:
        -----------
        input_data : array-like
            Input sequence of shape (seq_len, input_dim) or (batch, seq_len, input_dim)
        
        Returns:
        --------
        dict : Predicted outputs
        """
        with torch.no_grad():
            # Normalize input
            if isinstance(input_data, np.ndarray):
                input_data = self.norm_handler.normalize_input(input_data)
                input_tensor = torch.FloatTensor(input_data).to(self.device)
            else:
                input_tensor = input_data.to(self.device)
            
            # Add batch dimension if needed
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Denormalize output
            output_np = output.cpu().numpy()
            output_denorm = self.norm_handler.denormalize_output(output_np)
            
            # Parse output
            return self._parse_output(output_denorm[0] if output_denorm.shape[0] == 1 else output_denorm)
    
    def _parse_output(self, output_array):
        """Parse output array into structured format"""
        results = {}
        
        # CDU outputs (49 CDUs × 11 variables)
        for i in range(49):
            base_idx = i * 11
            results[f'CDU_{i+1}'] = {
                'V_flow_prim_GPM': float(output_array[base_idx + 0]),
                'V_flow_sec_GPM': float(output_array[base_idx + 1]),
                'W_flow_CDUP_kW': float(output_array[base_idx + 2]),
                'T_prim_s_C': float(output_array[base_idx + 3]),
                'T_prim_r_C': float(output_array[base_idx + 4]),
                'T_sec_s_C': float(output_array[base_idx + 5]),
                'T_sec_r_C': float(output_array[base_idx + 6]),
                'p_prim_s_psig': float(output_array[base_idx + 7]),
                'p_prim_r_psig': float(output_array[base_idx + 8]),
                'p_sec_s_psig': float(output_array[base_idx + 9]),
                'p_sec_r_psig': float(output_array[base_idx + 10])
            }
        
        # Global outputs
        results['datacenter_V_flow_prim_GPM'] = float(output_array[539])
        results['pue'] = float(output_array[540])
        
        # HTC values
        htc_values = output_array[541:590]
        for i in range(49):
            results[f'CDU_{i+1}']['htc'] = float(htc_values[i])
        
        return results
