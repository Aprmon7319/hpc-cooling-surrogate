"""
Compare different models (FMU vs ML models).
Refactored from deep_learning/compare_models.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

from .metrics import calculate_metrics


class ModelComparator:
    """Compare multiple models"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, predictions: np.ndarray, 
                         targets: np.ndarray):
        """Add results for a model"""
        metrics = calculate_metrics(predictions, targets)
        self.results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models
        
        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        comparison_data = []
        
        for model_name, data in self.results.items():
            row = {'Model': model_name}
            row.update(data['metrics'])
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def print_comparison(self):
        """Print comparison table"""
        df = self.compare_models()
        print("" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
    
    def get_best_model(self, metric: str = 'MAE') -> str:
        """Get name of best model by metric"""
        df = self.compare_models()
        best_idx = df[metric].idxmin()
        return df.loc[best_idx, 'Model']
