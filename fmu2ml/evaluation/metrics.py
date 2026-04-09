"""
Metrics calculation for model evaluation.
"""
import numpy as np
from typing import Dict


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Ground truth values
    
    Returns:
    --------
    Dict[str, float] : Dictionary of metrics
    """
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
    
    # R² score
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    ss_res = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


class MetricsCalculator:
    """Calculate and track metrics over time"""
    
    def __init__(self):
        self.metrics_history = []
    
    def add_batch(self, predictions: np.ndarray, targets: np.ndarray):
        """Add batch metrics"""
        metrics = calculate_metrics(predictions, targets)
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all metrics"""
        if not self.metrics_history:
            return {}
        
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in self.metrics_history])
        
        return avg_metrics
    
    def reset(self):
        """Reset metrics history"""
        self.metrics_history = []
