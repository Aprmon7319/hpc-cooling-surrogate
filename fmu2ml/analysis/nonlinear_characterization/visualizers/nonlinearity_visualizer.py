"""
Scatter plots with multiple polynomial fits overlaid.
Visualizes linear, quadratic, and cubic fits on the same plot.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_nonlinearity_plot(args):
    """Create scatter plot with multiple fits for a single input-output pair."""
    input_name, output_name, prediction_data, output_dir = args
    
    if not prediction_data:
        return (input_name, output_name, False)
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get data
        x_data = prediction_data.get('x_data', np.array([]))
        y_data = prediction_data.get('y_data', np.array([]))
        x_pred = prediction_data.get('x', np.array([]))
        
        if len(x_data) == 0:
            return (input_name, output_name, False)
        
        # Subsample if too many points
        if len(x_data) > 2000:
            sample_idx = np.random.choice(len(x_data), 2000, replace=False)
            x_plot = x_data[sample_idx]
            y_plot = y_data[sample_idx]
        else:
            x_plot = x_data
            y_plot = y_data
        
        # Scatter plot
        ax.scatter(x_plot, y_plot, alpha=0.3, s=10, color='gray', label='Data')
        
        # Plot each polynomial degree
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        labels = ['Linear', 'Quadratic', 'Cubic']
        linestyles = ['-', '--', '-.']
        
        for i, degree in enumerate([1, 2, 3]):
            y_key = f'y_degree_{degree}'
            r2_key = f'r2_degree_{degree}'
            
            if y_key in prediction_data:
                y_pred = prediction_data[y_key]
                r2 = prediction_data.get(r2_key, 0)
                
                ax.plot(x_pred, y_pred, color=colors[i], linewidth=2,
                       linestyle=linestyles[i],
                       label=f'{labels[i]} (R² = {r2:.4f})')
        
        ax.set_xlabel(input_name, fontsize=12)
        ax.set_ylabel(output_name, fontsize=12)
        ax.set_title(f'{output_name} vs {input_name}\nPolynomial Model Comparison', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'nonlinearity_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create nonlinearity plot for {input_name} vs {output_name}: {e}")
        return (input_name, output_name, False)


def create_nonlinearity_plots(
    prediction_data_dict: Dict[tuple, Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create scatter plots with multiple polynomial fits for all pairs.
    
    Args:
        prediction_data_dict: Dictionary mapping (input, output) to prediction data
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating nonlinearity plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments
    args_list = []
    for (input_name, output_name), pred_data in prediction_data_dict.items():
        args_list.append((input_name, output_name, pred_data, output_dir))
    
    # Use multiprocessing
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_nonlinearity_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} nonlinearity plots")
