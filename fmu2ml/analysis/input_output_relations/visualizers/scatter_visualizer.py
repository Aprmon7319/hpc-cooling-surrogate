import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_scatter_plot(args):
    """Create scatter plot for a single input variable. Used for parallel execution."""
    input_name, input_data, prepared_data, metrics_df, output_vars, output_dir = args
    
    n_outputs = len(output_vars)
    n_cols = 4
    n_rows = (n_outputs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    if input_data.ndim > 1:
        input_flat = input_data.flatten()
    else:
        input_flat = input_data
    
    for idx, output_name in enumerate(output_vars):
        ax = axes[idx]
        
        output_data = prepared_data['outputs'][output_name].flatten()
        
        if input_flat.ndim == 1 and len(input_flat) != len(output_data):
            input_plot = np.repeat(input_flat, output_data.size // input_flat.size)
        else:
            input_plot = input_flat
        
        mask = ~(np.isnan(input_plot) | np.isnan(output_data))
        input_clean = input_plot[mask]
        output_clean = output_data[mask]
        
        if len(input_clean) > 1000:
            sample_idx = np.random.choice(len(input_clean), 1000, replace=False)
            input_clean = input_clean[sample_idx]
            output_clean = output_clean[sample_idx]
        
        ax.scatter(input_clean, output_clean, alpha=0.5, s=10)
        
        pair_metrics = metrics_df[
            (metrics_df['input'] == input_name) &
            (metrics_df['output'] == output_name)
        ].iloc[0]
        
        x_line = np.linspace(input_clean.min(), input_clean.max(), 100)
        y_line = pair_metrics['linear_coef'] * x_line + pair_metrics['linear_intercept']
        ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')
        
        ax.text(
            0.05, 0.95,
            f"R² = {pair_metrics['r2_score']:.3f}\n"
            f"Pearson = {pair_metrics['pearson_r']:.3f}\n"
            f"MI = {pair_metrics['mutual_info']:.3f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8
        )
        
        ax.set_xlabel(input_name, fontsize=10)
        ax.set_ylabel(output_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(output_vars), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'Direct Effects: {input_name} on All Outputs',
        fontsize=16,
        y=1.00
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'scatter_plots_{input_name}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    return input_name


def create_scatter_plots(
    prepared_data: Dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    output_dir: str,
    n_workers: int = 8
):
    """Create scatter plots with regression lines for all pairs using parallel processing."""
    logger.info("Creating scatter plots in parallel...")
    
    input_vars = ['Q_flow', 'T_Air', 'T_ext']
    output_vars = list(metrics_df['output'].unique())
    
    # Prepare arguments for parallel processing
    args_list = []
    for input_name in input_vars:
        input_data = prepared_data['inputs'][input_name]
        args_list.append((input_name, input_data, prepared_data, metrics_df, output_vars, output_dir))
    
    # Use multiprocessing pool
    with mp.Pool(processes=min(n_workers, len(input_vars))) as pool:
        results = pool.map(_create_single_scatter_plot, args_list)
    
    logger.info(f"Created scatter plots for {len(results)} inputs")