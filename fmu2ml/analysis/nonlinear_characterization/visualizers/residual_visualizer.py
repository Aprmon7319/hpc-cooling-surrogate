"""
Residual analysis plots.
Checks if non-linear models capture structure better than linear models.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_residual_plot(args):
    """Create residual plot for a single input-output pair."""
    input_name, output_name, prediction_data, output_dir = args
    
    if not prediction_data:
        return (input_name, output_name, False)
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        x_data = prediction_data.get('x_data', np.array([]))
        y_data = prediction_data.get('y_data', np.array([]))
        
        if len(x_data) == 0:
            return (input_name, output_name, False)
        
        degrees = [1, 2, 3]
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        labels = ['Linear', 'Quadratic', 'Cubic']
        
        for i, degree in enumerate(degrees):
            residuals_key = f'residuals_degree_{degree}'
            r2_key = f'r2_degree_{degree}'
            
            if residuals_key not in prediction_data:
                continue
            
            residuals = prediction_data[residuals_key]
            r2 = prediction_data.get(r2_key, 0)
            
            # Top row: Residuals vs X
            ax_vs_x = axes[0, i]
            
            # Subsample for plotting
            if len(x_data) > 1000:
                idx = np.random.choice(len(x_data), 1000, replace=False)
                x_plot = x_data[idx]
                res_plot = residuals[idx]
            else:
                x_plot = x_data
                res_plot = residuals
            
            ax_vs_x.scatter(x_plot, res_plot, alpha=0.3, s=10, color=colors[i])
            ax_vs_x.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax_vs_x.set_xlabel(input_name, fontsize=10)
            ax_vs_x.set_ylabel('Residuals', fontsize=10)
            ax_vs_x.set_title(f'{labels[i]} (R²={r2:.4f})', fontsize=11)
            ax_vs_x.grid(True, alpha=0.3)
            
            # Add LOESS smoothing line to detect patterns
            try:
                from scipy.ndimage import uniform_filter1d
                sorted_idx = np.argsort(x_plot)
                x_sorted = x_plot[sorted_idx]
                res_sorted = res_plot[sorted_idx]
                smoothed = uniform_filter1d(res_sorted, size=max(10, len(res_sorted) // 20))
                ax_vs_x.plot(x_sorted, smoothed, color='black', linewidth=2, label='Trend')
            except:
                pass
            
            # Bottom row: Residual histogram with normality test
            ax_hist = axes[1, i]
            
            ax_hist.hist(residuals, bins=50, density=True, alpha=0.7, color=colors[i], edgecolor='black')
            
            # Fit normal distribution
            mu, std = residuals.mean(), residuals.std()
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = stats.norm.pdf(x_norm, mu, std)
            ax_hist.plot(x_norm, y_norm, 'k-', linewidth=2, label='Normal fit')
            
            # Normality test
            if len(residuals) >= 20:
                _, p_value = stats.normaltest(residuals)
                normality_text = f'Normality p={p_value:.4f}'
            else:
                normality_text = 'Too few samples'
            
            ax_hist.set_xlabel('Residual Value', fontsize=10)
            ax_hist.set_ylabel('Density', fontsize=10)
            ax_hist.set_title(f'{labels[i]} Residuals\n{normality_text}', fontsize=10)
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)
            
            # Add statistics
            skew = stats.skew(residuals)
            kurt = stats.kurtosis(residuals)
            ax_hist.text(0.95, 0.95, 
                        f'μ={mu:.3f}\nσ={std:.3f}\nskew={skew:.3f}\nkurt={kurt:.3f}',
                        transform=ax_hist.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Residual Analysis: {output_name} vs {input_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'residuals_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create residual plot for {input_name} vs {output_name}: {e}")
        return (input_name, output_name, False)


def create_residual_plots(
    prediction_data_dict: Dict[tuple, Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create residual plots for all input-output pairs.
    
    Args:
        prediction_data_dict: Dictionary mapping (input, output) to prediction data
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating residual plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    args_list = []
    for (input_name, output_name), pred_data in prediction_data_dict.items():
        args_list.append((input_name, output_name, pred_data, output_dir))
    
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_residual_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} residual plots")


def create_residual_summary_plot(
    prediction_data_dict: Dict[tuple, Dict],
    output_dir: str
):
    """
    Create summary plot of residual statistics across all pairs.
    """
    logger.info("Creating residual summary plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    
    for (input_name, output_name), pred_data in prediction_data_dict.items():
        if not pred_data:
            continue
        
        for degree in [1, 2, 3]:
            residuals_key = f'residuals_degree_{degree}'
            if residuals_key not in pred_data:
                continue
            
            residuals = pred_data[residuals_key]
            
            summary_data.append({
                'input': input_name,
                'output': output_name,
                'degree': degree,
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            })
    
    if not summary_data:
        return
    
    df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard deviation by degree
    ax1 = axes[0, 0]
    for degree in [1, 2, 3]:
        deg_data = df[df['degree'] == degree]
        ax1.scatter(range(len(deg_data)), deg_data['std'], 
                   label=f'Degree {degree}', alpha=0.7, s=50)
    ax1.set_xlabel('Input-Output Pair', fontsize=10)
    ax1.set_ylabel('Residual Std Dev', fontsize=10)
    ax1.set_title('Residual Standard Deviation by Polynomial Degree', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Skewness distribution
    ax2 = axes[0, 1]
    for i, degree in enumerate([1, 2, 3]):
        deg_data = df[df['degree'] == degree]
        ax2.hist(deg_data['skewness'], bins=20, alpha=0.5, label=f'Degree {degree}')
    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.set_xlabel('Skewness', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Distribution of Residual Skewness', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Kurtosis distribution
    ax3 = axes[1, 0]
    for i, degree in enumerate([1, 2, 3]):
        deg_data = df[df['degree'] == degree]
        ax3.hist(deg_data['kurtosis'], bins=20, alpha=0.5, label=f'Degree {degree}')
    ax3.axvline(x=0, color='black', linestyle='--')
    ax3.set_xlabel('Kurtosis', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Distribution of Residual Kurtosis', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Std reduction from linear to cubic
    ax4 = axes[1, 1]
    inputs = df['input'].unique()
    outputs = df['output'].unique()
    
    std_reduction = []
    for inp in inputs:
        for out in outputs:
            pair_data = df[(df['input'] == inp) & (df['output'] == out)]
            if len(pair_data) >= 3:
                linear_std = pair_data[pair_data['degree'] == 1]['std'].values
                cubic_std = pair_data[pair_data['degree'] == 3]['std'].values
                if len(linear_std) > 0 and len(cubic_std) > 0:
                    reduction = (linear_std[0] - cubic_std[0]) / linear_std[0] * 100
                    std_reduction.append(reduction)
    
    if std_reduction:
        ax4.hist(std_reduction, bins=20, color='steelblue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('% Reduction in Std Dev (Linear → Cubic)', fontsize=10)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.set_title('Std Dev Reduction from Linear to Cubic Model', fontsize=11)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
