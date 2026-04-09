"""
Operating regime visualization.
Scatter plots colored by detected operating regime.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_regime_plot(args):
    """Create regime scatter plot for a single input-output pair."""
    input_name, output_name, regime_data, output_dir = args
    
    if not regime_data:
        return (input_name, output_name, False)
    
    try:
        x_data = regime_data.get('x_data', np.array([]))
        y_data = regime_data.get('y_data', np.array([]))
        labels = regime_data.get('labels', np.array([]))
        regimes = regime_data.get('regimes', [])
        n_regimes = regime_data.get('n_regimes', 0)
        silhouette = regime_data.get('silhouette_score', 0)
        
        if len(x_data) == 0 or len(labels) == 0:
            return (input_name, output_name, False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Scatter colored by regime
        ax1 = axes[0]
        colors = plt.cm.Set1(np.linspace(0, 1, n_regimes))
        
        # Subsample if needed
        if len(x_data) > 3000:
            idx = np.random.choice(len(x_data), 3000, replace=False)
            x_plot = x_data[idx]
            y_plot = y_data[idx]
            labels_plot = labels[idx]
        else:
            x_plot = x_data
            y_plot = y_data
            labels_plot = labels
        
        for regime_id in range(n_regimes):
            mask = labels_plot == regime_id
            ax1.scatter(x_plot[mask], y_plot[mask], 
                       c=[colors[regime_id]], 
                       alpha=0.5, s=20,
                       label=f'Regime {regime_id + 1}')
        
        ax1.set_xlabel(input_name, fontsize=12)
        ax1.set_ylabel(output_name, fontsize=12)
        ax1.set_title(f'Operating Regimes (Silhouette: {silhouette:.3f})', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right: Regime characteristics
        ax2 = axes[1]
        
        if regimes:
            regime_labels = [f'Regime {r["regime_id"] + 1}' for r in regimes]
            slopes = [r['slope'] for r in regimes]
            r2_values = [r['r2'] for r in regimes]
            n_points = [r['n_points'] for r in regimes]
            
            x_pos = np.arange(len(regimes))
            width = 0.35
            
            # Plot slopes and R²
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(x_pos - width/2, slopes, width, 
                           label='Slope', color='steelblue', edgecolor='black')
            bars2 = ax2_twin.bar(x_pos + width/2, r2_values, width,
                                label='R²', color='coral', edgecolor='black')
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(regime_labels)
            ax2.set_xlabel('Operating Regime', fontsize=12)
            ax2.set_ylabel('Slope', fontsize=12, color='steelblue')
            ax2_twin.set_ylabel('R²', fontsize=12, color='coral')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            
            ax2.tick_params(axis='y', labelcolor='steelblue')
            ax2_twin.tick_params(axis='y', labelcolor='coral')
            
            # Add point counts
            for i, (bar, n) in enumerate(zip(bars1, n_points)):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        ax2.get_ylim()[1] * 0.95,
                        f'n={n}', ha='center', va='top', fontsize=8)
            
            ax2.set_title('Regime Characteristics', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No regime data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.axis('off')
        
        plt.suptitle(f'Regime Classification: {output_name} vs {input_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'regimes_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create regime plot for {input_name} vs {output_name}: {e}")
        return (input_name, output_name, False)


def create_regime_scatter_plots(
    regime_data_list: List[Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create regime scatter plots for all input-output pairs.
    
    Args:
        regime_data_list: List of regime analysis results
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating regime scatter plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    args_list = []
    for regime_data in regime_data_list:
        if regime_data:
            input_name = regime_data.get('input', 'unknown')
            output_name = regime_data.get('output', 'unknown')
            args_list.append((input_name, output_name, regime_data, output_dir))
    
    if not args_list:
        logger.warning("No regime data to plot")
        return
    
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_regime_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} regime plots")


def create_regime_summary_plot(
    regime_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary of regime classification results.
    """
    logger.info("Creating regime summary plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Distribution of number of regimes
    ax1 = axes[0]
    regime_counts = regime_df['n_regimes'].value_counts().sort_index()
    ax1.bar(regime_counts.index, regime_counts.values, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Regimes', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Detected Regimes', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Silhouette score distribution
    ax2 = axes[1]
    ax2.hist(regime_df['silhouette_score'], bins=20, color='coral', edgecolor='black')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Good (>0.5)')
    ax2.axvline(x=0.25, color='orange', linestyle='--', linewidth=2, label='Fair (>0.25)')
    ax2.set_xlabel('Silhouette Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Regime Separation Quality', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Number of regimes by input
    ax3 = axes[2]
    inputs = regime_df['input'].unique()
    
    positions = []
    data = []
    for inp in inputs:
        data.append(regime_df[regime_df['input'] == inp]['n_regimes'].values)
        positions.append(inp)
    
    bp = ax3.boxplot(data, labels=positions, patch_artist=True)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Input Variable', fontsize=12)
    ax3.set_ylabel('Number of Regimes', fontsize=12)
    ax3.set_title('Regimes by Input Variable', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Operating Regime Detection Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
