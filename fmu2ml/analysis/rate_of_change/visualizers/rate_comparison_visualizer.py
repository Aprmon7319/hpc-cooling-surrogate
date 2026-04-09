"""
Rate Comparison Visualizer

Creates dual scatter plots comparing:
- Output vs Input Level
- Output vs Input Rate (derivative)
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_rate_comparison(args):
    """Create rate comparison plot for single input-output pair. Used for parallel execution."""
    input_name, output_name, input_level, input_rate, output_level, metrics, output_dir = args
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Flatten data
        def flatten(arr):
            if arr.ndim > 1:
                return arr.flatten()
            return arr
        
        input_level_flat = flatten(input_level)
        input_rate_flat = flatten(input_rate)
        output_level_flat = flatten(output_level)
        
        # Ensure same length
        min_len = min(len(input_level_flat), len(input_rate_flat), len(output_level_flat))
        input_level_flat = input_level_flat[:min_len]
        input_rate_flat = input_rate_flat[:min_len]
        output_level_flat = output_level_flat[:min_len]
        
        # Remove NaN/Inf
        mask = ~(np.isnan(input_level_flat) | np.isnan(input_rate_flat) | 
                 np.isnan(output_level_flat) | np.isinf(input_rate_flat))
        
        if np.sum(mask) < 10:
            plt.close()
            return (input_name, output_name, False)
        
        input_level_clean = input_level_flat[mask]
        input_rate_clean = input_rate_flat[mask]
        output_level_clean = output_level_flat[mask]
        
        # Subsample if too many points
        if len(input_level_clean) > 2000:
            idx = np.random.choice(len(input_level_clean), 2000, replace=False)
            input_level_clean = input_level_clean[idx]
            input_rate_clean = input_rate_clean[idx]
            output_level_clean = output_level_clean[idx]
        
        # Left plot: Output vs Input Level
        ax1 = axes[0]
        scatter1 = ax1.scatter(input_level_clean, output_level_clean, 
                               alpha=0.5, s=10, c='#3498db')
        
        # Add regression line
        z = np.polyfit(input_level_clean, output_level_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(input_level_clean.min(), input_level_clean.max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
        
        level_corr = metrics.get('level_pearson_r', np.nan) if metrics else np.nan
        level_r2 = metrics.get('level_r2', np.nan) if metrics else np.nan
        
        ax1.text(0.05, 0.95, f"Pearson r = {level_corr:.3f}\nR² = {level_r2:.3f}",
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel(f'{input_name} Level', fontsize=11)
        ax1.set_ylabel(output_name, fontsize=11)
        ax1.set_title(f'{output_name} vs {input_name} Level', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Right plot: Output vs Input Rate
        ax2 = axes[1]
        scatter2 = ax2.scatter(input_rate_clean, output_level_clean,
                               alpha=0.5, s=10, c='#e74c3c')
        
        # Add regression line
        z = np.polyfit(input_rate_clean, output_level_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(input_rate_clean.min(), input_rate_clean.max(), 100)
        ax2.plot(x_line, p(x_line), 'b-', linewidth=2, label='Linear fit')
        
        rate_corr = metrics.get('rate_level_pearson_r', np.nan) if metrics else np.nan
        rate_r2 = metrics.get('rate_level_r2', np.nan) if metrics else np.nan
        
        ax2.text(0.05, 0.95, f"Pearson r = {rate_corr:.3f}\nR² = {rate_r2:.3f}",
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        ax2.set_xlabel(f'd{input_name}/dt (Rate)', fontsize=11)
        ax2.set_ylabel(output_name, fontsize=11)
        ax2.set_title(f'{output_name} vs {input_name} Rate of Change', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Determine dominant effect
        dominant = metrics.get('dominant_effect', 'unknown') if metrics else 'unknown'
        plt.suptitle(
            f'Level vs Rate Effect: {input_name} → {output_name}\n(Dominant: {dominant.upper()})',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'rate_comparison_{input_name}_{output_name}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Error creating rate comparison for {input_name} -> {output_name}: {e}")
        plt.close()
        return (input_name, output_name, False)


def create_rate_comparison_plots(
    derivatives_data: Dict[str, np.ndarray],
    level_rate_df: pd.DataFrame,
    output_dir: str,
    n_workers: int = 8
):
    """
    Create dual scatter plots comparing level vs rate effects.
    
    Args:
        derivatives_data: Data with computed derivatives
        level_rate_df: DataFrame with level/rate analysis results
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating rate comparison plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_vars = ['Q_flow', 'T_Air', 'T_ext']
    output_vars = list(level_rate_df['output'].unique()) if not level_rate_df.empty else []
    
    args_list = []
    
    for input_name in input_vars:
        input_level = derivatives_data['inputs'].get(f'{input_name}_level')
        input_rate = derivatives_data['inputs'].get(input_name)
        
        if input_level is None or input_rate is None:
            continue
        
        for output_name in output_vars:
            output_level = derivatives_data['outputs'].get(f'{output_name}_level')
            
            if output_level is None:
                continue
            
            # Get metrics for this pair
            pair_metrics = level_rate_df[
                (level_rate_df['input'] == input_name) & 
                (level_rate_df['output'] == output_name)
            ]
            
            metrics = pair_metrics.iloc[0].to_dict() if not pair_metrics.empty else {}
            
            args_list.append((
                input_name, output_name, input_level, input_rate,
                output_level, metrics, output_dir
            ))
    
    # Use multiprocessing pool
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_rate_comparison, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} rate comparison plots")


def create_rate_summary_plot(
    level_rate_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary visualization of level vs rate effects across all pairs.
    
    Args:
        level_rate_df: DataFrame with level/rate analysis results
        output_dir: Directory to save plot
    """
    logger.info("Creating rate summary plot...")
    
    if level_rate_df.empty:
        logger.warning("No data for rate summary plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Level vs Rate strength scatter
    ax1 = axes[0, 0]
    colors = {'level': '#3498db', 'rate': '#e74c3c', 'mixed': '#2ecc71'}
    for dominant in ['level', 'rate', 'mixed']:
        subset = level_rate_df[level_rate_df['dominant_effect'] == dominant]
        if not subset.empty:
            ax1.scatter(subset['level_strength'], subset['rate_strength'],
                       alpha=0.6, s=50, c=colors[dominant], label=dominant.capitalize())
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal effect')
    ax1.set_xlabel('Level Effect Strength (|r|)', fontsize=11)
    ax1.set_ylabel('Rate Effect Strength (|r|)', fontsize=11)
    ax1.set_title('Level vs Rate Effect Strength', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dominant effect distribution
    ax2 = axes[0, 1]
    effect_counts = level_rate_df['dominant_effect'].value_counts()
    colors_list = [colors[e] for e in effect_counts.index]
    ax2.pie(effect_counts.values, labels=effect_counts.index, colors=colors_list,
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Dominant Effect Distribution', fontsize=12)
    
    # 3. Effect ratio by input
    ax3 = axes[1, 0]
    for input_name in ['Q_flow', 'T_Air', 'T_ext']:
        input_data = level_rate_df[level_rate_df['input'] == input_name]
        if not input_data.empty:
            effect_ratios = input_data['effect_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
            if not effect_ratios.empty:
                ax3.boxplot(effect_ratios, positions=[['Q_flow', 'T_Air', 'T_ext'].index(input_name)],
                           widths=0.6)
    
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['Q_flow', 'T_Air', 'T_ext'])
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Equal effect')
    ax3.set_xlabel('Input Variable', fontsize=11)
    ax3.set_ylabel('Rate/Level Effect Ratio', fontsize=11)
    ax3.set_title('Effect Ratio by Input', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap of dominant effects
    ax4 = axes[1, 1]
    
    # Create pivot table for dominant effects
    effect_map = {'level': 0, 'rate': 1, 'mixed': 0.5}
    level_rate_df['effect_numeric'] = level_rate_df['dominant_effect'].map(effect_map)
    
    pivot = level_rate_df.pivot_table(
        index='input', columns='output', values='effect_numeric',
        aggfunc='mean'
    )
    
    import seaborn as sns
    cmap = plt.cm.colors.ListedColormap(['#3498db', '#2ecc71', '#e74c3c'])
    sns.heatmap(pivot, ax=ax4, cmap='RdYlBu_r', center=0.5,
                annot=True, fmt='.2f', cbar_kws={'label': 'Level (0) ← → Rate (1)'})
    ax4.set_title('Dominant Effect Heatmap', fontsize=12)
    
    plt.suptitle('Level vs Rate Effects Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rate_effects_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created rate summary plot")
