"""
Impulse Response Visualizer

Creates visualizations for impulse response analysis:
- Normalized response curves
- Box plots of response characteristics
- Response heatmaps
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _create_single_impulse_curve(args):
    """Create impulse response curve for a single event. Used for parallel execution."""
    curve_data, output_dir = args
    
    try:
        input_name = curve_data['input']
        output_name = curve_data['output']
        step_idx = curve_data['step_index']
        time = curve_data['time']
        values = curve_data['output_values']
        step_position = curve_data['step_position']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left: Raw response
        ax1 = axes[0]
        ax1.plot(time, values, 'b-', linewidth=1.5)
        ax1.axvline(x=time[step_position], color='r', linestyle='--', 
                    label='Step occurrence', alpha=0.7)
        
        ax1.set_xlabel('Time', fontsize=10)
        ax1.set_ylabel(output_name, fontsize=10)
        ax1.set_title(f'Raw Response: {output_name}', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Normalized response
        ax2 = axes[1]
        
        pre_values = values[:step_position]
        post_values = values[step_position:]
        
        if len(pre_values) > 0 and len(post_values) > 5:
            initial = np.nanmean(pre_values[-min(5, len(pre_values)):])
            final = np.nanmean(post_values[-min(10, len(post_values)):])
            
            if abs(final - initial) > 1e-10:
                normalized = (values - initial) / (final - initial)
                ax2.plot(range(len(normalized)), normalized, 'g-', linewidth=1.5)
                ax2.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='Final value')
                ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Initial value')
                ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
                ax2.axvline(x=step_position, color='r', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Time Steps', fontsize=10)
        ax2.set_ylabel('Normalized Response', fontsize=10)
        ax2.set_title('Normalized Response', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.2, 1.3)
        
        plt.suptitle(
            f'Impulse Response: {input_name} → {output_name} (step at t={step_idx})',
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'impulse_{input_name}_{output_name}_{step_idx}.png'),
            dpi=200, bbox_inches='tight'
        )
        plt.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating impulse curve: {e}")
        plt.close()
        return False


def create_impulse_response_plots(
    response_curves: List[Dict],
    response_df: pd.DataFrame,
    output_dir: str,
    n_workers: int = 8,
    max_plots: int = 50
):
    """
    Create impulse response visualizations.
    
    Args:
        response_curves: List of response curve data
        response_df: DataFrame with response analysis results
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
        max_plots: Maximum number of individual plots to create
    """
    logger.info("Creating impulse response plots...")
    
    impulse_dir = os.path.join(output_dir, 'impulse_responses')
    os.makedirs(impulse_dir, exist_ok=True)
    
    # Create individual response curves (limited number)
    if response_curves:
        curves_to_plot = response_curves[:max_plots]
        args_list = [(curve, impulse_dir) for curve in curves_to_plot]
        
        with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
            results = pool.map(_create_single_impulse_curve, args_list)
        
        successful = sum(1 for r in results if r)
        logger.info(f"Created {successful}/{len(curves_to_plot)} individual impulse plots")
    
    # Create summary plots
    if not response_df.empty:
        _create_response_summary_plots(response_df, output_dir)


def _create_response_summary_plots(response_df: pd.DataFrame, output_dir: str):
    """Create summary plots for impulse response analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Rise time distribution by input
    ax1 = axes[0, 0]
    inputs = response_df['input'].unique()
    rise_times_by_input = [
        response_df[response_df['input'] == inp]['rise_time'].dropna()
        for inp in inputs
    ]
    bp1 = ax1.boxplot(rise_times_by_input, labels=inputs, patch_artist=True)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel('Input Variable', fontsize=11)
    ax1.set_ylabel('Rise Time (steps)', fontsize=11)
    ax1.set_title('Rise Time Distribution by Input', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Gain distribution by input-output pair
    ax2 = axes[0, 1]
    summary = response_df.groupby(['input', 'output'])['gain'].agg(['mean', 'std']).reset_index()
    summary = summary.dropna()
    
    if not summary.empty:
        x = range(len(summary))
        ax2.bar(x, summary['mean'], yerr=summary['std'], capsize=3, alpha=0.7)
        labels = [f"{row['input'][:3]}→{row['output'][:6]}" for _, row in summary.iterrows()]
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_xlabel('Input → Output', fontsize=11)
        ax2.set_ylabel('Gain (Δoutput/Δinput)', fontsize=11)
        ax2.set_title('Mean Gain by Input-Output Pair', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Time constant distribution
    ax3 = axes[1, 0]
    time_constants = response_df['time_constant'].dropna()
    time_constants = time_constants[time_constants < np.percentile(time_constants, 95)]  # Remove outliers
    if len(time_constants) > 0:
        ax3.hist(time_constants, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax3.axvline(x=time_constants.median(), color='r', linestyle='--', 
                    label=f'Median: {time_constants.median():.1f}')
        ax3.set_xlabel('Time Constant (τ)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Time Constant Distribution', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Response order distribution
    ax4 = axes[1, 1]
    order_counts = response_df['response_order'].value_counts()
    colors_pie = ['#27ae60', '#f1c40f', '#e74c3c', '#95a5a6']
    ax4.pie(order_counts.values, labels=order_counts.index, colors=colors_pie[:len(order_counts)],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Response Order Classification', fontsize=12)
    
    plt.suptitle('Impulse Response Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'impulse_response_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created impulse response summary plots")


def create_averaged_impulse_plots(
    response_curves: List[Dict],
    response_df: pd.DataFrame,
    output_dir: str
):
    """
    Create averaged/normalized impulse response plots for each input-output pair.
    
    Args:
        response_curves: List of response curve data
        response_df: DataFrame with response analysis results
        output_dir: Directory to save plots
    """
    logger.info("Creating averaged impulse response plots...")
    
    if not response_curves:
        return
    
    # Group curves by input-output pair
    pairs = {}
    for curve in response_curves:
        key = (curve['input'], curve['output'])
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(curve)
    
    n_pairs = len(pairs)
    if n_pairs == 0:
        return
    
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, ((input_name, output_name), curves) in enumerate(pairs.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Normalize and align all curves
        normalized_curves = []
        max_len = 0
        
        for curve in curves:
            values = curve['output_values']
            step_pos = curve['step_position']
            
            pre_values = values[:step_pos]
            post_values = values[step_pos:]
            
            if len(pre_values) > 0 and len(post_values) > 5:
                initial = np.nanmean(pre_values[-min(5, len(pre_values)):])
                final = np.nanmean(post_values[-min(10, len(post_values)):])
                
                if abs(final - initial) > 1e-10:
                    # Normalize post-step portion
                    normalized = (post_values - initial) / (final - initial)
                    normalized_curves.append(normalized)
                    max_len = max(max_len, len(normalized))
        
        if not normalized_curves:
            ax.text(0.5, 0.5, 'No valid responses', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{input_name} → {output_name}', fontsize=10)
            continue
        
        # Pad and stack
        padded = []
        for curve in normalized_curves:
            if len(curve) < max_len:
                padded.append(np.pad(curve, (0, max_len - len(curve)), 
                                    mode='constant', constant_values=np.nan))
            else:
                padded.append(curve[:max_len])
        
        stacked = np.array(padded)
        
        # Plot individual curves (light)
        for curve in stacked:
            ax.plot(curve, 'b-', alpha=0.1, linewidth=0.5)
        
        # Plot mean curve
        mean_curve = np.nanmean(stacked, axis=0)
        std_curve = np.nanstd(stacked, axis=0)
        
        x = range(len(mean_curve))
        ax.plot(x, mean_curve, 'b-', linewidth=2, label='Mean response')
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                       alpha=0.3, color='blue')
        
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        
        ax.set_xlabel('Time after step', fontsize=9)
        ax.set_ylabel('Normalized response', fontsize=9)
        ax.set_title(f'{input_name} → {output_name} (n={len(normalized_curves)})', fontsize=10)
        ax.set_ylim(-0.2, 1.4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused axes
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Averaged Impulse Responses by Input-Output Pair', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'averaged_impulse_responses.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created averaged impulse response plots")
