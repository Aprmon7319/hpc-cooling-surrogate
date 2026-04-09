"""
Threshold and breakpoint visualizations.
Shows piecewise linear fits and detected breakpoints.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Dict, List

logger = logging.getLogger(__name__)


def _create_single_threshold_plot(args):
    """Create threshold/breakpoint plot for a single pair."""
    input_name, output_name, fit_data, output_dir = args
    
    if not fit_data:
        return (input_name, output_name, False)
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        x_data = fit_data.get('x_data', np.array([]))
        y_data = fit_data.get('y_data', np.array([]))
        x_pred = fit_data.get('x_pred', np.array([]))
        y_linear = fit_data.get('y_linear', np.array([]))
        y_segmented = fit_data.get('y_segmented', np.array([]))
        breakpoints = fit_data.get('breakpoints', [])
        segments = fit_data.get('segments', [])
        linear_r2 = fit_data.get('linear_r2', 0)
        segmented_r2 = fit_data.get('segmented_r2', 0)
        
        if len(x_data) == 0:
            return (input_name, output_name, False)
        
        # Subsample for plotting
        if len(x_data) > 2000:
            idx = np.random.choice(len(x_data), 2000, replace=False)
            x_plot = x_data[idx]
            y_plot = y_data[idx]
        else:
            x_plot = x_data
            y_plot = y_data
        
        # Left plot: Scatter with fits
        ax1 = axes[0]
        ax1.scatter(x_plot, y_plot, alpha=0.3, s=10, color='gray', label='Data')
        ax1.plot(x_pred, y_linear, color='#e74c3c', linewidth=2, 
                linestyle='--', label=f'Linear (R²={linear_r2:.4f})')
        ax1.plot(x_pred, y_segmented, color='#2ecc71', linewidth=2,
                label=f'Segmented (R²={segmented_r2:.4f})')
        
        # Mark breakpoints
        for bp in breakpoints:
            ax1.axvline(x=bp, color='#3498db', linestyle=':', linewidth=2, alpha=0.7)
            ax1.text(bp, ax1.get_ylim()[1], f'BP: {bp:.2f}', 
                    rotation=90, va='top', ha='right', fontsize=9, color='#3498db')
        
        ax1.set_xlabel(input_name, fontsize=12)
        ax1.set_ylabel(output_name, fontsize=12)
        ax1.set_title('Linear vs. Segmented Regression', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Segment details
        ax2 = axes[1]
        
        if segments:
            segment_labels = []
            slopes = []
            colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
            
            for i, seg in enumerate(segments):
                segment_labels.append(f"[{seg['x_start']:.1f}, {seg['x_end']:.1f}]")
                slopes.append(seg['slope'])
            
            bars = ax2.barh(range(len(segments)), slopes, color=colors, edgecolor='black')
            ax2.set_yticks(range(len(segments)))
            ax2.set_yticklabels(segment_labels)
            ax2.set_xlabel('Slope', fontsize=12)
            ax2.set_ylabel('Segment Range', fontsize=12)
            ax2.set_title('Segment Slopes', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linewidth=1)
            
            # Add slope values
            for bar, slope in zip(bars, slopes):
                ax2.text(bar.get_width() + 0.01 * abs(max(slopes) - min(slopes)),
                        bar.get_y() + bar.get_height() / 2,
                        f'{slope:.4f}', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No breakpoints detected',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.axis('off')
        
        plt.suptitle(f'Threshold Detection: {output_name} vs {input_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'threshold_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create threshold plot for {input_name} vs {output_name}: {e}")
        return (input_name, output_name, False)


def create_threshold_plots(
    fit_data_dict: Dict[tuple, Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create threshold/breakpoint plots for all input-output pairs.
    
    Args:
        fit_data_dict: Dictionary mapping (input, output) to segmented fit data
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating threshold plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    args_list = []
    for (input_name, output_name), fit_data in fit_data_dict.items():
        args_list.append((input_name, output_name, fit_data, output_dir))
    
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_threshold_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} threshold plots")


def create_threshold_summary_plot(
    threshold_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary visualization of threshold detection results.
    """
    logger.info("Creating threshold summary plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. R² Improvement Distribution
    ax1 = axes[0, 0]
    r2_improvement = threshold_df['r2_improvement'].values
    ax1.hist(r2_improvement, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Significance threshold (0.05)')
    ax1.set_xlabel('R² Improvement (Segmented - Linear)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('R² Improvement from Segmented Regression', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Number of Breakpoints Distribution
    ax2 = axes[0, 1]
    bp_counts = threshold_df['n_breakpoints'].value_counts().sort_index()
    ax2.bar(bp_counts.index, bp_counts.values, color='coral', edgecolor='black')
    ax2.set_xlabel('Number of Breakpoints', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Detected Breakpoints', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Threshold/Saturation Detection by Input
    ax3 = axes[1, 0]
    
    inputs = threshold_df['input'].unique()
    x = np.arange(len(inputs))
    width = 0.35
    
    threshold_counts = [threshold_df[threshold_df['input'] == inp]['has_threshold'].sum() for inp in inputs]
    saturation_counts = [threshold_df[threshold_df['input'] == inp]['has_saturation'].sum() for inp in inputs]
    
    ax3.bar(x - width/2, threshold_counts, width, label='Has Threshold', color='#e74c3c', edgecolor='black')
    ax3.bar(x + width/2, saturation_counts, width, label='Has Saturation', color='#3498db', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(inputs)
    ax3.set_xlabel('Input Variable', fontsize=11)
    ax3.set_ylabel('Number of Outputs', fontsize=11)
    ax3.set_title('Threshold/Saturation Detection by Input', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Regime Changes Summary
    ax4 = axes[1, 1]
    
    regime_counts = threshold_df['regime_changes'].value_counts()
    labels = ['Single Regime', 'Multiple Regimes']
    sizes = [regime_counts.get(False, 0), regime_counts.get(True, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0, 0.1)
        )
        ax4.set_title('Regime Changes Detection', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
    
    plt.suptitle('Threshold Detection Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
