"""
Level vs Rate Correlation Bar Visualizer

Creates bar charts comparing:
- Level effect correlations
- Rate effect correlations
- Side-by-side comparison
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_level_rate_correlation_bars(
    level_rate_df: pd.DataFrame,
    output_dir: str
):
    """
    Create bar charts comparing level vs rate effect correlations.
    
    Args:
        level_rate_df: DataFrame with level/rate analysis results
        output_dir: Directory to save plots
    """
    logger.info("Creating level vs rate correlation bar charts...")
    
    if level_rate_df.empty:
        logger.warning("No level-rate data for bar charts")
        return
    
    # Create plots for each input variable
    for input_name in level_rate_df['input'].unique():
        _create_single_input_comparison(input_name, level_rate_df, output_dir)
    
    # Create summary comparison
    _create_overall_comparison(level_rate_df, output_dir)
    
    logger.info("Created level vs rate correlation bar charts")


def _create_single_input_comparison(
    input_name: str,
    level_rate_df: pd.DataFrame,
    output_dir: str
):
    """Create comparison bar chart for a single input."""
    
    input_data = level_rate_df[level_rate_df['input'] == input_name].copy()
    
    if input_data.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by level correlation for consistent ordering
    input_data = input_data.sort_values('level_pearson_r', ascending=True)
    
    outputs = input_data['output'].tolist()
    x = np.arange(len(outputs))
    width = 0.35
    
    # Left plot: Correlation comparison
    ax1 = axes[0]
    
    level_corr = input_data['level_pearson_r'].values
    rate_corr = input_data['rate_level_pearson_r'].values
    
    bars1 = ax1.barh(x - width/2, np.abs(level_corr), width, 
                     label='Level Effect |r|', color='#3498db', alpha=0.8)
    bars2 = ax1.barh(x + width/2, np.abs(rate_corr), width, 
                     label='Rate Effect |r|', color='#e74c3c', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(outputs, fontsize=9)
    ax1.set_xlabel('Absolute Correlation |r|', fontsize=11)
    ax1.set_title(f'{input_name}: Level vs Rate Effect Strength', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars1, level_corr):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8, color='#3498db')
    for bar, val in zip(bars2, rate_corr):
        if not np.isnan(val):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8, color='#e74c3c')
    
    # Right plot: R² comparison
    ax2 = axes[1]
    
    level_r2 = input_data['level_r2'].values
    rate_r2 = input_data['rate_level_r2'].values
    
    bars3 = ax2.barh(x - width/2, level_r2, width, 
                     label='Level R²', color='#2ecc71', alpha=0.8)
    bars4 = ax2.barh(x + width/2, rate_r2, width, 
                     label='Rate R²', color='#f39c12', alpha=0.8)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(outputs, fontsize=9)
    ax2.set_xlabel('R² Score', fontsize=11)
    ax2.set_title(f'{input_name}: Level vs Rate Explained Variance', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1.1)
    
    plt.suptitle(
        f'Level vs Rate Effects: {input_name}',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'level_rate_bars_{input_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def _create_overall_comparison(
    level_rate_df: pd.DataFrame,
    output_dir: str
):
    """Create overall summary comparison across all inputs and outputs."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Average effect strength by input
    ax1 = axes[0, 0]
    
    avg_by_input = level_rate_df.groupby('input').agg({
        'level_strength': 'mean',
        'rate_strength': 'mean'
    }).reset_index()
    
    x = np.arange(len(avg_by_input))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_by_input['level_strength'], width,
                    label='Level Effect', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_by_input['rate_strength'], width,
                    label='Rate Effect', color='#e74c3c', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(avg_by_input['input'])
    ax1.set_ylabel('Average Effect Strength |r|', fontsize=11)
    ax1.set_title('Average Effect Strength by Input', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Dominant effect count by input
    ax2 = axes[0, 1]
    
    dominant_counts = level_rate_df.groupby(['input', 'dominant_effect']).size().unstack(fill_value=0)
    
    colors = {'level': '#3498db', 'rate': '#e74c3c', 'mixed': '#2ecc71'}
    bottom = np.zeros(len(dominant_counts))
    
    for effect in ['level', 'rate', 'mixed']:
        if effect in dominant_counts.columns:
            ax2.bar(dominant_counts.index, dominant_counts[effect], 
                   bottom=bottom, label=effect.capitalize(),
                   color=colors[effect], alpha=0.8)
            bottom += dominant_counts[effect].values
    
    ax2.set_ylabel('Number of Output Variables', fontsize=11)
    ax2.set_title('Dominant Effect Distribution by Input', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Effect ratio distribution
    ax3 = axes[1, 0]
    
    effect_ratios = level_rate_df['effect_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    effect_ratios = effect_ratios[effect_ratios < 5]  # Clip outliers
    
    if len(effect_ratios) > 0:
        ax3.hist(effect_ratios, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax3.axvline(x=1, color='k', linestyle='--', label='Equal effect', alpha=0.7)
        ax3.axvline(x=effect_ratios.median(), color='r', linestyle='-',
                    label=f'Median: {effect_ratios.median():.2f}', alpha=0.7)
        ax3.set_xlabel('Rate/Level Effect Ratio', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution of Rate/Level Effect Ratio', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_stats = []
    for input_name in level_rate_df['input'].unique():
        input_data = level_rate_df[level_rate_df['input'] == input_name]
        
        level_dominant = (input_data['dominant_effect'] == 'level').sum()
        rate_dominant = (input_data['dominant_effect'] == 'rate').sum()
        mixed = (input_data['dominant_effect'] == 'mixed').sum()
        
        avg_level_r = input_data['level_strength'].mean()
        avg_rate_r = input_data['rate_strength'].mean()
        
        summary_stats.append({
            'Input': input_name,
            'Level Dominant': level_dominant,
            'Rate Dominant': rate_dominant,
            'Mixed': mixed,
            'Avg Level |r|': f'{avg_level_r:.3f}',
            'Avg Rate |r|': f'{avg_rate_r:.3f}'
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    table = ax4.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db', '#ecf0f1', '#ecf0f1', '#ecf0f1', '#e8f6f3', '#fadbd8']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Summary Statistics', fontsize=12, pad=20)
    
    plt.suptitle(
        'Level vs Rate Effects: Overall Summary',
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'level_rate_overall_summary.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
