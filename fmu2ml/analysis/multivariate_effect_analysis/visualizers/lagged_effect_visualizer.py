"""
Lagged effect visualizations.
Creates lag coefficient plots and cumulative effect plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List
import multiprocessing as mp

logger = logging.getLogger(__name__)


def _create_single_lag_plot(args):
    """Create lag coefficient plot for a single input-output pair."""
    input_name, output_name, lag_coeffs, output_dir = args
    
    if not lag_coeffs:
        return (input_name, output_name, False)
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        lags = sorted(lag_coeffs.keys())
        coefficients = [lag_coeffs[lag] for lag in lags]
        
        # 1. Lag coefficient plot
        ax1 = axes[0]
        
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in coefficients]
        ax1.bar(lags, coefficients, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_xlabel('Lag', fontsize=12)
        ax1.set_ylabel('Coefficient', fontsize=12)
        ax1.set_title(f'Lag Coefficients: {input_name} → {output_name}', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative effect plot
        ax2 = axes[1]
        
        cumulative = np.cumsum(coefficients)
        ax2.plot(lags, cumulative, 'o-', color='#9b59b6', linewidth=2, markersize=6)
        ax2.fill_between(lags, 0, cumulative, alpha=0.3, color='#9b59b6')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xlabel('Lag', fontsize=12)
        ax2.set_ylabel('Cumulative Effect', fontsize=12)
        ax2.set_title(f'Cumulative Effect Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add final cumulative value annotation
        ax2.annotate(f'Total: {cumulative[-1]:.3f}',
                    xy=(lags[-1], cumulative[-1]),
                    xytext=(lags[-1] - 2, cumulative[-1] + 0.1 * abs(cumulative[-1])),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        filename = f'lag_coefficients_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create lag plot for {input_name} -> {output_name}: {e}")
        return (input_name, output_name, False)


def create_lag_coefficient_plots(
    lag_data: List[Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create lag coefficient plots for all input-output pairs.
    
    Args:
        lag_data: List of lag analysis results with lag_coefficients
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating lag coefficient plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments
    args_list = []
    for result in lag_data:
        lag_coeffs = result.get('lag_coefficients', {})
        if lag_coeffs:
            args_list.append((
                result['input'],
                result['output'],
                lag_coeffs,
                output_dir
            ))
    
    if not args_list:
        logger.warning("No lag coefficient data available")
        return
    
    # Use multiprocessing
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_lag_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} lag coefficient plots")


def create_lag_heatmap(
    lag_df: pd.DataFrame,
    output_dir: str
):
    """
    Create heatmap showing optimal lag for each input-output pair.
    
    Args:
        lag_df: DataFrame with lag analysis results
        output_dir: Directory to save plots
    """
    logger.info("Creating lag heatmap...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Optimal lag heatmap
    ax1 = axes[0]
    
    pivot_lag = lag_df.pivot(
        index='input',
        columns='output',
        values='optimal_lag'
    )
    
    sns.heatmap(
        pivot_lag,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        ax=ax1,
        cbar_kws={'label': 'Optimal Lag'}
    )
    ax1.set_title('Optimal Lag by Input-Output Pair', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Output Variables', fontsize=10)
    ax1.set_ylabel('Input Variables', fontsize=10)
    
    # 2. R² improvement heatmap
    ax2 = axes[1]
    
    pivot_r2 = lag_df.pivot(
        index='input',
        columns='output',
        values='r2_improvement'
    )
    
    sns.heatmap(
        pivot_r2,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        ax=ax2,
        cbar_kws={'label': 'R² Improvement'}
    )
    ax2.set_title('R² Improvement from Lagged Model', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Output Variables', fontsize=10)
    ax2.set_ylabel('Input Variables', fontsize=10)
    
    plt.suptitle('Lagged Effect Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lag_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created lag heatmap")


def create_memory_length_plot(
    memory_summary: pd.DataFrame,
    output_dir: str
):
    """
    Create bar chart showing memory length by input.
    
    Args:
        memory_summary: DataFrame with memory length statistics
        output_dir: Directory to save plots
    """
    logger.info("Creating memory length plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if memory_summary.empty:
        logger.warning("No memory summary data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Memory length by input
    ax1 = axes[0]
    
    x = np.arange(len(memory_summary))
    width = 0.35
    
    ax1.bar(x - width/2, memory_summary['mean_memory_length'], width,
            label='Mean Memory Length', color='#3498db', edgecolor='black')
    ax1.bar(x + width/2, memory_summary['max_memory_length'], width,
            label='Max Memory Length', color='#e74c3c', edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(memory_summary['input'], fontsize=12)
    ax1.set_xlabel('Input Variables', fontsize=12)
    ax1.set_ylabel('Memory Length (Lags)', fontsize=12)
    ax1.set_title('System Memory Length by Input', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Percentage needing sequence input
    ax2 = axes[1]
    
    colors = ['#2ecc71' if pct < 30 else '#f39c12' if pct < 60 else '#e74c3c'
              for pct in memory_summary['pct_needs_sequence']]
    
    ax2.bar(memory_summary['input'], memory_summary['pct_needs_sequence'],
            color=colors, edgecolor='black')
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50% threshold')
    ax2.set_xlabel('Input Variables', fontsize=12)
    ax2.set_ylabel('% Outputs Needing Sequence Input', fontsize=12)
    ax2.set_title('Recommendation: Use Sequence Inputs?', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (inp, pct) in enumerate(zip(memory_summary['input'], 
                                        memory_summary['pct_needs_sequence'])):
        ax2.text(i, pct + 2, f'{pct:.1f}%', ha='center', fontsize=10)
    
    plt.suptitle('Temporal Memory Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_length.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created memory length plot")


def create_cumulative_effect_summary(
    cumulative_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary plot of cumulative effects across all pairs.
    
    Args:
        cumulative_df: DataFrame with cumulative effects by lag
        output_dir: Directory to save plots
    """
    logger.info("Creating cumulative effect summary...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if cumulative_df.empty:
        logger.warning("No cumulative effect data available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    inputs = cumulative_df['input'].unique()
    colors = {'Q_flow': '#e74c3c', 'T_Air': '#2ecc71', 'T_ext': '#3498db'}
    
    # 1. Mean cumulative effect by lag (grouped by input)
    ax1 = axes[0]
    
    for input_name in inputs:
        input_data = cumulative_df[cumulative_df['input'] == input_name]
        mean_cumulative = input_data.groupby('lag')['cumulative_effect'].mean()
        
        color = colors.get(input_name, '#95a5a6')
        ax1.plot(mean_cumulative.index, mean_cumulative.values, 'o-',
                color=color, linewidth=2, markersize=4, label=input_name)
    
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.set_xlabel('Lag', fontsize=12)
    ax1.set_ylabel('Mean Cumulative Effect', fontsize=12)
    ax1.set_title('Average Cumulative Effect by Input', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Final cumulative effect distribution
    ax2 = axes[1]
    
    max_lags = cumulative_df.groupby(['input', 'output'])['lag'].max()
    final_effects = []
    
    for (inp, out), max_lag in max_lags.items():
        final = cumulative_df[(cumulative_df['input'] == inp) & 
                              (cumulative_df['output'] == out) &
                              (cumulative_df['lag'] == max_lag)]['cumulative_effect'].values
        if len(final) > 0:
            final_effects.append({'input': inp, 'output': out, 'final_effect': final[0]})
    
    final_df = pd.DataFrame(final_effects)
    
    for input_name in inputs:
        input_final = final_df[final_df['input'] == input_name]['final_effect']
        color = colors.get(input_name, '#95a5a6')
        ax2.hist(input_final, bins=15, alpha=0.5, color=color, 
                edgecolor='black', label=input_name)
    
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Total Cumulative Effect', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Total Effects', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Effect decay pattern
    ax3 = axes[2]
    
    for input_name in inputs:
        input_data = cumulative_df[cumulative_df['input'] == input_name]
        mean_coef = input_data.groupby('lag')['coefficient'].mean()
        
        color = colors.get(input_name, '#95a5a6')
        ax3.plot(mean_coef.index, mean_coef.values.cumsum() / (mean_coef.values.cumsum()[-1] + 1e-10),
                'o-', color=color, linewidth=2, markersize=4, label=input_name)
    
    ax3.axhline(y=0.9, color='gray', linestyle='--', label='90% effect')
    ax3.set_xlabel('Lag', fontsize=12)
    ax3.set_ylabel('Fraction of Total Effect', fontsize=12)
    ax3.set_title('Effect Decay: How Quickly Effects Accumulate', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    plt.suptitle('Cumulative Lagged Effect Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_effect_summary.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created cumulative effect summary")


def create_lag_analysis_summary(
    lag_df: pd.DataFrame,
    output_dir: str
):
    """
    Create overall summary of lag analysis results.
    
    Args:
        lag_df: DataFrame with lag analysis results
        output_dir: Directory to save plots
    """
    logger.info("Creating lag analysis summary...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of optimal lags
    ax1 = axes[0, 0]
    ax1.hist(lag_df['optimal_lag'].dropna(), bins=20, color='#3498db', 
             edgecolor='black', alpha=0.7)
    ax1.axvline(x=lag_df['optimal_lag'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {lag_df["optimal_lag"].mean():.1f}')
    ax1.set_xlabel('Optimal Lag', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Optimal Lags', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. R² improvement distribution
    ax2 = axes[0, 1]
    r2_imp = lag_df['r2_improvement'].dropna()
    
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in r2_imp]
    ax2.hist(r2_imp, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.axvline(x=0.05, color='green', linestyle='--', linewidth=2, 
               label='Significant improvement (0.05)')
    ax2.set_xlabel('R² Improvement from Lags', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Benefit of Including Lagged Inputs', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Sequence input recommendation
    ax3 = axes[1, 0]
    
    needs_seq = lag_df['needs_sequence_input'].sum()
    no_seq = len(lag_df) - needs_seq
    
    ax3.pie([needs_seq, no_seq],
            labels=['Needs Sequence\nInput', 'No Sequence\nNeeded'],
            colors=['#e74c3c', '#2ecc71'],
            autopct='%1.1f%%',
            explode=[0.05, 0],
            shadow=True)
    ax3.set_title('Neural Network Architecture Recommendation', 
                  fontsize=12, fontweight='bold')
    
    # 4. Memory length by output type
    ax4 = axes[1, 1]
    
    memory_by_output = lag_df.groupby('output')['memory_length'].mean().sort_values(ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(memory_by_output)))
    ax4.barh(range(len(memory_by_output)), memory_by_output.values, color=colors)
    ax4.set_yticks(range(len(memory_by_output)))
    ax4.set_yticklabels(memory_by_output.index, fontsize=9)
    ax4.set_xlabel('Mean Memory Length (Lags)', fontsize=12)
    ax4.set_title('Memory Length by Output Variable', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Lagged Effect Analysis Summary\nKey Questions: Do past inputs matter? What is the memory length?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lag_analysis_summary.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created lag analysis summary")
