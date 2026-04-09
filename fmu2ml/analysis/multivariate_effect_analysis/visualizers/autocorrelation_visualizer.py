"""
Autocorrelation visualizations.
Creates ACF, PACF, and cross-correlation plots.
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


def _create_single_acf_plot(args):
    """Create ACF/PACF plot for a single variable."""
    var_name, acf_data, output_dir = args
    
    if not acf_data:
        return (var_name, False)
    
    try:
        acf_values = acf_data.get('acf_values', [])
        pacf_values = acf_data.get('pacf_values', [])
        acf_conf = acf_data.get('acf_conf_int', [])
        pacf_conf = acf_data.get('pacf_conf_int', [])
        
        if not acf_values:
            return (var_name, False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        lags = list(range(len(acf_values)))
        
        # ACF plot
        ax1 = axes[0]
        
        ax1.bar(lags, acf_values, color='#3498db', edgecolor='black', 
                linewidth=0.5, alpha=0.7)
        ax1.axhline(y=0, color='black', linewidth=1)
        
        # Confidence interval
        if acf_conf:
            lower = [c[0] for c in acf_conf]
            upper = [c[1] for c in acf_conf]
            ax1.fill_between(lags, lower, upper, alpha=0.2, color='gray',
                           label='95% CI')
        
        ax1.set_xlabel('Lag', fontsize=12)
        ax1.set_ylabel('Autocorrelation', fontsize=12)
        ax1.set_title(f'ACF: {var_name}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # PACF plot
        ax2 = axes[1]
        
        if pacf_values:
            pacf_lags = list(range(len(pacf_values)))
            ax2.bar(pacf_lags, pacf_values, color='#e74c3c', edgecolor='black',
                   linewidth=0.5, alpha=0.7)
            ax2.axhline(y=0, color='black', linewidth=1)
            
            if pacf_conf:
                lower = [c[0] for c in pacf_conf]
                upper = [c[1] for c in pacf_conf]
                ax2.fill_between(pacf_lags, lower, upper, alpha=0.2, color='gray',
                               label='95% CI')
            
            ax2.set_xlabel('Lag', fontsize=12)
            ax2.set_ylabel('Partial Autocorrelation', fontsize=12)
            ax2.set_title(f'PACF: {var_name}', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'PACF not computed', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.set_title(f'PACF: {var_name}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = f'acf_pacf_{var_name.replace("/", "_")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (var_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create ACF plot for {var_name}: {e}")
        return (var_name, False)


def create_acf_plots(
    acf_results: Dict[str, List[Dict]],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create ACF/PACF plots for all variables.
    
    Args:
        acf_results: Dictionary with 'inputs' and 'outputs' ACF results
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating ACF/PACF plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for all variables
    args_list = []
    
    # Input variables
    for result in acf_results.get('inputs', []):
        acf_data = {
            'acf_values': result.get('acf_values', []),
            'pacf_values': result.get('pacf_values', []),
            'acf_conf_int': result.get('acf_conf_int', []),
            'pacf_conf_int': result.get('pacf_conf_int', [])
        }
        args_list.append((f"input_{result['variable']}", acf_data, output_dir))
    
    # Output variables
    for result in acf_results.get('outputs', []):
        acf_data = {
            'acf_values': result.get('acf_values', []),
            'pacf_values': result.get('pacf_values', []),
            'acf_conf_int': result.get('acf_conf_int', []),
            'pacf_conf_int': result.get('pacf_conf_int', [])
        }
        args_list.append((f"output_{result['variable']}", acf_data, output_dir))
    
    if not args_list:
        logger.warning("No ACF data available")
        return
    
    # Use multiprocessing
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_acf_plot, args_list)
    
    successful = sum(1 for r in results if r[1])
    logger.info(f"Created {successful}/{len(results)} ACF/PACF plots")


def _create_single_ccf_plot(args):
    """Create CCF plot for a single input-output pair."""
    input_name, output_name, ccf_data, output_dir = args
    
    if not ccf_data:
        return (input_name, output_name, False)
    
    try:
        ccf_values = ccf_data.get('ccf_values', [])
        lags = ccf_data.get('lags', [])
        peak_lag = ccf_data.get('peak_lag', 0)
        peak_ccf = ccf_data.get('peak_ccf', 0)
        
        if not ccf_values or not lags:
            return (input_name, output_name, False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars by sign
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in ccf_values]
        
        ax.bar(lags, ccf_values, color=colors, edgecolor='black', linewidth=0.3, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        
        # Mark peak
        ax.axvline(x=peak_lag, color='purple', linestyle='-', linewidth=2,
                  label=f'Peak lag = {peak_lag}')
        
        # Significance bands
        n = len(ccf_values)
        threshold = 1.96 / np.sqrt(n)
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Lag (negative = input leads)', fontsize=12)
        ax.set_ylabel('Cross-Correlation', fontsize=12)
        ax.set_title(f'Cross-Correlation: {input_name} → {output_name}\n'
                    f'(Peak CCF = {peak_ccf:.3f} at lag {peak_lag})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'ccf_{input_name}_{output_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return (input_name, output_name, True)
        
    except Exception as e:
        logger.error(f"Failed to create CCF plot for {input_name} -> {output_name}: {e}")
        return (input_name, output_name, False)


def create_ccf_plots(
    ccf_results: List[Dict],
    output_dir: str,
    n_workers: int = 8
):
    """
    Create CCF plots for all input-output pairs.
    
    Args:
        ccf_results: List of CCF analysis results
        output_dir: Directory to save plots
        n_workers: Number of parallel workers
    """
    logger.info("Creating CCF plots in parallel...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments
    args_list = []
    for result in ccf_results:
        ccf_data = {
            'ccf_values': result.get('ccf_values', []),
            'lags': result.get('lags', []),
            'peak_lag': result.get('peak_lag', 0),
            'peak_ccf': result.get('peak_ccf', 0)
        }
        args_list.append((result['input'], result['output'], ccf_data, output_dir))
    
    if not args_list:
        logger.warning("No CCF data available")
        return
    
    # Use multiprocessing
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_ccf_plot, args_list)
    
    successful = sum(1 for r in results if r[2])
    logger.info(f"Created {successful}/{len(results)} CCF plots")


def create_persistence_summary(
    input_acf_df: pd.DataFrame,
    output_acf_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary plot of temporal persistence.
    
    Args:
        input_acf_df: DataFrame with input ACF summary
        output_acf_df: DataFrame with output ACF summary
        output_dir: Directory to save plots
    """
    logger.info("Creating persistence summary...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Input persistence
    ax1 = axes[0, 0]
    
    if not input_acf_df.empty:
        ax1.bar(input_acf_df['variable'], input_acf_df['persistence'],
               color='#3498db', edgecolor='black')
        ax1.set_xlabel('Input Variables', fontsize=12)
        ax1.set_ylabel('Persistence Score', fontsize=12)
        ax1.set_title('Input Temporal Persistence', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Output persistence
    ax2 = axes[0, 1]
    
    if not output_acf_df.empty:
        sorted_df = output_acf_df.sort_values('persistence', ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
        
        ax2.barh(range(len(sorted_df)), sorted_df['persistence'], color=colors)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels(sorted_df['variable'], fontsize=9)
        ax2.set_xlabel('Persistence Score', fontsize=12)
        ax2.set_title('Output Temporal Persistence', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Suggested AR order distribution
    ax3 = axes[1, 0]
    
    if not output_acf_df.empty:
        ar_orders = output_acf_df['ar_order_suggestion'].value_counts().sort_index()
        ax3.bar(ar_orders.index, ar_orders.values, color='#9b59b6', edgecolor='black')
        ax3.set_xlabel('Suggested AR Order', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Distribution of Suggested AR Orders\n(Based on PACF)',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. First insignificant lag distribution
    ax4 = axes[1, 1]
    
    if not output_acf_df.empty:
        first_insig = output_acf_df['first_insignificant_lag'].dropna()
        ax4.hist(first_insig, bins=20, color='#f39c12', edgecolor='black', alpha=0.7)
        ax4.axvline(x=first_insig.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {first_insig.mean():.1f}')
        ax4.set_xlabel('First Insignificant ACF Lag', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_title('How Quickly Autocorrelation Decays',
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Persistence Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'persistence_summary.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created persistence summary")


def create_ccf_heatmap(
    ccf_df: pd.DataFrame,
    output_dir: str
):
    """
    Create heatmap of peak CCF values and lags.
    
    Args:
        ccf_df: DataFrame with CCF analysis results
        output_dir: Directory to save plots
    """
    logger.info("Creating CCF heatmap...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Peak CCF heatmap
    ax1 = axes[0]
    
    pivot_ccf = ccf_df.pivot(
        index='input',
        columns='output',
        values='peak_ccf'
    )
    
    sns.heatmap(
        pivot_ccf,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        ax=ax1,
        cbar_kws={'label': 'Peak Cross-Correlation'}
    )
    ax1.set_title('Peak Cross-Correlation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Output Variables', fontsize=10)
    ax1.set_ylabel('Input Variables', fontsize=10)
    
    # 2. Peak lag heatmap
    ax2 = axes[1]
    
    pivot_lag = ccf_df.pivot(
        index='input',
        columns='output',
        values='peak_lag'
    )
    
    # Custom colormap for lag (centered at 0)
    max_abs_lag = max(abs(pivot_lag.values.min()), abs(pivot_lag.values.max()))
    
    sns.heatmap(
        pivot_lag,
        annot=True,
        fmt='.0f',
        cmap='PuOr',
        center=0,
        vmin=-max_abs_lag, vmax=max_abs_lag,
        ax=ax2,
        cbar_kws={'label': 'Peak Lag (- = input leads)'}
    )
    ax2.set_title('Lag at Peak Cross-Correlation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Output Variables', fontsize=10)
    ax2.set_ylabel('Input Variables', fontsize=10)
    
    plt.suptitle('Cross-Correlation Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ccf_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created CCF heatmap")


def create_temporal_recommendations_plot(
    recommendations_df: pd.DataFrame,
    output_dir: str
):
    """
    Create visualization of temporal modeling recommendations.
    
    Args:
        recommendations_df: DataFrame with temporal recommendations
        output_dir: Directory to save plots
    """
    logger.info("Creating temporal recommendations plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if recommendations_df.empty:
        logger.warning("No recommendations data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Needs temporal model pie chart
    ax1 = axes[0]
    
    needs_temporal = recommendations_df['needs_temporal_model'].sum()
    no_temporal = len(recommendations_df) - needs_temporal
    
    ax1.pie([needs_temporal, no_temporal],
            labels=['Needs Temporal\nModel', 'Feedforward\nSufficient'],
            colors=['#e74c3c', '#2ecc71'],
            autopct='%1.1f%%',
            explode=[0.05, 0],
            shadow=True)
    ax1.set_title('Temporal Modeling Requirements', fontsize=12, fontweight='bold')
    
    # 2. Persistence vs AR order
    ax2 = axes[1]
    
    colors = ['#e74c3c' if needs else '#2ecc71' 
              for needs in recommendations_df['needs_temporal_model']]
    
    ax2.scatter(recommendations_df['persistence_score'],
               recommendations_df['suggested_ar_order'],
               c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add output labels
    for i, row in recommendations_df.iterrows():
        ax2.annotate(row['output'][:10], 
                    (row['persistence_score'], row['suggested_ar_order']),
                    fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Persistence Score', fontsize=12)
    ax2.set_ylabel('Suggested AR Order', fontsize=12)
    ax2.set_title('Outputs by Temporal Complexity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Needs Temporal Model'),
        mpatches.Patch(facecolor='#2ecc71', label='Feedforward Sufficient')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle('Neural Network Architecture Recommendations\n'
                'Based on Autocorrelation Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_recommendations.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created temporal recommendations plot")
