import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def create_isolated_effect_heatmap(
    isolated_df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[List[str]] = None
):
    """
    Create heatmaps showing isolated effects of each input on outputs.
    
    Args:
        isolated_df: DataFrame from analyze_isolated_effects()
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: ['pearson_r', 'r2_score', 'mutual_info'])
    """
    if isolated_df.empty:
        logger.warning("Empty isolated effects DataFrame, skipping heatmap creation")
        return
    
    logger.info("Creating isolated effect heatmaps...")
    
    if metrics is None:
        metrics = ['pearson_r', 'r2_score', 'mutual_info']
    
    for metric in metrics:
        if metric not in isolated_df.columns:
            logger.warning(f"Metric {metric} not found in isolated effects DataFrame")
            continue
        
        pivot_data = isolated_df.pivot(
            index='input',
            columns='output',
            values=metric
        )
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0 if metric in ['pearson_r', 'spearman_r'] else None,
            vmin=-1 if metric in ['pearson_r', 'spearman_r'] else 0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        ax.set_title(f'Isolated Effects - {metric.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('Output Variables', fontsize=12)
        ax.set_ylabel('Input Variables', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'isolated_effect_heatmap_{metric}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    logger.info(f"Created isolated effect heatmaps for {len(metrics)} metrics")


def create_isolated_effect_comparison(
    metrics_df: pd.DataFrame,
    isolated_df: pd.DataFrame,
    output_dir: str
):
    """
    Create comparison plots between full and isolated effects.
    
    Args:
        metrics_df: DataFrame from analyze_all_pairs() (full effects)
        isolated_df: DataFrame from analyze_isolated_effects()
        output_dir: Directory to save plots
    """
    if isolated_df.empty:
        logger.warning("Empty isolated effects DataFrame, skipping comparison")
        return
    
    logger.info("Creating isolated vs full effect comparison plots...")
    
    # Merge full and isolated metrics
    full_df = metrics_df[['input', 'output', 'pearson_r', 'r2_score', 'mutual_info']].copy()
    full_df = full_df.rename(columns={
        'pearson_r': 'pearson_r_full',
        'r2_score': 'r2_score_full',
        'mutual_info': 'mutual_info_full'
    })
    
    iso_df = isolated_df[['input', 'output', 'pearson_r', 'r2_score', 'mutual_info']].copy()
    iso_df = iso_df.rename(columns={
        'pearson_r': 'pearson_r_isolated',
        'r2_score': 'r2_score_isolated',
        'mutual_info': 'mutual_info_isolated'
    })
    
    comparison_df = pd.merge(full_df, iso_df, on=['input', 'output'], how='inner')
    
    if comparison_df.empty:
        logger.warning("No matching pairs for comparison")
        return
    
    # Create comparison bar plots for each input
    input_vars = comparison_df['input'].unique()
    
    for input_var in input_vars:
        input_data = comparison_df[comparison_df['input'] == input_var]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics_pairs = [
            ('pearson_r_full', 'pearson_r_isolated', 'Pearson Correlation'),
            ('r2_score_full', 'r2_score_isolated', 'R² Score'),
            ('mutual_info_full', 'mutual_info_isolated', 'Mutual Information')
        ]
        
        for ax, (full_col, iso_col, title) in zip(axes, metrics_pairs):
            outputs = input_data['output'].values
            full_vals = input_data[full_col].values
            iso_vals = input_data[iso_col].values
            
            x = np.arange(len(outputs))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, full_vals, width, label='Full Effect', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, iso_vals, width, label='Isolated Effect', color='#e74c3c', alpha=0.8)
            
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(outputs, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7)
            
            for bar in bars2:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7)
        
        plt.suptitle(f'Full vs Isolated Effects: {input_var}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'isolated_vs_full_comparison_{input_var}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    logger.info(f"Created comparison plots for {len(input_vars)} inputs")


def create_isolated_effect_summary(
    isolated_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary visualization of isolated effects with sample counts.
    
    Args:
        isolated_df: DataFrame from analyze_isolated_effects()
        output_dir: Directory to save plots
    """
    if isolated_df.empty:
        logger.warning("Empty isolated effects DataFrame, skipping summary")
        return
    
    logger.info("Creating isolated effect summary plots...")
    
    input_vars = isolated_df['input'].unique()
    
    fig, axes = plt.subplots(len(input_vars), 1, figsize=(14, 5 * len(input_vars)))
    
    if len(input_vars) == 1:
        axes = [axes]
    
    colors = {'Q_flow': '#e74c3c', 'T_Air': '#3498db', 'T_ext': '#2ecc71'}
    
    for ax, input_var in zip(axes, input_vars):
        input_data = isolated_df[isolated_df['input'] == input_var].copy()
        input_data = input_data.sort_values('mutual_info', ascending=True)
        
        outputs = input_data['output'].values
        mi_values = input_data['mutual_info'].values
        n_samples = input_data['n_samples'].values if 'n_samples' in input_data.columns else None
        
        y_pos = np.arange(len(outputs))
        
        bars = ax.barh(y_pos, mi_values, color=colors.get(input_var, '#95a5a6'), alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(outputs, fontsize=10)
        ax.set_xlabel('Mutual Information (Isolated Effect)', fontsize=11)
        ax.set_title(f'Isolated Effect of {input_var} on Outputs', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value and sample count labels
        for bar, mi, n in zip(bars, mi_values, n_samples if n_samples is not None else [None]*len(bars)):
            width = bar.get_width()
            label = f'{mi:.3f}'
            if n is not None:
                label += f' (n={int(n)})'
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   label, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'isolated_effect_summary.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    logger.info("Created isolated effect summary plot")