"""
Scaling Analysis Visualizer.

Creates visualizations for analyzing how cooling characteristics
scale with system size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

# Define consistent color palette for systems
SYSTEM_COLORS = {
    'marconi100': '#1f77b4',
    'summit': '#ff7f0e',
    'frontier': '#2ca02c',
    'lassen': '#d62728',
    'fugaku': '#9467bd',
    'setonix': '#8c564b',
    'adastraMI250': '#e377c2',
}

def get_system_color(system_name: str) -> str:
    """Get consistent color for a system."""
    return SYSTEM_COLORS.get(system_name, '#7f7f7f')


def create_scaling_analysis_plot(
    comparison_df: pd.DataFrame,
    x_metric: str = 'num_cdus',
    y_metrics: List[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create scaling analysis plots showing how metrics change with system size.
    
    Args:
        comparison_df: DataFrame with system metrics
        x_metric: Metric to use for x-axis (typically 'num_cdus')
        y_metrics: List of metrics to plot against x_metric
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if y_metrics is None:
        y_metrics = [
            'mean_cdup_power_kw',
            'cdup_power_per_cdu_kw',
            'total_sec_flow_gpm',
            'sec_flow_per_cdu_gpm'
        ]
    
    # Filter to available metrics
    available_metrics = [m for m in y_metrics if m in comparison_df.columns]
    if not available_metrics:
        logger.warning("No valid metrics for scaling analysis")
        return None
    
    n_plots = len(available_metrics)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    systems = comparison_df['system'].tolist()
    x_values = comparison_df[x_metric].values
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        y_values = comparison_df[metric].values
        
        # Scatter plot with system labels
        for i, system in enumerate(systems):
            color = get_system_color(system)
            ax.scatter(x_values[i], y_values[i], s=150, c=color, 
                      label=system.upper(), edgecolors='white', linewidth=2)
            ax.annotate(system.upper(), (x_values[i], y_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Fit trend line if enough points
        if len(x_values) >= 3:
            valid_mask = ~np.isnan(y_values)
            if valid_mask.sum() >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_values[valid_mask], y_values[valid_mask]
                )
                x_line = np.linspace(x_values.min(), x_values.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'k--', alpha=0.5, 
                       label=f'Trend (R²={r_value**2:.3f})')
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Scaling Analysis: Metrics vs System Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scaling analysis to {output_path}")
    
    return fig


def create_efficiency_scaling_plot(
    comparison_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Create efficiency scaling analysis focusing on power ratios.
    
    Args:
        comparison_df: DataFrame with efficiency metrics
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    systems = comparison_df['system'].tolist()
    colors = [get_system_color(s) for s in systems]
    
    if 'num_cdus' not in comparison_df.columns:
        logger.warning("num_cdus column required for efficiency scaling plot")
        return None
    
    x = comparison_df['num_cdus'].values
    
    # 1. Total Power vs CDUs
    ax1 = axes[0, 0]
    if 'mean_cdup_power_kw' in comparison_df.columns:
        y = comparison_df['mean_cdup_power_kw'].values
        for i, system in enumerate(systems):
            ax1.scatter(x[i], y[i], s=150, c=colors[i], 
                       label=system.upper(), edgecolors='white')
        
        # Fit linear trend
        valid = ~np.isnan(y)
        if valid.sum() >= 2:
            slope, intercept, r, _, _ = stats.linregress(x[valid], y[valid])
            x_line = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
            ax1.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
            ax1.set_title(f'Total CDUP Power (slope={slope:.2f} kW/CDU)')
        else:
            ax1.set_title('Total CDUP Power')
        
        ax1.set_xlabel('Number of CDUs')
        ax1.set_ylabel('Power (kW)')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
    
    # 2. Power per CDU vs CDUs (shows scaling efficiency)
    ax2 = axes[0, 1]
    if 'cdup_power_per_cdu_kw' in comparison_df.columns:
        y = comparison_df['cdup_power_per_cdu_kw'].values
        for i, system in enumerate(systems):
            ax2.scatter(x[i], y[i], s=150, c=colors[i], 
                       label=system.upper(), edgecolors='white')
        
        ax2.axhline(y=np.nanmean(y), color='red', linestyle='--', alpha=0.5, 
                   label=f'Mean: {np.nanmean(y):.2f}')
        ax2.set_xlabel('Number of CDUs')
        ax2.set_ylabel('Power per CDU (kW)')
        ax2.set_title('Power per CDU (Scaling Efficiency)')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
    
    # 3. Cooling Ratio vs CDUs
    ax3 = axes[1, 0]
    if 'cooling_power_ratio' in comparison_df.columns:
        y = comparison_df['cooling_power_ratio'].values
        for i, system in enumerate(systems):
            ax3.scatter(x[i], y[i], s=150, c=colors[i], 
                       label=system.upper(), edgecolors='white')
        
        ax3.set_xlabel('Number of CDUs')
        ax3.set_ylabel('Cooling Power Ratio')
        ax3.set_title('Cooling Efficiency Ratio vs Scale')
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)
    
    # 4. Heat Load per CDU
    ax4 = axes[1, 1]
    if 'heat_load_per_cdu_kw' in comparison_df.columns:
        y = comparison_df['heat_load_per_cdu_kw'].values
        for i, system in enumerate(systems):
            ax4.scatter(x[i], y[i], s=150, c=colors[i], 
                       label=system.upper(), edgecolors='white')
        
        ax4.set_xlabel('Number of CDUs')
        ax4.set_ylabel('Heat Load per CDU (kW)')
        ax4.set_title('Heat Load per CDU')
        ax4.legend(loc='best')
        ax4.grid(alpha=0.3)
    
    plt.suptitle('Efficiency Scaling Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved efficiency scaling plot to {output_path}")
    
    return fig
