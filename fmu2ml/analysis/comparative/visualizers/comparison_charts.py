"""
Comparison Charts for Cooling Model Analysis.

Creates bar charts, grouped comparisons, and radar charts for
multi-system comparison visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

# Define consistent color palette for systems
SYSTEM_COLORS = {
    'marconi100': '#1f77b4',  # Blue
    'summit': '#ff7f0e',       # Orange
    'frontier': '#2ca02c',     # Green
    'lassen': '#d62728',       # Red
    'fugaku': '#9467bd',       # Purple
    'setonix': '#8c564b',      # Brown
    'adastraMI250': '#e377c2', # Pink
}

def get_system_color(system_name: str) -> str:
    """Get consistent color for a system."""
    return SYSTEM_COLORS.get(system_name, '#7f7f7f')


def create_system_comparison_chart(
    comparison_df: pd.DataFrame,
    metric_columns: List[str],
    metric_labels: Optional[Dict[str, str]] = None,
    title: str = "System Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create a grouped bar chart comparing systems across multiple metrics.
    
    Args:
        comparison_df: DataFrame with 'system' column and metric columns
        metric_columns: List of column names to compare
        metric_labels: Optional mapping of column names to display labels
        title: Chart title
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metric_labels is None:
        metric_labels = {col: col.replace('_', ' ').title() for col in metric_columns}
    
    # Filter to valid metrics
    valid_metrics = [col for col in metric_columns if col in comparison_df.columns]
    if not valid_metrics:
        logger.warning("No valid metrics found for comparison chart")
        return None
    
    systems = comparison_df['system'].tolist()
    n_systems = len(systems)
    n_metrics = len(valid_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_systems
    
    # Create bars for each system
    for i, system in enumerate(systems):
        values = comparison_df[comparison_df['system'] == system][valid_metrics].values[0]
        offset = (i - n_systems / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, 
            values, 
            width, 
            label=system.upper(),
            color=get_system_color(system),
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                height = bar.get_height()
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8
                )
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in valid_metrics], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison chart to {output_path}")
    
    return fig


def create_efficiency_comparison_plot(
    efficiency_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create a comprehensive efficiency comparison visualization.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics per system
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    systems = efficiency_df['system'].tolist()
    colors = [get_system_color(s) for s in systems]
    
    # 1. Total CDUP Power
    ax1 = axes[0, 0]
    if 'mean_cdup_power_kw' in efficiency_df.columns:
        values = efficiency_df['mean_cdup_power_kw'].values
        bars = ax1.bar(systems, values, color=colors, edgecolor='white')
        ax1.set_title('Total CDUP Power Consumption')
        ax1.set_ylabel('Power (kW)')
        ax1.set_xlabel('System')
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Power per CDU (normalized)
    ax2 = axes[0, 1]
    if 'cdup_power_per_cdu_kw' in efficiency_df.columns:
        values = efficiency_df['cdup_power_per_cdu_kw'].values
        bars = ax2.bar(systems, values, color=colors, edgecolor='white')
        ax2.set_title('CDUP Power per CDU (Normalized)')
        ax2.set_ylabel('Power per CDU (kW)')
        ax2.set_xlabel('System')
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Cooling Power Ratio
    ax3 = axes[1, 0]
    if 'cooling_power_ratio' in efficiency_df.columns:
        values = efficiency_df['cooling_power_ratio'].values
        bars = ax3.bar(systems, values, color=colors, edgecolor='white')
        ax3.set_title('Cooling Power Ratio (Lower is Better)')
        ax3.set_ylabel('Ratio (CDUP/Heat Load)')
        ax3.set_xlabel('System')
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. System Scale Overview
    ax4 = axes[1, 1]
    if 'num_cdus' in efficiency_df.columns:
        values = efficiency_df['num_cdus'].values
        bars = ax4.bar(systems, values, color=colors, edgecolor='white')
        ax4.set_title('System Scale (Number of CDUs)')
        ax4.set_ylabel('Number of CDUs')
        ax4.set_xlabel('System')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(val)}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Cooling Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved efficiency comparison to {output_path}")
    
    return fig


def create_radar_comparison_chart(
    normalized_df: pd.DataFrame,
    metrics: List[str],
    metric_labels: Optional[Dict[str, str]] = None,
    title: str = "System Characteristics Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 10)
) -> plt.Figure:
    """
    Create a radar/spider chart for multi-dimensional comparison.
    
    Args:
        normalized_df: DataFrame with normalized metrics (0-1 scale preferred)
        metrics: List of metric columns to include
        metric_labels: Optional mapping of column names to display labels
        title: Chart title
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metric_labels is None:
        metric_labels = {col: col.replace('_', ' ').title() for col in metrics}
    
    # Filter to valid metrics
    valid_metrics = [col for col in metrics if col in normalized_df.columns]
    if len(valid_metrics) < 3:
        logger.warning("Need at least 3 metrics for radar chart")
        return None
    
    systems = normalized_df['system'].tolist()
    
    # Normalize metrics to 0-1 scale for radar chart
    data = normalized_df[valid_metrics].copy()
    for col in valid_metrics:
        col_min = data[col].min()
        col_max = data[col].max()
        if col_max > col_min:
            data[col] = (data[col] - col_min) / (col_max - col_min)
        else:
            data[col] = 0.5
    
    # Number of variables
    num_vars = len(valid_metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [metric_labels.get(m, m) for m in valid_metrics], size=10)
    
    # Draw the chart for each system
    for idx, system in enumerate(systems):
        values = data.iloc[idx].tolist()
        values += values[:1]  # Complete the loop
        
        color = get_system_color(system)
        ax.plot(angles, values, 'o-', linewidth=2, label=system.upper(), color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_title(title, size=14, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved radar chart to {output_path}")
    
    return fig
