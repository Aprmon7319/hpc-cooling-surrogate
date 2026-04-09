"""
Power Profile Comparison Visualizer.

Creates visualizations comparing power consumption patterns across systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def create_power_profile_comparison(
    efficiency_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive power profile comparison.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        flow_df: DataFrame with flow metrics
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    systems = efficiency_df['system'].tolist()
    colors = [get_system_color(s) for s in systems]
    
    # 1. Power Breakdown
    ax1 = axes[0, 0]
    metrics = ['mean_cdup_power_kw', 'mean_heat_load_kw']
    available = [m for m in metrics if m in efficiency_df.columns]
    
    if available:
        x = np.arange(len(systems))
        width = 0.35
        
        for i, metric in enumerate(available):
            values = efficiency_df[metric].values
            label = 'CDUP Power' if 'cdup' in metric else 'Heat Load'
            ax1.bar(x + i * width - width/2, values, width, label=label, alpha=0.8)
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('Power Breakdown by System')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in systems])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Flow Rate Comparison
    ax2 = axes[0, 1]
    if 'total_sec_flow_gpm' in flow_df.columns:
        values = flow_df['total_sec_flow_gpm'].values
        bars = ax2.bar(systems, values, color=colors, edgecolor='white')
        ax2.set_title('Total Secondary Flow Rate')
        ax2.set_ylabel('Flow Rate (GPM)')
        ax2.set_xlabel('System')
        ax2.set_xticklabels([s.upper() for s in systems])
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Power Efficiency Ranking
    ax3 = axes[1, 0]
    if 'cooling_power_ratio' in efficiency_df.columns:
        sorted_df = efficiency_df.sort_values('cooling_power_ratio')
        sorted_systems = sorted_df['system'].tolist()
        sorted_values = sorted_df['cooling_power_ratio'].values
        sorted_colors = [get_system_color(s) for s in sorted_systems]
        
        bars = ax3.barh(sorted_systems, sorted_values, color=sorted_colors, edgecolor='white')
        ax3.set_title('Cooling Efficiency Ranking (Lower is Better)')
        ax3.set_xlabel('Cooling Power Ratio')
        ax3.set_yticklabels([s.upper() for s in sorted_systems])
        
        for bar, val in zip(bars, sorted_values):
            if not np.isnan(val):
                ax3.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f' {val:.4f}', ha='left', va='center', fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
    
    # 4. Per-CDU Metrics
    ax4 = axes[1, 1]
    normalized_metrics = ['cdup_power_per_cdu_kw', 'heat_load_per_cdu_kw']
    available_norm = [m for m in normalized_metrics if m in efficiency_df.columns]
    
    if available_norm:
        x = np.arange(len(systems))
        width = 0.35
        
        for i, metric in enumerate(available_norm):
            values = efficiency_df[metric].values
            label = 'CDUP/CDU' if 'cdup' in metric else 'Heat/CDU'
            ax4.bar(x + i * width - width/2, values, width, label=label, alpha=0.8)
        
        ax4.set_xlabel('System')
        ax4.set_ylabel('Power per CDU (kW)')
        ax4.set_title('Normalized Power Metrics (Per CDU)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.upper() for s in systems])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Power Profile Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved power profile comparison to {output_path}")
    
    return fig


def create_power_time_series_comparison(
    system_data: Dict[str, pd.DataFrame],
    system_configs: Dict[str, Dict],
    sample_duration: int = 3600,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Create time series comparison of power consumption.
    
    Args:
        system_data: Dictionary of simulation data per system
        system_configs: Dictionary of system configurations
        sample_duration: Number of time steps to plot
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for system_name, data in system_data.items():
        color = get_system_color(system_name)
        num_cdus = system_configs.get(system_name, {}).get('NUM_CDUS', 1)
        
        # Get CDUP power columns
        power_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.W_flow_CDUP_kW'
            for i in range(1, num_cdus + 1)
        ]
        available_cols = [col for col in power_cols if col in data.columns]
        
        if available_cols:
            total_power = data[available_cols].sum(axis=1)[:sample_duration]
            time = np.arange(len(total_power))
            
            ax.plot(time, total_power, label=system_name.upper(),
                   color=color, alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Total CDUP Power (kW)')
    ax.set_title('CDUP Power Consumption Over Time')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved power time series to {output_path}")
    
    return fig
