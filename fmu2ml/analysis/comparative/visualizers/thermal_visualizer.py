"""
Thermal Response Comparison Visualizer.

Creates visualizations comparing thermal behavior across systems.
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


def create_thermal_response_comparison(
    thermal_df: pd.DataFrame,
    system_data: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive thermal response comparison visualization.
    
    Args:
        thermal_df: DataFrame with thermal metrics per system
        system_data: Optional dict of raw simulation data per system
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    systems = thermal_df['system'].tolist()
    colors = [get_system_color(s) for s in systems]
    
    # 1. Temperature Comparison
    ax1 = axes[0, 0]
    metrics = ['mean_rack_return_temp_c', 'mean_rack_supply_temp_c']
    available = [m for m in metrics if m in thermal_df.columns]
    
    if available:
        x = np.arange(len(systems))
        width = 0.35
        
        for i, metric in enumerate(available):
            values = thermal_df[metric].values
            label = 'Return Temp' if 'return' in metric else 'Supply Temp'
            ax1.bar(x + i * width - width/2, values, width, 
                   label=label, alpha=0.8)
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Rack Temperature Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in systems])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Temperature Delta
    ax2 = axes[0, 1]
    if 'mean_delta_t_c' in thermal_df.columns:
        values = thermal_df['mean_delta_t_c'].values
        bars = ax2.bar(systems, values, color=colors, edgecolor='white')
        ax2.set_title('Mean Temperature Delta (Return - Supply)')
        ax2.set_ylabel('ΔT (°C)')
        ax2.set_xlabel('System')
        ax2.set_xticklabels([s.upper() for s in systems])
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Temperature Variability
    ax3 = axes[1, 0]
    if 'rack_return_temp_std' in thermal_df.columns:
        values = thermal_df['rack_return_temp_std'].values
        bars = ax3.bar(systems, values, color=colors, edgecolor='white')
        ax3.set_title('Temperature Variability (Std Dev)')
        ax3.set_ylabel('Standard Deviation (°C)')
        ax3.set_xlabel('System')
        ax3.set_xticklabels([s.upper() for s in systems])
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Max Temperature Comparison
    ax4 = axes[1, 1]
    if 'max_rack_return_temp_c' in thermal_df.columns:
        values = thermal_df['max_rack_return_temp_c'].values
        bars = ax4.bar(systems, values, color=colors, edgecolor='white')
        ax4.set_title('Maximum Rack Return Temperature')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_xlabel('System')
        ax4.set_xticklabels([s.upper() for s in systems])
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Thermal Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved thermal comparison to {output_path}")
    
    return fig


def create_thermal_time_series_comparison(
    system_data: Dict[str, pd.DataFrame],
    system_configs: Dict[str, Dict],
    sample_duration: int = 3600,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Create time series comparison of thermal response.
    
    Args:
        system_data: Dictionary of simulation data per system
        system_configs: Dictionary of system configurations
        sample_duration: Number of time steps to plot
        output_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    for system_name, data in system_data.items():
        color = get_system_color(system_name)
        num_cdus = system_configs.get(system_name, {}).get('NUM_CDUS', 1)
        
        # Get temperature columns
        temp_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_sec_r_C'
            for i in range(1, min(num_cdus + 1, 10))  # Limit to first 10 CDUs
        ]
        available_cols = [col for col in temp_cols if col in data.columns]
        
        if available_cols:
            # Average temperature
            avg_temp = data[available_cols].mean(axis=1)[:sample_duration]
            time = np.arange(len(avg_temp))
            
            axes[0].plot(time, avg_temp, label=system_name.upper(), 
                        color=color, alpha=0.8, linewidth=1.5)
            
            # Temperature rate of change
            temp_rate = avg_temp.diff().abs()
            axes[1].plot(time, temp_rate, label=system_name.upper(),
                        color=color, alpha=0.8, linewidth=1.5)
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Average Rack Return Temperature Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('|dT/dt| (°C/s)')
    axes[1].set_title('Temperature Rate of Change')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Thermal Response Time Series Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved thermal time series to {output_path}")
    
    return fig
