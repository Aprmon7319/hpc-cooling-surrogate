"""
Time Series Overlay Visualizer

Creates time series plots showing:
- Input changes (spikes/steps)
- Corresponding output responses
- Visual correlation of dynamics
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_time_series_overlay_plots(
    prepared_data: Dict[str, np.ndarray],
    derivatives_data: Dict[str, np.ndarray],
    impulse_events: Dict[str, List[Dict]],
    output_dir: str,
    sample_range: tuple = None
):
    """
    Create time series overlay plots showing input changes and output responses.
    
    Args:
        prepared_data: Prepared data dictionary
        derivatives_data: Data with computed derivatives
        impulse_events: Detected impulse events
        output_dir: Directory to save plots
        sample_range: Optional (start, end) range to plot
    """
    logger.info("Creating time series overlay plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_vars = ['Q_flow', 'T_Air', 'T_ext']
    key_outputs = ['T_prim_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM']
    
    # Determine sample range
    if sample_range is None:
        time = prepared_data.get('time')
        if time is not None:
            n_samples = len(time)
            # Take a representative window
            if n_samples > 1000:
                sample_range = (n_samples // 4, min(n_samples // 4 + 500, n_samples))
            else:
                sample_range = (0, min(500, n_samples))
    
    start_idx, end_idx = sample_range
    
    for input_name in input_vars:
        _create_input_output_overlay(
            input_name, key_outputs, prepared_data, derivatives_data,
            impulse_events.get(input_name, []),
            start_idx, end_idx, output_dir
        )
    
    # Create combined overview plot
    _create_combined_overlay(
        input_vars, key_outputs, prepared_data, derivatives_data,
        impulse_events, start_idx, end_idx, output_dir
    )
    
    logger.info("Created time series overlay plots")


def _create_input_output_overlay(
    input_name: str,
    output_vars: List[str],
    prepared_data: Dict[str, np.ndarray],
    derivatives_data: Dict[str, np.ndarray],
    events: List[Dict],
    start_idx: int,
    end_idx: int,
    output_dir: str
):
    """Create overlay plot for a single input and multiple outputs."""
    
    fig, axes = plt.subplots(len(output_vars) + 2, 1, figsize=(14, 3 * (len(output_vars) + 2)),
                             sharex=True)
    
    time = prepared_data.get('time', np.arange(end_idx))[start_idx:end_idx]
    
    # Get input data
    input_data = prepared_data['inputs'].get(input_name)
    if input_data is None:
        plt.close()
        return
    
    if input_data.ndim > 1:
        input_data = input_data.mean(axis=1)
    input_data = input_data[start_idx:end_idx]
    
    # Get input derivative
    input_rate = derivatives_data['inputs'].get(input_name)
    if input_rate is not None:
        if input_rate.ndim > 1:
            input_rate = input_rate.mean(axis=1)
        # Align with truncated time
        rate_start = max(0, start_idx - 1)
        rate_end = min(len(input_rate), end_idx - 1)
        input_rate = input_rate[rate_start:rate_end]
    
    # Plot input level
    ax0 = axes[0]
    ax0.plot(time, input_data, 'b-', linewidth=1.5, label=f'{input_name} Level')
    ax0.set_ylabel(f'{input_name}', fontsize=10)
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # Mark impulse events
    event_times = [e['time'] for e in events if start_idx <= e['step_index'] < end_idx]
    for event_time in event_times:
        ax0.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    
    # Plot input rate
    ax1 = axes[1]
    if input_rate is not None and len(input_rate) > 0:
        rate_time = time[:len(input_rate)]
        ax1.plot(rate_time, input_rate, 'r-', linewidth=1.5, label=f'd{input_name}/dt')
        ax1.fill_between(rate_time, 0, input_rate, alpha=0.3, color='red')
    ax1.set_ylabel(f'd{input_name}/dt', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Mark impulse events
    for event_time in event_times:
        ax1.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    
    # Plot outputs
    colors = plt.cm.tab10(np.linspace(0, 1, len(output_vars)))
    
    for i, output_name in enumerate(output_vars):
        ax = axes[i + 2]
        
        output_data = prepared_data['outputs'].get(output_name)
        if output_data is None:
            continue
        
        if output_data.ndim > 1:
            output_data = output_data.mean(axis=1)
        output_data = output_data[start_idx:end_idx]
        
        ax.plot(time, output_data, color=colors[i], linewidth=1.5, label=output_name)
        ax.set_ylabel(output_name, fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Mark impulse events
        for event_time in event_times:
            ax.axvline(x=event_time, color='r', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Time', fontsize=11)
    
    plt.suptitle(
        f'Time Series: {input_name} and Output Responses\n(Red dashed lines = detected step changes)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'time_series_overlay_{input_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def _create_combined_overlay(
    input_vars: List[str],
    output_vars: List[str],
    prepared_data: Dict[str, np.ndarray],
    derivatives_data: Dict[str, np.ndarray],
    impulse_events: Dict[str, List[Dict]],
    start_idx: int,
    end_idx: int,
    output_dir: str
):
    """Create combined overlay plot with all inputs and outputs."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    time = prepared_data.get('time', np.arange(end_idx))[start_idx:end_idx]
    
    # Top plot: All inputs
    ax0 = axes[0]
    colors_input = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, input_name in enumerate(input_vars):
        input_data = prepared_data['inputs'].get(input_name)
        if input_data is None:
            continue
        
        if input_data.ndim > 1:
            input_data = input_data.mean(axis=1)
        input_data = input_data[start_idx:end_idx]
        
        # Normalize for comparison
        input_norm = (input_data - np.nanmean(input_data)) / (np.nanstd(input_data) + 1e-10)
        ax0.plot(time, input_norm, color=colors_input[i], linewidth=1.5, 
                label=f'{input_name} (normalized)')
        
        # Mark impulse events
        events = impulse_events.get(input_name, [])
        for event in events:
            if start_idx <= event['step_index'] < end_idx:
                ax0.axvline(x=event['time'], color=colors_input[i], 
                           linestyle='--', alpha=0.3)
    
    ax0.set_ylabel('Normalized Input', fontsize=11)
    ax0.set_title('Input Signals (Normalized)', fontsize=12)
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # Bottom plot: Key outputs
    ax1 = axes[1]
    colors_output = plt.cm.Set2(np.linspace(0, 1, len(output_vars)))
    
    for i, output_name in enumerate(output_vars):
        output_data = prepared_data['outputs'].get(output_name)
        if output_data is None:
            continue
        
        if output_data.ndim > 1:
            output_data = output_data.mean(axis=1)
        output_data = output_data[start_idx:end_idx]
        
        # Normalize for comparison
        output_norm = (output_data - np.nanmean(output_data)) / (np.nanstd(output_data) + 1e-10)
        ax1.plot(time, output_norm, color=colors_output[i], linewidth=1.5, 
                label=f'{output_name} (normalized)')
    
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Normalized Output', fontsize=11)
    ax1.set_title('Output Responses (Normalized)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.suptitle(
        'Combined Time Series: Inputs and Outputs',
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'time_series_combined.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def create_lag_correlation_plot(
    lag_correlation_data: List[Dict],
    output_dir: str
):
    """
    Create plot showing cross-correlation at different lags.
    
    Args:
        lag_correlation_data: List of lag correlation data from analyzer
        output_dir: Directory to save plot
    """
    logger.info("Creating lag correlation plots...")
    
    if not lag_correlation_data:
        logger.warning("No lag correlation data available")
        return
    
    # Group by input
    inputs = set(d['input'] for d in lag_correlation_data)
    
    for input_name in inputs:
        input_data = [d for d in lag_correlation_data if d['input'] == input_name]
        
        n_outputs = len(input_data)
        n_cols = 3
        n_rows = (n_outputs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_outputs > 1 else [axes]
        
        for idx, data in enumerate(input_data):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            lags = data['lags']
            correlations = data['correlations']
            
            ax.bar(lags, correlations, color='#3498db', alpha=0.7, width=0.8)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Mark optimal lag
            max_idx = np.argmax(np.abs(correlations))
            ax.bar(lags[max_idx], correlations[max_idx], color='#e74c3c', 
                   alpha=0.9, width=0.8, label=f'Optimal lag: {lags[max_idx]}')
            
            ax.set_xlabel('Lag (time steps)', fontsize=10)
            ax.set_ylabel('Correlation', fontsize=10)
            ax.set_title(f'{data["output"]}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused axes
        for idx in range(len(input_data), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Lagged Cross-Correlation: {input_name} → Outputs',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'lag_correlation_{input_name}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    logger.info("Created lag correlation plots")
