import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


def create_slice_plots(
    prepared_data: Dict[str, np.ndarray],
    output_vars: List[str],
    input1: str = 'Q_flow',
    input2: str = 'T_Air',
    n_slices: int = 5,
    output_dir: str = '.'
):
    """
    Create slice plots showing output vs input1 at different input2 levels.
    
    Args:
        prepared_data: Prepared data dictionary
        output_vars: List of output variables to plot
        input1: Primary input variable (x-axis)
        input2: Slicing variable (different lines)
        n_slices: Number of slices for input2
        output_dir: Directory to save plots
    """
    logger.info(f"Creating slice plots: {output_vars[0]} vs {input1} at different {input2} levels...")
    
    # Get data
    input1_data = prepared_data['inputs'][input1]
    input2_data = prepared_data['inputs'][input2]
    
    # Get a sample output to determine expected length
    sample_output = prepared_data['outputs'][output_vars[0]]
    if sample_output.ndim > 1:
        expected_length = sample_output.flatten().shape[0]
    else:
        expected_length = len(sample_output)
    
    # Flatten and broadcast input data to match output length
    if input1_data.ndim > 1:
        input1_flat = input1_data.flatten()
    else:
        # Broadcast to match output length (for T_ext)
        num_cdus = expected_length // len(input1_data)
        input1_flat = np.repeat(input1_data, num_cdus)
    
    if input2_data.ndim > 1:
        input2_flat = input2_data.flatten()
    else:
        # Broadcast to match output length (for T_ext)
        num_cdus = expected_length // len(input2_data)
        input2_flat = np.repeat(input2_data, num_cdus)
    
    # Ensure all arrays have the same length
    min_length = min(len(input1_flat), len(input2_flat), expected_length)
    input1_flat = input1_flat[:min_length]
    input2_flat = input2_flat[:min_length]
    
    # Create slices for input2
    input2_min, input2_max = np.nanpercentile(input2_flat, [5, 95])
    slice_levels = np.linspace(input2_min, input2_max, n_slices)
    slice_tolerance = (input2_max - input2_min) / (2 * n_slices)
    
    # Plot for each output
    for output_name in output_vars:
        output_data = prepared_data['outputs'][output_name]
        
        # Flatten output data
        if output_data.ndim > 1:
            output_flat = output_data.flatten()
        else:
            output_flat = output_data.copy()
        
        # Ensure output has same length
        output_flat = output_flat[:min_length]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_slices))
        
        for idx, slice_level in enumerate(slice_levels):
            # Find data points near this slice level
            mask = np.abs(input2_flat - slice_level) < slice_tolerance
            mask &= ~(np.isnan(input1_flat) | np.isnan(output_flat))
            
            if np.sum(mask) < 10:
                continue
            
            input1_slice = input1_flat[mask]
            output_slice = output_flat[mask]
            
            # Sort by input1 for smooth line
            sort_idx = np.argsort(input1_slice)
            input1_sorted = input1_slice[sort_idx]
            output_sorted = output_slice[sort_idx]
            
            # Plot with binning for clarity
            n_bins = 20
            bin_edges = np.linspace(input1_sorted.min(), input1_sorted.max(), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_means = []
            
            for i in range(n_bins):
                bin_mask = (input1_sorted >= bin_edges[i]) & (input1_sorted < bin_edges[i+1])
                if np.sum(bin_mask) > 0:
                    bin_means.append(np.mean(output_sorted[bin_mask]))
                else:
                    bin_means.append(np.nan)
            
            # Convert to Celsius/kW if needed for labels
            if input2 == 'T_Air' or input2 == 'T_ext':
                label_val = slice_level - 273.15  # K to °C
                label_unit = '°C'
            elif input2 == 'Q_flow':
                label_val = slice_level / 1000  # W to kW
                label_unit = 'kW'
            else:
                label_val = slice_level
                label_unit = ''
            
            ax.plot(
                bin_centers, bin_means,
                marker='o', linewidth=2, markersize=4,
                color=colors[idx],
                label=f'{input2} = {label_val:.1f} {label_unit}'
            )
        
        # Format x-axis label
        if input1 == 'Q_flow':
            xlabel = f'{input1} (W)'
        elif input1 == 'T_Air' or input1 == 'T_ext':
            xlabel = f'{input1} (K)'
        else:
            xlabel = input1
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(output_name, fontsize=12)
        ax.set_title(f'{output_name} Response to {input1}\nat Different {input2} Levels', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'slice_plot_{output_name}_{input1}_vs_{input2}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    logger.info(f"Created slice plots for {len(output_vars)} outputs")