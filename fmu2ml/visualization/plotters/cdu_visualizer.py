import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import random
from typing import List, Optional, Dict

class CDUVisualizer:
    """Visualizer for CDU (Cooling Distribution Unit) parameters"""
    
    def __init__(self):
        self.parameter_groups = {
            'Flow Rates': {
                'Primary': ['m_flow_prim', 'V_flow_prim_GPM'],
                'Secondary': ['m_flow_sec', 'V_flow_sec_GPM']
            },
            'Power': ['W_flow_CDUP', 'W_flow_CDUP_kW'],
            'Temperatures': {
                'Primary': ['T_prim_s', 'T_prim_s_C', 'T_prim_r', 'T_prim_r_C'],
                'Secondary': ['T_sec_s', 'T_sec_s_C', 'T_sec_r', 'T_sec_r_C']
            },
            'Pressures': {
                'Primary': ['p_prim_s', 'p_prim_s_psig', 'p_prim_r', 'p_prim_r_psig'],
                'Secondary': ['p_sec_s', 'p_sec_s_psig', 'p_sec_r', 'p_sec_r_psig']
            }
        }
        
        self.units = {
            'm_flow_prim': 'kg/s', 'V_flow_prim_GPM': 'GPM',
            'm_flow_sec': 'kg/s', 'V_flow_sec_GPM': 'GPM',
            'W_flow_CDUP': 'W', 'W_flow_CDUP_kW': 'kW',
            'T_prim_s': 'K', 'T_prim_s_C': '°C',
            'T_prim_r': 'K', 'T_prim_r_C': '°C',
            'T_sec_s': 'K', 'T_sec_s_C': '°C',
            'T_sec_r': 'K', 'T_sec_r_C': '°C',
            'p_prim_s': 'Pa', 'p_prim_s_psig': 'psig',
            'p_prim_r': 'Pa', 'p_prim_r_psig': 'psig',
            'p_sec_s': 'Pa', 'p_sec_s_psig': 'psig',
            'p_sec_r': 'Pa', 'p_sec_r_psig': 'psig'
        }
    
    def extract_cdu_data(self, df: pd.DataFrame, compute_blocks: List[int]) -> Dict[int, pd.DataFrame]:
        """Extract CDU data for specified compute blocks"""
        cdu_data = {}
        
        for cb in compute_blocks:
            # Collect all columns for this compute block
            cb_cols = [col for col in df.columns 
                    if f'simulator[1].datacenter[1].computeBlock[{cb}].cdu[1].summary.' in col]
            
            if cb_cols:
                cdu_df = df[cb_cols].copy()
                # Rename columns to remove prefix for easier access
                cdu_df.columns = [col.split('.')[-1] for col in cdu_df.columns]
                cdu_data[cb] = cdu_df
        
        return cdu_data
    def plot_random_cdus_overview(self, df: pd.DataFrame, n_cdus: int = 5, selected_cbs: Optional[List[int]] = None, 
                             time_range: Optional[tuple] = None,
                             figsize: tuple = (20, 16)):
        """Plot overview of randomly selected CDUs"""
        
        # Randomly select compute blocks
        all_cbs = list(range(1, 52))
        if selected_cbs is None:
            selected_cbs = random.sample(all_cbs, min(n_cdus, len(all_cbs)))
        selected_cbs.sort()
        
        print(f"Selected Compute Blocks: {selected_cbs}")
        
        # Extract data
        cdu_data = self.extract_cdu_data(df, selected_cbs)
        
        if not cdu_data:
            print("No CDU data found for selected compute blocks")
            return
        
        # Apply time range if specified
        if time_range:
            for cb in cdu_data:
                cdu_data[cb] = cdu_data[cb].iloc[time_range[0]:time_range[1]]
        
        # Calculate base data length for rolling window calculations
        data_lengths = [len(cdu_data[cb]) for cb in cdu_data if len(cdu_data[cb]) > 0]
        if data_lengths:
            avg_length = int(np.mean(data_lengths))
        else:
            avg_length = 1000
        
        # Define rolling window sizes for different measurement types
        # Different measurements may need different smoothing levels
        rolling_windows = {
            'power': max(10, min(300, avg_length // 50)),      # 2% of data
            'flow': max(15, min(300, avg_length // 75)),       # 1.33% of data
            'temperature': max(20, min(400, avg_length // 40)), # 2.5% of data (more smoothing)
            'pressure': max(10, min(200, avg_length // 100)),   # 1% of data
            'efficiency': max(25, min(500, avg_length // 30)),  # 3.33% of data (most smoothing)
        }
        
        print(f"Rolling windows: {rolling_windows}")
        
        # Create copies of data for different rolling windows
        cdu_data_smoothed = {}
        for cb in cdu_data:
            cdu_data_smoothed[cb] = {}
            
            # Power measurements
            if 'W_flow_CDUP_kW' in cdu_data[cb].columns:
                cdu_data_smoothed[cb]['W_flow_CDUP_kW'] = cdu_data[cb]['W_flow_CDUP_kW'].rolling(
                    window=rolling_windows['power'], center=True).mean()
            
            # Flow measurements
            for col in ['V_flow_prim_GPM', 'V_flow_sec_GPM']:
                if col in cdu_data[cb].columns:
                    cdu_data_smoothed[cb][col] = cdu_data[cb][col].rolling(
                        window=rolling_windows['flow'], center=True).mean()
            
            # Temperature measurements
            for col in ['T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C']:
                if col in cdu_data[cb].columns:
                    cdu_data_smoothed[cb][col] = cdu_data[cb][col].rolling(
                        window=rolling_windows['temperature'], center=True).mean()
            
            # Pressure measurements
            for col in ['p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']:
                if col in cdu_data[cb].columns:
                    cdu_data_smoothed[cb][col] = cdu_data[cb][col].rolling(
                        window=rolling_windows['pressure'], center=True).mean()
        
        # Get the overall time range for vertical lines
        all_indices = []
        for cb in cdu_data:
            if len(cdu_data[cb]) > 0:
                all_indices.extend(cdu_data[cb].index.tolist())
        
        if all_indices:
            min_time = min(all_indices)
            max_time = max(all_indices)
            # Calculate hour markers
            hour_markers = list(range(int(min_time // 3600) * 3600, int(max_time) + 3600, 3600))
        else:
            hour_markers = []
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'CDU Parameters Overview - Compute Blocks: {selected_cbs}', fontsize=16)
        
        # 1. Power consumption
        ax1 = fig.add_subplot(gs[0, :])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'W_flow_CDUP_kW' in cdu_data_smoothed[cb]:
                ax1.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['W_flow_CDUP_kW'], 
                        label=f'CB-{cb}', linewidth=2)
        # Add hour markers
        for hour in hour_markers:
            ax1.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.set_title(f'CDU Power Consumption (RW: {rolling_windows["power"]})')
        ax1.set_ylabel('Power (kW)')
        ax1.legend(ncol=min(5, len(selected_cbs)))
        ax1.grid(True, alpha=0.3)
        
        # 2. Primary flow rates
        ax2 = fig.add_subplot(gs[1, 0])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'V_flow_prim_GPM' in cdu_data_smoothed[cb]:
                ax2.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['V_flow_prim_GPM'], 
                        label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax2.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.set_title(f'Primary Flow Rate (RW: {rolling_windows["flow"]})')
        ax2.set_ylabel('Flow Rate (GPM)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Secondary flow rates
        ax3 = fig.add_subplot(gs[1, 1])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'V_flow_sec_GPM' in cdu_data_smoothed[cb]:
                ax3.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['V_flow_sec_GPM'], 
                        label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax3.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax3.set_title(f'Secondary Flow Rate (RW: {rolling_windows["flow"]})')
        ax3.set_ylabel('Flow Rate (GPM)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Temperature differentials
        ax4 = fig.add_subplot(gs[1, 2])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed:
                if 'T_prim_r_C' in cdu_data_smoothed[cb] and 'T_prim_s_C' in cdu_data_smoothed[cb]:
                    delta_t = cdu_data_smoothed[cb]['T_prim_r_C'] - cdu_data_smoothed[cb]['T_prim_s_C']
                    ax4.plot(cdu_data[cb].index, delta_t, label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax4.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax4.set_title(f'Primary Temperature Differential (RW: {rolling_windows["temperature"]})')
        ax4.set_ylabel('Delta T (°C)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Primary supply temperature
        ax5 = fig.add_subplot(gs[2, 0])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'T_prim_s_C' in cdu_data_smoothed[cb]:
                ax5.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['T_prim_s_C'], 
                        label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax5.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax5.set_title(f'Primary Supply Temperature (RW: {rolling_windows["temperature"]})')
        ax5.set_ylabel('Temperature (°C)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Primary return temperature
        ax6 = fig.add_subplot(gs[2, 1])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'T_prim_r_C' in cdu_data_smoothed[cb]:
                ax6.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['T_prim_r_C'], 
                        label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax6.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax6.set_title(f'Primary Return Temperature (RW: {rolling_windows["temperature"]})')
        ax6.set_ylabel('Temperature (°C)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Pressure drop (primary)
        ax7 = fig.add_subplot(gs[2, 2])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed:
                if 'p_prim_s_psig' in cdu_data_smoothed[cb] and 'p_prim_r_psig' in cdu_data_smoothed[cb]:
                    pressure_drop = cdu_data_smoothed[cb]['p_prim_s_psig'] - cdu_data_smoothed[cb]['p_prim_r_psig']
                    ax7.plot(cdu_data[cb].index, pressure_drop, label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax7.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax7.set_title(f'Primary Pressure Drop (RW: {rolling_windows["pressure"]})')
        ax7.set_ylabel('Pressure Drop (psig)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Secondary temperatures
        ax8 = fig.add_subplot(gs[3, 0])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed and 'T_sec_s_C' in cdu_data_smoothed[cb]:
                ax8.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['T_sec_s_C'], 
                        label=f'CB-{cb} Supply', linewidth=1.5, linestyle='-')
            if cb in cdu_data_smoothed and 'T_sec_r_C' in cdu_data_smoothed[cb]:
                ax8.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['T_sec_r_C'], 
                        label=f'CB-{cb} Return', linewidth=1.5, linestyle='--')
        # Add hour markers
        for hour in hour_markers:
            ax8.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax8.set_title(f'Secondary Temperatures (RW: {rolling_windows["temperature"]})')
        ax8.set_ylabel('Temperature (°C)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Efficiency metric (Power per flow) - uses heaviest smoothing
        ax9 = fig.add_subplot(gs[3, 1])
        for cb in selected_cbs:
            if cb in cdu_data_smoothed:
                if 'W_flow_CDUP_kW' in cdu_data_smoothed[cb] and 'V_flow_prim_GPM' in cdu_data_smoothed[cb]:
                    # Apply extra smoothing to efficiency since it's a derived metric
                    power_smooth = cdu_data[cb]['W_flow_CDUP_kW'].rolling(
                        window=rolling_windows['efficiency'], center=True).mean()
                    flow_smooth = cdu_data[cb]['V_flow_prim_GPM'].rolling(
                        window=rolling_windows['efficiency'], center=True).mean()
                    # Avoid division by zero
                    flow_smooth = flow_smooth.replace(0, np.nan)
                    efficiency = power_smooth / flow_smooth
                    ax9.plot(cdu_data[cb].index, efficiency, label=f'CB-{cb}', linewidth=1.5)
        # Add hour markers
        for hour in hour_markers:
            ax9.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax9.set_title(f'CDU Efficiency (kW/GPM) (RW: {rolling_windows["efficiency"]})')
        ax9.set_ylabel('Efficiency')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Pressure values comparison
        ax10 = fig.add_subplot(gs[3, 2])
        for cb in selected_cbs[:3]:  # Limit to 3 for clarity
            if cb in cdu_data_smoothed:
                if 'p_prim_s_psig' in cdu_data_smoothed[cb]:
                    ax10.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['p_prim_s_psig'], 
                            label=f'CB-{cb} Pri Supply', linewidth=1.5)
                if 'p_sec_s_psig' in cdu_data_smoothed[cb]:
                    ax10.plot(cdu_data[cb].index, cdu_data_smoothed[cb]['p_sec_s_psig'], 
                            label=f'CB-{cb} Sec Supply', linewidth=1.5, linestyle='--')
        # Add hour markers
        for hour in hour_markers:
            ax10.axvline(x=hour, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax10.set_title(f'Supply Pressures Comparison (RW: {rolling_windows["pressure"]})')
        ax10.set_ylabel('Pressure (psig)')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, selected_cbs
    def plot_single_cdu_detailed(self, df: pd.DataFrame, compute_block: int,
                                figsize: tuple = (20, 12)):
        """Detailed view of a single CDU"""
        
        # Extract data for single compute block
        cdu_data = self.extract_cdu_data(df, [compute_block])
        
        if compute_block not in cdu_data:
            print(f"No data found for compute block {compute_block}")
            return
        
        data = cdu_data[compute_block]
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle(f'Detailed CDU Analysis - Compute Block {compute_block}', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot each parameter
        plot_configs = [
            ('W_flow_CDUP_kW', 'Power Consumption', 'kW', 'tab:red'),
            ('V_flow_prim_GPM', 'Primary Flow Rate', 'GPM', 'tab:blue'),
            ('V_flow_sec_GPM', 'Secondary Flow Rate', 'GPM', 'tab:cyan'),
            ('T_prim_s_C', 'Primary Supply Temp', '°C', 'tab:orange'),
            ('T_prim_r_C', 'Primary Return Temp', '°C', 'tab:red'),
            ('T_sec_s_C', 'Secondary Supply Temp', '°C', 'tab:green'),
            ('T_sec_r_C', 'Secondary Return Temp', '°C', 'tab:olive'),
            ('p_prim_s_psig', 'Primary Supply Pressure', 'psig', 'tab:purple'),
            ('p_prim_r_psig', 'Primary Return Pressure', 'psig', 'tab:pink'),
            ('p_sec_s_psig', 'Secondary Supply Pressure', 'psig', 'tab:brown'),
            ('p_sec_r_psig', 'Secondary Return Pressure', 'psig', 'tab:gray'),
        ]
        
        for idx, (param, title, unit, color) in enumerate(plot_configs):
            if idx < len(axes) and param in data:
                ax = axes[idx]
                ax.plot(data.index, data[param], color=color, linewidth=1.5)
                ax.set_title(title)
                ax.set_ylabel(f'{unit}')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data[param].mean()
                std_val = data[param].std()
                ax.axhline(mean_val, color='black', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
                ax.fill_between(data.index, mean_val - std_val, mean_val + std_val, 
                               alpha=0.2, color='gray', label=f'±1 STD')
                ax.legend(loc='upper right', fontsize=8)
        
        # Use last subplot for correlation heatmap
        ax_corr = axes[-1]
        # Calculate correlations between numeric parameters
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, ax=ax_corr, cmap='coolwarm', center=0, 
                       annot=True, fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
            ax_corr.set_title('Parameter Correlations')
        
        plt.tight_layout()
        return fig
    
# Simple usage example
def visualize_cdu_data(df: pd.DataFrame, save: bool = False, selected_cbs: Optional[List[int]] = None):
    """Simple function to create CDU visualizations"""
    
    visualizer = CDUVisualizer()
    
    # 1. Overview of random CDUs
    print("Creating overview visualization...")
    fig1, selected_cbs = visualizer.plot_random_cdus_overview(df, n_cdus=5, selected_cbs=selected_cbs)
    if save == True:    
        plt.savefig('cdu_overview.png', dpi=300, bbox_inches='tight')
    
    # 2. Detailed view of one CDU
    print(f"\nCreating detailed view for CDU {selected_cbs[0]}...")
    fig2 = visualizer.plot_single_cdu_detailed(df, selected_cbs[0])
    if save == True:
        plt.savefig(f'cdu_{selected_cbs[0]}_detailed.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return visualizer

