import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from matplotlib.gridspec import GridSpec
import random
from raps.config import ConfigManager

def _get_cdu_output_columns(compute_block: int, variables: list = None):
    """Get output column names for a specific CDU"""
    if variables is None:
        variables = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                    'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                    'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
    
    return {
        var: f'simulator[1].datacenter[1].computeBlock[{compute_block}].cdu[1].summary.{var}'
        for var in variables
    }


@dataclass
class StandardInputOutputVisualizer:
    """Visualizer for datacenter cooling model inputs and outputs"""
    data: pd.DataFrame
    cdu_count: int = 49
    input_columns: Dict[str, List[str]] = None
    cdu_outputs: List[str] = None
    dc_outputs: List[str] = None
    system_name: str = 'marconi100'
    

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with the dataset
        
        Args:
            data: DataFrame containing all the metrics
        """
        self.data = data
        config  = ConfigManager(system_name=self.system_name).get_config()
        self.cdu_count = 49 if self.system_name == 'marconi100' else config.get('NUM_CDUS', 257)
        
        # Define input columns
        self.input_columns = {
            'Q_flow_total': [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total' 
                            for i in range(1, self.cdu_count + 1)],
            'T_Air': [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air' 
                     for i in range(1, self.cdu_count + 1)],
            'T_ext': 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        }
        
        # Define output columns per CDU
        self.cdu_outputs = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW', 
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        # Define datacenter-level outputs
        self.dc_outputs = ['V_flow_prim_GPM', 'pue']
        
    def apply_rolling_window(self, series: pd.Series, window: int) -> pd.Series:
        """Apply rolling window average to a series"""
        return series.rolling(window=window, center=True, min_periods=1).mean()
    
    def select_cdus(self, cdu_list: Optional[List[int]] = None, random_select: int = 5) -> List[int]:
        """
        Select CDUs based on input or randomly
        
        Args:
            cdu_list: Specific list of CDUs to select (1-indexed)
            random_select: Number of CDUs to randomly select if cdu_list is None
            
        Returns:
            List of selected CDU indices (1-indexed)
        """
        if cdu_list is not None:
            # Validate CDU numbers
            valid_cdus = [cdu for cdu in cdu_list if 1 <= cdu <= self.cdu_count]
            if len(valid_cdus) != len(cdu_list):
                print(f"Warning: Some CDUs were out of range (1-{self.cdu_count})")
            return valid_cdus
        else:
            return random.sample(range(1, self.cdu_count + 1), min(random_select, self.cdu_count))
    
    def plot_inputs(self, 
                   selected_cdus: List[int],
                   rolling_windows: Union[int, Dict[str, int]] = 1,
                   time_range: Optional[tuple] = None,
                   figsize: tuple = (16, 10)):
        """
        Plot input metrics for selected CDUs
        
        Args:
            selected_cdus: List of CDU numbers to plot (1-indexed)
            rolling_windows: Either a single window size or dict with feature-specific windows
            time_range: Optional tuple of (start_idx, end_idx) to limit time range
            figsize: Figure size tuple
            
        Returns:
            matplotlib figure object
        """
        # Convert rolling_windows to dict if single value provided
        if isinstance(rolling_windows, int):
            default_window = rolling_windows
            rolling_windows = {}
        else:
            default_window = 1
        
        # Set up time range
        if time_range:
            start_idx, end_idx = time_range
            time_data = self.data.iloc[start_idx:end_idx]
        else:
            time_data = self.data
            
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('Datacenter Cooling Model - Input Metrics', fontsize=16, y=0.995)
        
        # Plot 1: Q_flow_total for selected CDUs
        ax1 = axes[0]
        for cdu in selected_cdus:
            col_name = f'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_Q_flow_total'
            if col_name in time_data.columns:
                window = rolling_windows.get('Q_flow_total', default_window)
                values = self.apply_rolling_window(time_data[col_name], window)
                ax1.plot(time_data.index, values, label=f'CDU {cdu}', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Q_flow_total [kW]', fontsize=12)
        ax1.set_title(f'Heat Flow Total (Rolling window: {rolling_windows.get("Q_flow_total", default_window)})', 
                     fontsize=12)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: T_Air for selected CDUs
        ax2 = axes[1]
        for cdu in selected_cdus:
            col_name = f'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_T_Air'
            if col_name in time_data.columns:
                window = rolling_windows.get('T_Air', default_window)
                values = self.apply_rolling_window(time_data[col_name], window)
                ax2.plot(time_data.index, values, label=f'CDU {cdu}', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('T_Air [°K]', fontsize=12)
        ax2.set_title(f'Air Temperature (Rolling window: {rolling_windows.get("T_Air", default_window)})', 
                     fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: External Temperature
        ax3 = axes[2]
        if self.input_columns['T_ext'] in time_data.columns:
            window = rolling_windows.get('T_ext', default_window)
            values = self.apply_rolling_window(time_data[self.input_columns['T_ext']], window)
            ax3.plot(time_data.index, values, color='red', linewidth=2)
        ax3.set_ylabel('T_ext [°K]', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_title(f'External Temperature (Rolling window: {rolling_windows.get("T_ext", default_window)})', 
                     fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_outputs(self,
                    selected_cdus: List[int],
                    rolling_windows: Union[int, Dict[str, int]] = 1,
                    time_range: Optional[tuple] = None,
                    figsize: tuple = (20, 16)):
        """
        Plot output metrics for selected CDUs (12 metrics + HTC)
        
        Args:
            selected_cdus: List of CDU numbers to plot (1-indexed)
            rolling_windows: Either a single window size or dict with feature-specific windows
            time_range: Optional tuple of (start_idx, end_idx) to limit time range
            figsize: Figure size tuple
            
        Returns:
            matplotlib figure object
        """
        # Convert rolling_windows to dict if single value provided
        if isinstance(rolling_windows, int):
            default_window = rolling_windows
            rolling_windows = {}
        else:
            default_window = 1
        
        # Set up time range
        if time_range:
            start_idx, end_idx = time_range
            time_data = self.data.iloc[start_idx:end_idx]
        else:
            time_data = self.data
            
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Datacenter Cooling Model - Output Metrics', fontsize=16, y=0.995)
        
        # Plot layout mapping
        plot_config = [
            # Column 1: Temperatures
            (0, 0, 'T_prim_s_C', 'Primary Supply Temp [°C]'),
            (1, 0, 'T_prim_r_C', 'Primary Return Temp [°C]'),
            (2, 0, 'T_sec_s_C', 'Secondary Supply Temp [°C]'),
            (3, 0, 'T_sec_r_C', 'Secondary Return Temp [°C]'),
            # Column 2: Pressures
            (0, 1, 'p_prim_s_psig', 'Primary Supply Pressure [psig]'),
            (1, 1, 'p_prim_r_psig', 'Primary Return Pressure [psig]'),
            (2, 1, 'p_sec_s_psig', 'Secondary Supply Pressure [psig]'),
            (3, 1, 'p_sec_r_psig', 'Secondary Return Pressure [psig]'),

            # Column 3: Flow rates and Work Done
            (0, 2, 'V_flow_prim_GPM', 'Primary Flow Rate [GPM]'),
            (1, 2, 'V_flow_sec_GPM', 'Secondary Flow Rate [GPM]'),
            (2, 2, 'W_flow_CDUP_kW', 'CDUP Power [kW]'),
        ]
        
        # Plot each metric
        for row, col, metric, ylabel in plot_config:
            ax = fig.add_subplot(gs[row, col])
            
            for cdu in selected_cdus:
                col_name = f'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.{metric}'
                if col_name in time_data.columns:
                    window = rolling_windows.get(metric, default_window)
                    values = self.apply_rolling_window(time_data[col_name], window)
                    ax.plot(time_data.index, values, label=f'CDU {cdu}', linewidth=1.5, alpha=0.8)
            
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{ylabel.split("[")[0].strip()} (Window: {rolling_windows.get(metric, default_window)})', 
                        fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Only show x-label for bottom plots
            if (row, col) not in [(3, 0), (3, 1), (2, 2)]:
                ax.set_xticklabels([])
        
        # Plot HTC spanning bottom row
        ax_htc = fig.add_subplot(gs[4, :])
        for cdu in selected_cdus:
            col_name = f'simulator[1].datacenter[1].computeBlock[{cdu}].cabinet[1].summary.htc'
            if col_name in time_data.columns:
                window = rolling_windows.get('htc', default_window)
                values = self.apply_rolling_window(time_data[col_name], window)
                ax_htc.plot(time_data.index, values, label=f'CDU {cdu}', linewidth=1.5, alpha=0.8)
        
        ax_htc.set_ylabel('HTC [W/(m²·K)]', fontsize=12)
        ax_htc.set_xlabel('Time', fontsize=12)
        ax_htc.set_title(f'Heat Transfer Coefficient (Rolling window: {rolling_windows.get("htc", default_window)})', 
                        fontsize=12)
        ax_htc.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax_htc.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_datacenter_statistics(self,
                                 rolling_windows: Union[int, Dict[str, int]] = 1,
                                 time_range: Optional[tuple] = None,
                                 figsize: tuple = (14, 8)):
        """
        Plot overall datacenter statistics (total flow and PUE)
        
        Args:
            rolling_windows: Either a single window size or dict with feature-specific windows
            time_range: Optional tuple of (start_idx, end_idx) to limit time range
            figsize: Figure size tuple
            
        Returns:
            matplotlib figure object
        """
        # Convert rolling_windows to dict if single value provided
        if isinstance(rolling_windows, int):
            default_window = rolling_windows
            rolling_windows = {}
        else:
            default_window = 1
        
        # Set up time range
        if time_range:
            start_idx, end_idx = time_range
            time_data = self.data.iloc[start_idx:end_idx]
        else:
            time_data = self.data
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle('Datacenter-Level Statistics', fontsize=16, y=0.995)
        
        # Plot 1: Datacenter total flow
        col_name = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if col_name in time_data.columns:
            window = rolling_windows.get('datacenter_V_flow_prim_GPM', default_window)
            values = self.apply_rolling_window(time_data[col_name], window)
            ax1.plot(time_data.index, values, linewidth=2, color='blue')
            ax1.fill_between(time_data.index, values, alpha=0.3, color='blue')
        
        ax1.set_ylabel('Total Primary Flow [GPM]', fontsize=12)
        ax1.set_title(f'Datacenter Total Primary Flow Rate (Rolling window: {rolling_windows.get("datacenter_V_flow_prim_GPM", default_window)})', 
                     fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        if col_name in time_data.columns:
            mean_val = values.mean()
            std_val = values.std()
            ax1.text(0.02, 0.95, f'Mean: {mean_val:.2f} GPM\nStd: {std_val:.2f} GPM', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: PUE
        if 'pue' in time_data.columns:
            window = rolling_windows.get('pue', default_window)
            values = self.apply_rolling_window(time_data['pue'], window)
            ax2.plot(time_data.index, values, linewidth=2, color='red')
            ax2.fill_between(time_data.index, values, alpha=0.3, color='red')
            
            # Add statistics text
            mean_val = values.mean()
            std_val = values.std()
            ax2.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_ylabel('PUE', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_title(f'Power Usage Effectiveness (Rolling window: {rolling_windows.get("pue", default_window)})', 
                     fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add PUE reference lines
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal PUE')
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Good PUE')
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Average PUE')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, 
                  selected_cdus: List[int],
                  rolling_windows: Union[int, Dict[str, int]] = 1,
                  time_range: Optional[tuple] = None,
                  output_prefix: str = 'datacenter_metrics'):
        """
        Generate and save all plots
        
        Args:
            selected_cdus: List of CDU numbers to plot
            rolling_windows: Rolling window configuration
            time_range: Optional time range tuple
            output_prefix: Prefix for saved files
        """
        # Generate input plots
        fig_inputs = self.plot_inputs(selected_cdus, rolling_windows, time_range)
        fig_inputs.savefig(f'{output_prefix}_inputs.png', dpi=300, bbox_inches='tight')
        print(f"Saved input plots to {output_prefix}_inputs.png")
        
        # Generate output plots
        fig_outputs = self.plot_outputs(selected_cdus, rolling_windows, time_range)
        fig_outputs.savefig(f'{output_prefix}_outputs.png', dpi=300, bbox_inches='tight')
        print(f"Saved output plots to {output_prefix}_outputs.png")
        
        # Generate datacenter statistics plots
        fig_stats = self.plot_datacenter_statistics(rolling_windows, time_range)
        fig_stats.savefig(f'{output_prefix}_statistics.png', dpi=300, bbox_inches='tight')
        print(f"Saved statistics plots to {output_prefix}_statistics.png")
        
        plt.close('all')
    
    def generate_summary_stats(self, selected_cdus: List[int]) -> pd.DataFrame:
        """
        Generate summary statistics for selected CDUs
        
        Args:
            selected_cdus: List of CDU numbers
            
        Returns:
            DataFrame with summary statistics
        """
        stats_data = []
        
        for cdu in selected_cdus:
            cdu_stats = {'CDU': cdu}
            
            # Input stats
            q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_Q_flow_total'
            t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_T_Air'
            
            if q_flow_col in self.data.columns:
                cdu_stats['Q_flow_mean'] = self.data[q_flow_col].mean()
                cdu_stats['Q_flow_std'] = self.data[q_flow_col].std()
            
            if t_air_col in self.data.columns:
                cdu_stats['T_Air_mean'] = self.data[t_air_col].mean()
                cdu_stats['T_Air_std'] = self.data[t_air_col].std()
            
            # Output stats
            for metric in self.cdu_outputs:
                col_name = f'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.{metric}'
                if col_name in self.data.columns:
                    cdu_stats[f'{metric}_mean'] = self.data[col_name].mean()
                    cdu_stats[f'{metric}_std'] = self.data[col_name].std()
            
            # HTC stats
            htc_col = f'simulator[1].datacenter[1].computeBlock[{cdu}].cabinet[1].summary.htc'
            if htc_col in self.data.columns:
                cdu_stats['htc_mean'] = self.data[htc_col].mean()
                cdu_stats['htc_std'] = self.data[htc_col].std()
            
            stats_data.append(cdu_stats)
        
        return pd.DataFrame(stats_data)


@dataclass
class FMUOutputAnalyzer:
    """Analyzer for FMU datacenter cooling model outputs"""
    
    def __init__(self):
        self.n_compute_blocks = 51
        self.output_categories = {
            'heat_flow': [],
            'htc': [],
            'datacenter_summary': [],
            'cdu_metrics': [],
            'temperatures': [],
            'efficiency': [],
            'environment': []
        }
        self.cdu_suffixes = [
            'm_flow_prim', 'V_flow_prim_GPM', 'm_flow_sec', 'V_flow_sec_GPM',
            'W_flow_CDUP', 'W_flow_CDUP_kW', 'T_prim_s', 'T_prim_s_C',
            'T_prim_r', 'T_prim_r_C', 'T_sec_s', 'T_sec_s_C',
            'T_sec_r', 'T_sec_r_C', 'p_prim_s', 'p_prim_s_psig',
            'p_prim_r', 'p_prim_r_psig', 'p_sec_s', 'p_sec_s_psig',
            'p_sec_r', 'p_sec_r_psig'
        ]
    
    def parse_fmu_data(self, raw_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Parse raw FMU output into organized categories"""
        
        parsed_data = {}
        
        # Extract heat flow data for all compute blocks
        heat_flow_cols = [col for col in raw_data.columns if 'sources_Q_flow_total' in col]
        if heat_flow_cols:
            parsed_data['heat_flow'] = raw_data[heat_flow_cols].copy()
            # Rename columns to simpler format
            parsed_data['heat_flow'].columns = [f'CB_{i+1}_Q_flow' for i in range(len(heat_flow_cols))]
        
        # Extract CDU data for all compute blocks
        cdu_data = {}
        for suffix in self.cdu_suffixes:
            cols = [col for col in raw_data.columns if f'.cdu[1].summary.{suffix}' in col]
            if cols:
                cdu_data[suffix] = raw_data[cols].copy()
                cdu_data[suffix].columns = [f'CB_{i+1}' for i in range(len(cols))]
        
        if cdu_data:
            parsed_data['cdu'] = pd.concat(cdu_data, axis=1, keys=self.cdu_suffixes)
        
        # Extract datacenter summary
        datacenter_cols = [col for col in raw_data.columns if 'datacenter[1].summary' in col]
        if datacenter_cols:
            parsed_data['datacenter_summary'] = raw_data[datacenter_cols].copy()
        
        # Extract air temperatures
        air_temp_cols = [col for col in raw_data.columns if 'sources_T_Air' in col]
        if air_temp_cols:
            parsed_data['air_temps'] = raw_data[air_temp_cols].copy()
            parsed_data['air_temps'].columns = [f'CB_{i+1}_T_Air' for i in range(len(air_temp_cols))]
        
        # Extract PUE and external temperature
        if 'pue' in raw_data.columns:
            parsed_data['pue'] = raw_data['pue'].copy()
        
        ext_temp_cols = [col for col in raw_data.columns if 'sources_T_ext' in col]
        if ext_temp_cols:
            parsed_data['external_temp'] = raw_data[ext_temp_cols].copy()
        
        return parsed_data
    
    def calculate_metrics(self, parsed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate key performance metrics"""
        
        metrics = pd.DataFrame(index=parsed_data.get('pue', pd.Series()).index)
        
        # Total heat load
        if 'heat_flow' in parsed_data:
            metrics['total_heat_load'] = parsed_data['heat_flow'].sum(axis=1)
            metrics['avg_heat_per_cb'] = parsed_data['heat_flow'].mean(axis=1)
            metrics['heat_load_std'] = parsed_data['heat_flow'].std(axis=1)
        
        # CDU metrics
        if 'cdu' in parsed_data:
            # Total CDU power
            if 'W_flow_CDUP_kW' in parsed_data['cdu']:
                metrics['total_cdu_power_kW'] = parsed_data['cdu']['W_flow_CDUP_kW'].sum(axis=1)
            
            # Average temperatures
            if 'T_prim_s_C' in parsed_data['cdu']:
                metrics['avg_prim_supply_temp_C'] = parsed_data['cdu']['T_prim_s_C'].mean(axis=1)
            if 'T_prim_r_C' in parsed_data['cdu']:
                metrics['avg_prim_return_temp_C'] = parsed_data['cdu']['T_prim_r_C'].mean(axis=1)
            
            # Temperature differentials
            if 'T_prim_r_C' in parsed_data['cdu'] and 'T_prim_s_C' in parsed_data['cdu']:
                metrics['avg_prim_delta_T'] = (
                    parsed_data['cdu']['T_prim_r_C'].mean(axis=1) - 
                    parsed_data['cdu']['T_prim_s_C'].mean(axis=1)
                )
            
            # Flow rates
            if 'V_flow_prim_GPM' in parsed_data['cdu']:
                metrics['total_prim_flow_GPM'] = parsed_data['cdu']['V_flow_prim_GPM'].sum(axis=1)
        
        # PUE
        if 'pue' in parsed_data:
            metrics['pue'] = parsed_data['pue']
        
        # External temperature
        if 'external_temp' in parsed_data:
            metrics['external_temp'] = parsed_data['external_temp'].iloc[:, 0]
        
        return metrics
    
    def analyze_cdu_performance(self, parsed_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze individual CDU performance"""
        
        if 'cdu' not in parsed_data:
            return {}
        
        cdu_analysis = {}
        
        # Identify underperforming or anomalous CDUs
        if 'W_flow_CDUP_kW' in parsed_data['cdu']:
            power_data = parsed_data['cdu']['W_flow_CDUP_kW']
            
            # Calculate statistics for each CDU
            cdu_stats = pd.DataFrame({
                'mean_power': power_data.mean(),
                'std_power': power_data.std(),
                'max_power': power_data.max(),
                'min_power': power_data.min()
            })
            
            # Identify outliers (CDUs with mean power > 2 std from overall mean)
            overall_mean = cdu_stats['mean_power'].mean()
            overall_std = cdu_stats['mean_power'].std()
            cdu_stats['is_outlier'] = abs(cdu_stats['mean_power'] - overall_mean) > 2 * overall_std
            
            cdu_analysis['power_stats'] = cdu_stats
        
        # Temperature efficiency analysis
        if 'T_prim_r_C' in parsed_data['cdu'] and 'T_prim_s_C' in parsed_data['cdu']:
            delta_t = parsed_data['cdu']['T_prim_r_C'] - parsed_data['cdu']['T_prim_s_C']
            
            cdu_analysis['delta_t_stats'] = pd.DataFrame({
                'mean_delta_t': delta_t.mean(),
                'std_delta_t': delta_t.std(),
                'efficiency_score': delta_t.mean() / delta_t.std()  # Higher is better
            })
        
        return cdu_analysis
    
    def generate_report(self, raw_data: pd.DataFrame, save_path: Optional[str] = None):
        """Generate comprehensive analysis report"""
        
        # Parse data
        parsed_data = self.parse_fmu_data(raw_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(parsed_data)
        
        # CDU analysis
        cdu_analysis = self.analyze_cdu_performance(parsed_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Center Cooling Analysis Report', fontsize=16)
        
        # 1. PUE over time
        if 'pue' in metrics:
            axes[0, 0].plot(metrics.index, metrics['pue'])
            axes[0, 0].set_title('Power Usage Effectiveness (PUE)')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('PUE')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total heat load
        if 'total_heat_load' in metrics:
            axes[0, 1].plot(metrics.index, metrics['total_heat_load'])
            axes[0, 1].set_title('Total Data Center Heat Load')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Heat Load')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CDU power distribution
        if 'power_stats' in cdu_analysis:
            cdu_analysis['power_stats']['mean_power'].plot(kind='bar', ax=axes[0, 2])
            axes[0, 2].set_title('Average CDU Power by Compute Block')
            axes[0, 2].set_xlabel('Compute Block')
            axes[0, 2].set_ylabel('Average Power (kW)')
        
        # 4. Temperature differential
        if 'avg_prim_delta_T' in metrics:
            axes[1, 0].plot(metrics.index, metrics['avg_prim_delta_T'])
            axes[1, 0].set_title('Average Primary Loop Temperature Differential')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Delta T (°K)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Flow rate vs power correlation
        if 'total_prim_flow_GPM' in metrics and 'total_cdu_power_kW' in metrics:
            axes[1, 1].scatter(metrics['total_prim_flow_GPM'], metrics['total_cdu_power_kW'], alpha=0.5)
            axes[1, 1].set_title('Flow Rate vs CDU Power')
            axes[1, 1].set_xlabel('Total Primary Flow (GPM)')
            axes[1, 1].set_ylabel('Total CDU Power (kW)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Heat map of CDU power over time (sample)
        if 'cdu' in parsed_data and 'W_flow_CDUP_kW' in parsed_data['cdu']:
            power_data = parsed_data['cdu']['W_flow_CDUP_kW']
            # Sample every 100th row for visualization
            sampled_data = power_data.iloc[::100, :].T
            sns.heatmap(sampled_data, ax=axes[1, 2], cmap='YlOrRd', cbar_kws={'label': 'Power (kW)'})
            axes[1, 2].set_title('CDU Power Heatmap (Sampled)')
            axes[1, 2].set_xlabel('Time (sampled)')
            axes[1, 2].set_ylabel('Compute Block')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Return analysis results
        return {
            'parsed_data': parsed_data,
            'metrics': metrics,
            'cdu_analysis': cdu_analysis
        }



def plot_cdu_power_minimal(df: pd.DataFrame, 
                          compute_blocks: list = [13, 16, 17, 38, 42],
                          figsize: tuple = (12, 6),
                          save_path: str = None, rolling_window: int = 180, range_limit: tuple = (0, 3600)):
    """
    Minimal visualization of CDU input power for specific compute blocks
    
    Parameters:
    - df: DataFrame with FMU output data
    - compute_blocks: List of compute block numbers to visualize
    - figsize: Figure size (width, height)
    - save_path: Optional path to save the figure
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    
    # Plot each CDU
    for idx, cb in enumerate(compute_blocks):
        col_name = f'CDU_{0 if cb < 10 else ""}{cb}'
        
        if col_name in df.columns:
            power_diff = df.loc[range_limit[0]:range_limit[1], col_name].rolling(window=rolling_window).mean()/1000

            # Plot with distinct color and style
            ax.plot(power_diff.index, power_diff, 
                   label=f'CDU {cb}', 
                   color=colors[idx % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time (Seconds)', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_title('CDU Input Power Comparison', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax



def plot_cdu_power_with_stats(df: pd.DataFrame,
                             compute_blocks: list = [13, 16, 17, 38, 42],
                             figsize: tuple = (14, 8), rolling_window: int = 180):
    """
    CDU power visualization with statistics panel
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main plot
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    # Statistics table
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.axis('tight')
    ax2.axis('off')
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Collect statistics
    stats_data = []
    power_diff = df.rolling(window=rolling_window).mean()

    # Plot each CDU
    for idx, cb in enumerate(compute_blocks):
        col_name = f'CDU_{0 if cb < 10 else ""}{cb}'

        if col_name in power_diff.columns:
            data = power_diff[col_name]/1000

            # Plot
            ax1.plot(power_diff.index, data,
                    label=f'CDU {cb}',
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    alpha=0.8)
            
            # Calculate statistics
            stats_data.append([
                f'CDU {cb}',
                f'{data.mean():.2f}',
                f'{data.std():.2f}',
                f'{data.min():.2f}',
                f'{data.max():.2f}'
            ])
    
    # Main plot styling
    ax1.set_xlabel('Time (Seconds)', fontsize=12)
    ax1.set_ylabel('Power (kW)', fontsize=12)
    ax1.set_title('CDU Input Power Comparison', fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add statistics table
    if stats_data:
        table = ax2.table(cellText=stats_data,
                         colLabels=['CDU', 'Mean (kW)', 'Std Dev', 'Min (kW)', 'Max (kW)'],
                         cellLoc='center',
                         loc='center',
                         colColours=['lightgray']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def cdu_output_metrics_visualization(cooling_df, n_cdus=None, selected_cbs: Optional[List[int]] = None):
    
    if n_cdus is None:
        n_cdus = 49
    
    # Extract CDU 1 related columns (every second column starting from index 1)
    cdu_1_cols = [c for c in cooling_df.columns if 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.' in c][1::2]
    
    # Determine which compute blocks to use
    if selected_cbs is None:
        cb_list = list(range(1, n_cdus + 1))
    else:
        cb_list = sorted(selected_cbs)  # Sort to ensure consistent ordering
    
    n_selected = len(cb_list)
    
    # Create a figure with subplots arranged in a grid
    n_rows = int(np.ceil(len(cdu_1_cols) / 3))  # 3 columns per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4*n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Use a color palette for better visualization based on selected CBs
    colors = plt.cm.viridis(np.linspace(0, 1, n_selected))
    
    # Process each CDU metric
    for idx, col in enumerate(cdu_1_cols):
        ax = axes[idx]
        name = col.split('.')[-1]
        
        # Create column names for selected compute blocks
        V_flow_prim = ['simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}'.format(cb, name) 
                      for cb in cb_list]
        
        # Get the data
        df_v_flow = cooling_df[V_flow_prim].copy()
        
        # Plot selected compute blocks
        for i, (col_name, cb_num) in enumerate(zip(df_v_flow.columns, cb_list)):
            # Determine which CBs to label
            label = None
            if i == 0:  # First selected CB
                label = f'CB{cb_num}'
            elif i == n_selected - 1:  # Last selected CB
                label = f'CB{cb_num}'
            elif n_selected > 5 and i % max(1, n_selected // 4) == 0:  # Show ~4 labels for many CBs
                label = f'CB{cb_num}'
            elif n_selected <= 5:  # Show all labels if few CBs
                label = f'CB{cb_num}'
            
            ax.plot(df_v_flow[col_name], color=colors[i], alpha=0.7, 
                   linewidth=2 if label is not None else 1, label=label)
        
        # Customize subplot
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add legend only if there are labeled lines
        if any([line.get_label()[0] != '_' for line in ax.get_lines()]):
            ax.legend(loc='best', fontsize=6, ncol=2)
    
    # Hide empty subplots if total subplots > number of metrics
    for idx in range(len(cdu_1_cols), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout and add main title
    if selected_cbs is None:
        title = f'CDU Metrics Across All {n_cdus} Compute Blocks'
    else:
        title = f'CDU Metrics for Selected Compute Blocks: {", ".join(map(str, cb_list))}'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def cdu_output_metrics_heatmap(cooling_df, num_cdus=None):
    if num_cdus is None:
        num_cdus = 49
    cdu_1_cols = [c for c in cooling_df.columns if 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.' in c][1::2]
    
    n_rows = int(np.ceil(len(cdu_1_cols) / 3))

    fig, axes3 = plt.subplots(n_rows, 3, figsize=(18, 4*n_rows))
    axes3 = axes3.flatten()

    for idx, col in enumerate(cdu_1_cols):
        ax = axes3[idx]
        name = col.split('.')[-1]
        
        V_flow_prim = ['simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}'.format(i, name) 
                    for i in range(1, 50)]
        df_v_flow = cooling_df[V_flow_prim].copy()
        
        # Create heatmap
        im = ax.imshow(df_v_flow.T, aspect='auto', cmap='coolwarm')
        
        ax.set_title(f'{name} Heatmap', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Index', fontsize=8)
        ax.set_ylabel('Compute Block', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=6)
        
        # Set y-tick labels for every 5th compute block
        ax.set_yticks(range(0, 49, 5))
        ax.set_yticklabels([f'CB{i+1}' for i in range(0, 49, 5)], fontsize=7)
        ax.tick_params(axis='x', labelsize=7)

    # Hide empty subplots
    for idx in range(len(cdu_1_cols), len(axes3)):
        axes3[idx].set_visible(False)

    plt.suptitle('CDU Metrics Heatmap - All Compute Blocks Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

def plot_cdu_output_metrics(df: pd.DataFrame, 
                            selected_cbs: Optional[List[int]] = None,
                            num_cdus: int = 49):
    """Visualize CDU output metrics"""
    
    if selected_cbs is None:
        selected_cbs = list(range(1, num_cdus + 1))
    
    # Get available output variables from first CDU
    sample_cols = [c for c in df.columns 
                  if f'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.' in c]
    
    # Extract variable names
    output_vars = [col.split('.')[-1] for col in sample_cols]
    output_vars = [v for v in output_vars if v != 'time']  # Exclude time
    
    n_vars = len(output_vars)
    n_rows = int(np.ceil(n_vars / 3))
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_vars == 1 else axes
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_cbs)))
    
    for idx, var in enumerate(output_vars):
        ax = axes[idx]
        
        for i, cb in enumerate(selected_cbs):
            col_name = f'simulator[1].datacenter[1].computeBlock[{cb}].cdu[1].summary.{var}'
            
            if col_name in df.columns:
                ax.plot(df.index, df[col_name].values, 
                       label=f'CB {cb}', color=colors[i], alpha=0.7)
        
        ax.set_title(f'{var}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        if idx == 0:  # Add legend only to first plot
            ax.legend(fontsize=7, ncol=2)
    
    # Hide empty subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)
    
    title = f'CDU Metrics for Compute Blocks: {", ".join(map(str, selected_cbs))}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_cdu_means_with_dc_metrics(df, compute_blocks=None, num_cdus=49):
    """Plot mean CDU values and data center metrics"""
    
    if compute_blocks is None:
        compute_blocks = [13, 16, 17, 38, 42]
    
    # CDU parameters to plot
    cdu_params = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW', 
                  'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C']
    
    # Calculate CDU means
    cdu_means = {}
    for param in cdu_params:
        param_cols = [f'simulator[1].datacenter[1].computeBlock[{cb}].cdu[1].summary.{param}'
                     for cb in compute_blocks]
        param_cols = [col for col in param_cols if col in df.columns]
        
        if param_cols:
            cdu_means[param] = df[param_cols].mean(axis=1)
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot CDU means
    for i, (param, data) in enumerate(cdu_means.items()):
        if i < len(axes):
            axes[i].plot(data.index, data.values)
            axes[i].set_title(f'Mean {param}')
            axes[i].set_xlabel('Time')
            axes[i].grid(True, alpha=0.3)
    
    # Plot DC metrics if available
    dc_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
    if dc_flow_col in df.columns and len(cdu_means) < len(axes):
        axes[len(cdu_means)].plot(df.index, df[dc_flow_col].values)
        axes[len(cdu_means)].set_title('DC Total Flow')
        axes[len(cdu_means)].set_xlabel('Time')
        axes[len(cdu_means)].grid(True, alpha=0.3)
    
    if 'pue' in df.columns and len(cdu_means) + 1 < len(axes):
        axes[len(cdu_means) + 1].plot(df.index, df['pue'].values)
        axes[len(cdu_means) + 1].set_title('PUE')
        axes[len(cdu_means) + 1].set_xlabel('Time')
        axes[len(cdu_means) + 1].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(cdu_means) + 2, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('CDU Mean Values & Data Center Metrics', fontsize=14, y=1.02)
    return fig

def plot_cdu_variability(df, compute_blocks=[13, 16, 17, 38, 42]):
    """Plot CDU parameter variability (std dev) across compute blocks."""
    params = ['V_flow_prim_GPM', 'W_flow_CDUP_kW', 'T_prim_s_C', 'T_sec_s_C']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        cols = [f'simulator[1].datacenter[1].computeBlock[{b}].cdu[1].summary.{param}' 
                for b in compute_blocks]
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            std_dev = df[valid_cols].std(axis=1)
            axes[i].plot(df.index, std_dev, 'g-', linewidth=2)
            axes[i].fill_between(df.index, 0, std_dev, alpha=0.3, color='green')
            axes[i].set_title(f'{param} Variability (σ)', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('CDU Parameter Variability Across Blocks', fontsize=14, y=1.02)
    return fig

