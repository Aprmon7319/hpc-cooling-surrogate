import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
from functools import partial
import dask.dataframe as dd
import warnings
warnings.filterwarnings('ignore')


def _compute_persistence_for_pair(args):
    """Helper function for parallel persistence computation."""
    pair_idx, inlet_series, outlet_series, window_size = args
    
    if len(inlet_series) > window_size:
        inlet_corr, _ = pearsonr(inlet_series[:-window_size], inlet_series[window_size:])
        outlet_corr, _ = pearsonr(outlet_series[:-window_size], outlet_series[window_size:])
        return pair_idx, inlet_corr, outlet_corr
    else:
        return pair_idx, np.nan, np.nan


def _analyze_hot_zone_for_cdu(args):
    """Helper function for parallel hot zone analysis."""
    cdu_idx, Q_flow_cdu, T_outlet, n_cdus, threshold_percentile = args
    
    if len(Q_flow_cdu) == 0:
        return cdu_idx, None
    
    threshold = np.percentile(Q_flow_cdu, threshold_percentile)
    high_load_mask = Q_flow_cdu > threshold
    
    if np.sum(high_load_mask) == 0:
        return cdu_idx, None
    
    avg_temp_high_load = np.mean(T_outlet[high_load_mask, cdu_idx])
    
    neighbor_temps = []
    if cdu_idx > 0 and T_outlet.shape[1] > cdu_idx - 1:
        neighbor_temps.append(np.mean(T_outlet[high_load_mask, cdu_idx-1]))
    if cdu_idx < n_cdus - 1 and T_outlet.shape[1] > cdu_idx + 1:
        neighbor_temps.append(np.mean(T_outlet[high_load_mask, cdu_idx+1]))
    
    avg_neighbor_temp = np.mean(neighbor_temps) if neighbor_temps else avg_temp_high_load
    temp_elevation = avg_temp_high_load - avg_neighbor_temp
    
    return cdu_idx, {
        'high_load_count': np.sum(high_load_mask),
        'avg_temp_high_load': avg_temp_high_load,
        'avg_neighbor_temp': avg_neighbor_temp,
        'temp_elevation': temp_elevation,
        'high_load_indices': np.where(high_load_mask)[0]
    }


class TemperatureGradientAnalyzer:
    """Analyze temperature gradients and spatial patterns across CDUs."""

    def __init__(self, data_path: str, output_dir: str = "results/temperature_gradient", n_cdus: int = None, n_workers: int = None):
        """
        Initialize the analyzer.

        Args:
            data_path: Path to the dataset (parquet or zarr)
            output_dir: Directory to save results
            n_cdus: Expected number of CDUs. If None, tries to infer from data.
            n_workers: Number of parallel workers. If None, uses cpu_count() - 1.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.n_cdus = n_cdus
        self.timesteps = None
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

    def load_data(self, sample_size: Optional[int] = None):
        """Load dataset and extract temperature-related features using Dask."""
        print(f"Loading data from {self.data_path}...")
        print(f"Using {self.n_workers} workers for parallel processing")

        if self.data_path.suffix == '.parquet':
            # Use Dask for parallel loading
            ddf = dd.read_parquet(self.data_path)
            df = ddf.compute()
        else:
            raise ValueError("Unsupported data format. Use .parquet")

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        # If n_cdus was not provided at init, try to infer
        if self.n_cdus is None:
            q_flow_cols_in_df = [col for col in df.columns if 'cabinet_1_sources_Q_flow_total' in col]
            if q_flow_cols_in_df:
                block_indices = [int(col.split('computeBlock_')[1].split('_')[0]) for col in q_flow_cols_in_df]
                self.n_cdus = max(block_indices) if block_indices else 0
            else:
                raise ValueError("n_cdus not provided at initialization and could not be inferred from data. "
                                 "Please pass n_cdus to the constructor or ensure Q_flow columns exist.")

        if self.n_cdus == 0:
            raise ValueError("No CDUs found or inferred. Check data and n_cdus configuration.")

        print(f"Using {self.n_cdus} CDUs for column identification.")

        # Construct column names based on self.n_cdus
        # These patterns are derived from the DeepONet script's data loading
        t_air_cols_expected = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air'
                               for i in range(1, self.n_cdus + 1)]
        q_flow_cols_expected = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total'
                                for i in range(1, self.n_cdus + 1)]
        # Assuming T_prim_r_C as the primary outlet temperature based on DeepONet outputs
        t_outlet_cols_expected = [f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_prim_r_C'
                                  for i in range(1, self.n_cdus + 1)]


        # Filter for existing columns to avoid KeyError
        t_air_cols = [col for col in t_air_cols_expected if col in df.columns]
        t_outlet_cols = [col for col in t_outlet_cols_expected if col in df.columns]
        q_flow_cols = [col for col in q_flow_cols_expected if col in df.columns]


        print(f"Found {len(t_air_cols)} T_inlet columns (expected {self.n_cdus})")
        print(f"Found {len(t_outlet_cols)} T_outlet columns (expected {self.n_cdus})")
        print(f"Found {len(q_flow_cols)} Q_flow columns (expected {self.n_cdus})")


        # Check if enough columns were found
        if len(t_air_cols) != self.n_cdus or len(t_outlet_cols) != self.n_cdus or len(q_flow_cols) != self.n_cdus:
            # If not all expected columns are found, it's a critical error for this analysis
            raise ValueError(f"Mismatch in expected vs. found CDU columns. "
                             f"Expected {self.n_cdus} CDUs, but found: "
                             f"{len(t_air_cols)} inlet, {len(t_outlet_cols)} outlet, {len(q_flow_cols)} Q_flow.")
                             
        if not t_air_cols or not t_outlet_cols:
            raise ValueError(f"Could not find any T_inlet or T_outlet columns. Check column naming conventions in data.")


        # Store data
        self.data = {
            'T_inlet': df[t_air_cols].values - 273.15,
            'T_outlet': df[t_outlet_cols].values,
            'Q_flow': df[q_flow_cols].values if q_flow_cols else None,
            'inlet_cols': t_air_cols,
            'outlet_cols': t_outlet_cols
        }

        self.timesteps = len(df)
        print(f"Loaded {self.timesteps} timesteps")
        print(f"Temperature range - Inlet: [{self.data['T_inlet'].min():.2f}, {self.data['T_inlet'].max():.2f}]")
        print(f"Temperature range - Outlet: [{self.data['T_outlet'].min():.2f}, {self.data['T_outlet'].max():.2f}]")

    def calculate_spatial_gradients(self, temperature_data: np.ndarray) -> np.ndarray:
        """
        Calculate temperature gradients between adjacent CDUs.

        Args:
            temperature_data: Array of shape (timesteps, n_cdus)

        Returns:
            Gradient array of shape (timesteps, n_cdus-1)
            Gradient[t, i] = (T[t, i+1] - T[t, i]) / Δposition
            Assuming Δposition = 1 (unit distance between CDUs)
        """
        # Calculate differences between adjacent CDUs
        if temperature_data.shape[1] < 2: # Check if there are at least 2 CDUs to calculate a gradient
            return np.array([]) # Return empty array if no gradients can be calculated
        gradients = np.diff(temperature_data, axis=1)
        return gradients

    def calculate_gradient_metrics(self, gradients: np.ndarray) -> Dict:
        """
        Calculate key gradient metrics.

        Args:
            gradients: Gradient array from calculate_spatial_gradients

        Returns:
            Dictionary with gradient statistics
        """
        if gradients.size == 0:
            # Handle case where no gradients could be calculated (e.g., n_cdus < 2)
            return {
                'mean_magnitude': 0.0,
                'max_magnitude': 0.0,
                'std_magnitude': 0.0,
                'mean_gradient': np.array([]),
                'std_gradient': np.array([]),
                'temporal_variance': np.array([]),
                'spatial_variance': np.array([])
            }

        metrics = {
            'mean_magnitude': np.mean(np.abs(gradients)),
            'max_magnitude': np.max(np.abs(gradients)),
            'std_magnitude': np.std(np.abs(gradients)),
            'mean_gradient': np.mean(gradients, axis=0),  # Per CDU pair
            'std_gradient': np.std(gradients, axis=0),
            'temporal_variance': np.var(gradients, axis=0),  # How much each pair varies
            'spatial_variance': np.var(gradients, axis=1)   # How much gradients vary across space
        }

        return metrics

    def identify_hot_zones(self, threshold_percentile: float = 90) -> Dict:
        """
        Identify high-load CDUs that create "hot zones" using parallel processing.

        Args:
            threshold_percentile: Percentile to define high-load events

        Returns:
            Dictionary with hot zone information
        """
        print("\nIdentifying hot zones (parallelized)...")

        if self.data['Q_flow'] is None or self.data['Q_flow'].shape[1] == 0:
            print("Q_flow data not available or empty. Skipping hot zone analysis.")
            return {}
        if self.n_cdus < 2:
            print("Less than 2 CDUs available. Skipping hot zone analysis as neighbor comparison is not meaningful.")
            return {}


        Q_flow = self.data['Q_flow']
        T_outlet = self.data['T_outlet']

        # Prepare arguments for parallel processing
        args_list = [
            (cdu_idx, Q_flow[:, cdu_idx], T_outlet, self.n_cdus, threshold_percentile)
            for cdu_idx in range(self.n_cdus) if cdu_idx < Q_flow.shape[1]
        ]

        # Parallel hot zone analysis
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_analyze_hot_zone_for_cdu, args_list)

        hot_zones = {}
        for cdu_idx, info in results:
            if info is not None:
                hot_zones[cdu_idx] = info

        # Sort by temperature elevation
        if hot_zones:
            sorted_zones = sorted(hot_zones.items(), key=lambda x: x[1]['temp_elevation'], reverse=True)
            print(f"Identified {len(hot_zones)} potential hot zones")
            print("\nTop 5 hot zones by temperature elevation:")
            for cdu_idx, info in sorted_zones[:5]:
                print(f"  CDU {cdu_idx}: ΔT = {info['temp_elevation']:.2f}°C")
        else:
            print("No hot zones identified.")

        return hot_zones

    def analyze_gradient_persistence(self, window_size: int = 10) -> Dict:
        """
        Analyze how long temperature gradients persist using parallel processing.

        Args:
            window_size: Size of rolling window for persistence analysis

        Returns:
            Dictionary with persistence metrics
        """
        print("\nAnalyzing gradient persistence (parallelized)...")

        if self.n_cdus < 2:
            print("Less than 2 CDUs available. Skipping gradient persistence analysis.")
            return {
                'inlet_persistence': np.array([]),
                'outlet_persistence': np.array([]),
                'mean_inlet_persistence': 0.0,
                'mean_outlet_persistence': 0.0
            }

        inlet_gradients = self.calculate_spatial_gradients(self.data['T_inlet'])
        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])

        if inlet_gradients.size == 0 or outlet_gradients.size == 0:
             print("No gradients available for persistence analysis.")
             return {
                'inlet_persistence': np.array([]),
                'outlet_persistence': np.array([]),
                'mean_inlet_persistence': 0.0,
                'mean_outlet_persistence': 0.0
            }

        # Prepare arguments for parallel processing
        args_list = [
            (pair_idx, inlet_gradients[:, pair_idx], outlet_gradients[:, pair_idx], window_size)
            for pair_idx in range(inlet_gradients.shape[1])
        ]

        # Parallel persistence computation
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_compute_persistence_for_pair, args_list)

        inlet_persistence = [np.nan] * inlet_gradients.shape[1]
        outlet_persistence = [np.nan] * outlet_gradients.shape[1]
        
        for pair_idx, inlet_corr, outlet_corr in results:
            inlet_persistence[pair_idx] = inlet_corr
            outlet_persistence[pair_idx] = outlet_corr

        mean_inlet_persistence = np.nanmean(inlet_persistence) if inlet_persistence else 0.0
        mean_outlet_persistence = np.nanmean(outlet_persistence) if outlet_persistence else 0.0

        return {
            'inlet_persistence': np.array(inlet_persistence),
            'outlet_persistence': np.array(outlet_persistence),
            'mean_inlet_persistence': mean_inlet_persistence,
            'mean_outlet_persistence': mean_outlet_persistence
        }

    def analyze_asymmetry(self) -> Dict:
        """
        Analyze if temperature effects propagate differently upstream vs downstream.

        Returns:
            Dictionary with asymmetry metrics
        """
        print("\nAnalyzing propagation asymmetry...")

        if self.n_cdus < 2:
            print("Less than 2 CDUs available. Skipping asymmetry analysis.")
            return {
                'positive_count': 0,
                'negative_count': 0,
                'positive_mean': 0.0,
                'negative_mean': 0.0,
                'asymmetry_ratio': 0.0
            }

        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])

        if outlet_gradients.size == 0:
            print("No outlet gradients available for asymmetry analysis.")
            return {
                'positive_count': 0,
                'negative_count': 0,
                'positive_mean': 0.0,
                'negative_mean': 0.0,
                'asymmetry_ratio': 0.0
            }

        # Positive gradients (heating in forward direction)
        positive_gradients = outlet_gradients[outlet_gradients > 0]
        negative_gradients = outlet_gradients[outlet_gradients < 0]

        asymmetry = {
            'positive_count': len(positive_gradients),
            'negative_count': len(negative_gradients),
            'positive_mean': np.mean(positive_gradients) if len(positive_gradients) > 0 else 0,
            'negative_mean': np.mean(negative_gradients) if len(negative_gradients) > 0 else 0,
            'asymmetry_ratio': len(positive_gradients) / len(negative_gradients) if len(negative_gradients) > 0 else np.inf
        }

        print(f"Positive gradients: {asymmetry['positive_count']} (mean: {asymmetry['positive_mean']:.3f}°C)")
        print(f"Negative gradients: {asymmetry['negative_count']} (mean: {asymmetry['negative_mean']:.3f}°C)")

        return asymmetry

    def plot_temperature_profile(self, timesteps_to_plot: Optional[List[int]] = None,
                                 smooth: bool = True):
        """
        Plot temperature profiles across CDUs at different timesteps.

        Args:
            timesteps_to_plot: Specific timesteps to visualize
            smooth: Apply smoothing for better visualization
        """
        if self.n_cdus < 1:
            print("Cannot plot temperature profile, no CDUs found.")
            return

        if timesteps_to_plot is None:
            # Select interesting timesteps: min, max, median load
            if self.data['Q_flow'] is not None and self.data['Q_flow'].size > 0:
                total_load = np.sum(self.data['Q_flow'], axis=1)
                # Ensure total_load has at least one element
                if len(total_load) > 0:
                    timesteps_to_plot = [
                        np.argmin(total_load),
                        np.argmax(total_load),
                        np.argsort(total_load)[len(total_load)//2],
                        len(total_load)//4,
                        3*len(total_load)//4
                    ]
                    timesteps_to_plot = [ts for ts in timesteps_to_plot if ts < self.timesteps] # Ensure valid timesteps
                else:
                    timesteps_to_plot = [0] # Default to first timestep if no load data
            else:
                timesteps_to_plot = [0, self.timesteps//4, self.timesteps//2,
                                    3*self.timesteps//4, self.timesteps-1]
                timesteps_to_plot = [ts for ts in timesteps_to_plot if ts >=0 and ts < self.timesteps] # Ensure valid timesteps
        
        if not timesteps_to_plot:
            print("No valid timesteps to plot temperature profiles.")
            return


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        cdu_indices = np.arange(self.n_cdus)
        # Handle case where len(timesteps_to_plot) might be 0 after filtering
        if len(timesteps_to_plot) == 0:
            print("No valid timesteps to plot.")
            plt.close(fig)
            return
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_to_plot)))

        for idx, timestep in enumerate(timesteps_to_plot):
            if timestep >= self.timesteps: # Already filtered, but good defensive check
                continue

            inlet_temps = self.data['T_inlet'][timestep, :]
            outlet_temps = self.data['T_outlet'][timestep, :]

            if smooth:
                inlet_temps = gaussian_filter1d(inlet_temps, sigma=1.0)
                outlet_temps = gaussian_filter1d(outlet_temps, sigma=1.0)

            load_label = ""
            if self.data['Q_flow'] is not None and self.data['Q_flow'].size > 0 and timestep < self.data['Q_flow'].shape[0]:
                total_load = np.sum(self.data['Q_flow'][timestep, :])
                load_label = f" (Load: {total_load/1e6:.2f} MW)"

            ax1.plot(cdu_indices, inlet_temps, marker='o', linewidth=2,
                    color=colors[idx], label=f't={timestep}{load_label}', alpha=0.7)
            ax2.plot(cdu_indices, outlet_temps, marker='s', linewidth=2,
                    color=colors[idx], label=f't={timestep}{load_label}', alpha=0.7)

        ax1.set_xlabel('CDU Index', fontsize=12)
        ax1.set_ylabel('Inlet Temperature (°C)', fontsize=12)
        ax1.set_title('Inlet Temperature Profile Across CDUs', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('CDU Index', fontsize=12)
        ax2.set_ylabel('Outlet Temperature (°C)', fontsize=12)
        ax2.set_title('Outlet Temperature Profile Across CDUs', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'temperature_profiles.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved temperature profiles to {self.output_dir}")

    def plot_gradient_vector_field(self, timestep: int = None):
        """
        Plot gradient vector field showing heat flow direction.

        Args:
            timestep: Specific timestep to visualize (default: max gradient timestep)
        """
        if self.n_cdus < 2:
            print("Cannot plot gradient vector field, less than 2 CDUs found.")
            return

        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])
        if outlet_gradients.size == 0:
            print("No outlet gradients to plot vector field.")
            return


        if timestep is None:
            # Find timestep with maximum gradient activity
            gradient_magnitudes = np.sum(np.abs(outlet_gradients), axis=1)
            if gradient_magnitudes.size == 0:
                print("No gradient magnitudes to determine a timestep for vector field.")
                return
            timestep = np.argmax(gradient_magnitudes)
        
        if timestep >= self.timesteps:
            print(f"Selected timestep {timestep} is out of bounds for data with {self.timesteps} timesteps.")
            return


        fig, ax = plt.subplots(figsize=(16, 6))

        cdu_indices = np.arange(self.n_cdus)
        temperatures = self.data['T_outlet'][timestep, :]

        # Plot temperature as background
        ax.plot(cdu_indices, temperatures, 'o-', color='gray',
               linewidth=2, markersize=8, label='Temperature', alpha=0.6)

        # Plot gradient vectors
        # Ensure that outlet_gradients[timestep] is not empty
        if outlet_gradients.shape[1] > 0:
            max_abs_gradient = np.max(np.abs(outlet_gradients[timestep]))
            if max_abs_gradient == 0: # Avoid division by zero if all gradients are zero
                max_abs_gradient = 1.0 # Set to 1 to allow plotting, but alpha will be 0

            for i in range(outlet_gradients.shape[1]): # Iterate over CDU pairs
                gradient = outlet_gradients[timestep, i]
                x_start = i + 0.5 # Position between CDU i and i+1

                # Arrow properties based on gradient magnitude and direction
                color = 'red' if gradient > 0 else 'blue'
                alpha = min(abs(gradient) / max_abs_gradient, 1.0)
                arrow_length = gradient * 0.1  # Scale for visualization

                # Check if there is a next temperature point for the arrow end
                if i + 1 < self.n_cdus:
                    ax.arrow(x_start, temperatures[i], 0.0, arrow_length, # arrow from i to i+1
                            head_width=0.3, head_length=abs(arrow_length)*0.3,
                            fc=color, ec=color, alpha=alpha, linewidth=2)
                else: # For the last CDU, point the arrow from the second to last to the last
                     ax.arrow(x_start, temperatures[i], 0.0, arrow_length, # arrow from i to i+1
                            head_width=0.3, head_length=abs(arrow_length)*0.3,
                            fc=color, ec=color, alpha=alpha, linewidth=2)


        ax.set_xlabel('CDU Index', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'Temperature Gradient Vector Field (t={timestep})\n'
                    f'Red: Heating (positive gradient), Blue: Cooling (negative gradient)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f'gradient_vector_field_t{timestep}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved gradient vector field to {self.output_dir}")

    def plot_temperature_difference_heatmap(self, max_timesteps: int = 500):
        """
        Create heatmap of temperature differences between adjacent CDUs over time.

        Args:
            max_timesteps: Maximum number of timesteps to plot (for readability)
        """
        if self.n_cdus < 2:
            print("Cannot plot temperature difference heatmap, less than 2 CDUs found.")
            return

        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])

        if outlet_gradients.size == 0:
            print("No outlet gradients to plot heatmap.")
            return

        # Limit timesteps for visualization
        if self.timesteps > max_timesteps:
            step = self.timesteps // max_timesteps
            gradients_plot = outlet_gradients[::step, :]
            time_indices = np.arange(0, self.timesteps, step)
        else:
            gradients_plot = outlet_gradients
            time_indices = np.arange(self.timesteps)
        
        if gradients_plot.shape[0] == 0 or gradients_plot.shape[1] == 0:
            print("Gradients plot array is empty after sampling. Skipping heatmap.")
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        # Ensure vmin and vmax are not zero to avoid issues with imshow
        abs_max_grad = np.max(np.abs(gradients_plot))
        if abs_max_grad > 0:
            vmin_val = -abs_max_grad
            vmax_val = abs_max_grad
        else: # All gradients are zero, set a default range to show a flat color
            vmin_val = -1
            vmax_val = 1


        im = ax.imshow(gradients_plot, aspect='auto', cmap='RdBu_r',
                      interpolation='nearest', vmin=vmin_val, vmax=vmax_val)

        ax.set_xlabel('CDU Pair (i → i+1)', fontsize=12)
        ax.set_ylabel('Timestep', fontsize=12)
        ax.set_title('Temperature Difference Heatmap (Adjacent CDUs)\n'
                    'Red: Downstream hotter, Blue: Upstream hotter',
                    fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature Difference (°C)', fontsize=12)

        # Set tick labels
        if len(time_indices) > 20:
            tick_step = len(time_indices) // 10
            ax.set_yticks(np.arange(0, len(time_indices), tick_step))
            ax.set_yticklabels(time_indices[::tick_step])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'temperature_difference_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved temperature difference heatmap to {self.output_dir}")

    def plot_gradient_statistics(self):
        """Plot comprehensive gradient statistics."""
        if self.n_cdus < 2:
            print("Cannot plot gradient statistics, less than 2 CDUs found.")
            return

        inlet_gradients = self.calculate_spatial_gradients(self.data['T_inlet'])
        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])
        
        if inlet_gradients.size == 0 or outlet_gradients.size == 0:
            print("No gradients available for plotting statistics.")
            return


        inlet_metrics = self.calculate_gradient_metrics(inlet_gradients)
        outlet_metrics = self.calculate_gradient_metrics(outlet_gradients)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean gradient per CDU pair
        ax = axes[0, 0]
        x = np.arange(len(inlet_metrics['mean_gradient']))
        width = 0.35
        ax.bar(x - width/2, inlet_metrics['mean_gradient'], width,
              label='Inlet', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, outlet_metrics['mean_gradient'], width,
              label='Outlet', alpha=0.8, color='coral')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('CDU Pair Index')
        ax.set_ylabel('Mean Temperature Gradient (°C)')
        ax.set_title('Average Gradient by Location')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Gradient magnitude distribution
        ax = axes[0, 1]
        ax.hist(np.abs(inlet_gradients).flatten(), bins=50, alpha=0.6,
               label='Inlet', color='steelblue', density=True)
        ax.hist(np.abs(outlet_gradients).flatten(), bins=50, alpha=0.6,
               label='Outlet', color='coral', density=True)
        ax.set_xlabel('Gradient Magnitude (°C)')
        ax.set_ylabel('Density')
        ax.set_title('Gradient Magnitude Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Temporal variance
        ax = axes[1, 0]
        ax.plot(inlet_metrics['temporal_variance'], marker='o',
               label='Inlet', linewidth=2, color='steelblue')
        ax.plot(outlet_metrics['temporal_variance'], marker='s',
               label='Outlet', linewidth=2, color='coral')
        ax.set_xlabel('CDU Pair Index')
        ax.set_ylabel('Temporal Variance')
        ax.set_title('Gradient Stability Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Spatial variance over time
        ax = axes[1, 1]
        time_sample = min(500, len(inlet_metrics['spatial_variance']))
        t = np.arange(time_sample)
        ax.plot(t, inlet_metrics['spatial_variance'][:time_sample],
               label='Inlet', linewidth=1.5, alpha=0.7, color='steelblue')
        ax.plot(t, outlet_metrics['spatial_variance'][:time_sample],
               label='Outlet', linewidth=1.5, alpha=0.7, color='coral')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Spatial Variance')
        ax.set_title('Spatial Gradient Variation Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Temperature Gradient Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gradient_statistics.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved gradient statistics to {self.output_dir}")

    def generate_summary_report(self) -> str:
        """Generate text summary of gradient analysis."""
        if self.n_cdus < 1:
            return "No CDUs to analyze. Summary report not generated."

        inlet_gradients = self.calculate_spatial_gradients(self.data['T_inlet'])
        outlet_gradients = self.calculate_spatial_gradients(self.data['T_outlet'])

        inlet_metrics = self.calculate_gradient_metrics(inlet_gradients)
        outlet_metrics = self.calculate_gradient_metrics(outlet_gradients)

        persistence = self.analyze_gradient_persistence()
        asymmetry = self.analyze_asymmetry()
        hot_zones = self.identify_hot_zones()

        report = []
        report.append("="*80)
        report.append("TEMPERATURE GRADIENT ANALYSIS SUMMARY")
        report.append("="*80)
        report.append("")

        if self.n_cdus < 2:
            report.append(f"Analysis limited as only {self.n_cdus} CDU(s) found. Gradient calculations require at least 2 CDUs.")
            report.append("")

        report.append("### GRADIENT MAGNITUDE ###")
        if inlet_gradients.size > 0:
            report.append(f"Inlet  - Mean: {inlet_metrics['mean_magnitude']:.3f}°C, "
                        f"Max: {inlet_metrics['max_magnitude']:.3f}°C, "
                        f"Std: {inlet_metrics['std_magnitude']:.3f}°C")
        else:
            report.append("Inlet gradients not calculable (less than 2 CDUs).")

        if outlet_gradients.size > 0:
            report.append(f"Outlet - Mean: {outlet_metrics['mean_magnitude']:.3f}°C, "
                        f"Max: {outlet_metrics['max_magnitude']:.3f}°C, "
                        f"Std: {outlet_metrics['std_magnitude']:.3f}°C")
        else:
            report.append("Outlet gradients not calculable (less than 2 CDUs).")
        report.append("")

        report.append("### GRADIENT PERSISTENCE ###")
        if persistence['inlet_persistence'].size > 0:
            report.append(f"Inlet  - Mean persistence: {persistence['mean_inlet_persistence']:.3f}")
        else:
            report.append("Inlet persistence not calculable.")
        if persistence['outlet_persistence'].size > 0:
            report.append(f"Outlet - Mean persistence: {persistence['mean_outlet_persistence']:.3f}")
        else:
            report.append("Outlet persistence not calculable.")
        report.append("")

        report.append("### ASYMMETRY ###")
        if asymmetry['positive_count'] > 0 or asymmetry['negative_count'] > 0:
            report.append(f"Positive gradients: {asymmetry['positive_count']} "
                        f"(mean: {asymmetry['positive_mean']:.3f}°C)")
            report.append(f"Negative gradients: {asymmetry['negative_count']} "
                        f"(mean: {asymmetry['negative_mean']:.3f}°C)")
            report.append(f"Asymmetry ratio: {asymmetry['asymmetry_ratio']:.2f}")
        else:
            report.append("Asymmetry not calculable (no gradients).")
        report.append("")

        if hot_zones:
            report.append("### HOT ZONES ###")
            sorted_zones = sorted(hot_zones.items(),
                                 key=lambda x: x[1]['temp_elevation'], reverse=True)
            report.append(f"Total hot zones identified: {len(hot_zones)}")
            report.append("\nTop 5 hot zones:")
            for cdu_idx, info in sorted_zones[:5]:
                report.append(f"  CDU {cdu_idx}: Temperature elevation = {info['temp_elevation']:.2f}°C, "
                            f"High-load events = {info['high_load_count']}")
        else:
            report.append("No hot zones identified or Q_flow data unavailable.")

        report.append("")
        report.append("="*80)

        report_text = "\n".join(report)

        # Save to file
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def run_full_analysis(self, sample_size: Optional[int] = None):
        """Run complete temperature gradient analysis."""
        print("="*80)
        print("TEMPERATURE GRADIENT ANALYSIS")
        print("="*80)

        # Load data
        self.load_data(sample_size)

        # Generate all visualizations
        print("\n[1/5] Generating temperature profiles...")
        self.plot_temperature_profile()

        print("\n[2/5] Generating gradient vector field...")
        self.plot_gradient_vector_field()

        print("\n[3/5] Generating temperature difference heatmap...")
        self.plot_temperature_difference_heatmap()

        print("\n[4/5] Generating gradient statistics...")
        self.plot_gradient_statistics()

        print("\n[5/5] Generating summary report...")
        self.generate_summary_report()

        print("\n" + "="*80)
        print("Analysis complete! Results saved to:", self.output_dir)
        print("="*80)