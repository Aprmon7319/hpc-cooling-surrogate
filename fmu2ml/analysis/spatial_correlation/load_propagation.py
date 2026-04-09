import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
from functools import partial
import dask.dataframe as dd
import warnings
warnings.filterwarnings('ignore')


def _compute_lagged_correlation_pair(args):
    """Helper function for parallel lagged correlation computation."""
    source_idx, target_idx, source_signal, target_signal, max_lag = args
    
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros_like(lags, dtype=float)
    
    for i, lag in enumerate(lags):
        if lag < 0:
            corr, _ = pearsonr(source_signal[:lag], target_signal[-lag:])
        elif lag > 0:
            corr, _ = pearsonr(source_signal[lag:], target_signal[:-lag])
        else:
            corr, _ = pearsonr(source_signal, target_signal)
        correlations[i] = corr
    
    return source_idx, target_idx, lags, correlations


def _compute_propagation_for_target(args):
    """Helper function for parallel propagation speed analysis."""
    source_cdu, target_cdu, Q_flow, max_lag = args
    
    if target_cdu == source_cdu:
        return None
    
    source_signal = Q_flow[:, source_cdu]
    target_signal = Q_flow[:, target_cdu]
    
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros_like(lags, dtype=float)
    
    for i, lag in enumerate(lags):
        if lag < 0:
            corr, _ = pearsonr(source_signal[:lag], target_signal[-lag:])
        elif lag > 0:
            corr, _ = pearsonr(source_signal[lag:], target_signal[:-lag])
        else:
            corr, _ = pearsonr(source_signal, target_signal)
        correlations[i] = corr
    
    optimal_idx = np.argmax(np.abs(correlations))
    optimal_lag = lags[optimal_idx]
    max_corr = correlations[optimal_idx]
    distance = abs(target_cdu - source_cdu)
    attenuation = 1 - abs(max_corr)
    
    return target_cdu, distance, optimal_lag, max_corr, attenuation


class LoadPropagationAnalyzer:
    """Analyze how load changes in one CDU propagate to others."""

    def __init__(self, data_path: str, output_dir: str = "results/load_propagation", n_workers: int = None):
        """
        Initialize the analyzer.

        Args:
            data_path: Path to the dataset (parquet or zarr)
            output_dir: Directory to save results
            n_workers: Number of parallel workers. If None, uses cpu_count() - 1.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.n_cdus = None
        self.timesteps = None
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

    def load_data(self, sample_size: Optional[int] = None):
        """Load dataset and extract relevant features using Dask."""
        print(f"Loading data from {self.data_path}...")
        print(f"Using {self.n_workers} workers for parallel processing")

        if self.data_path.suffix == '.parquet':
            # Use Dask for parallel loading
            ddf = dd.read_parquet(self.data_path)
            df = ddf.compute()
        else:
            raise ValueError("Unsupported data format. Use .parquet")

        if sample_size:
            df = df.iloc[:sample_size]

        input_cols = [col for col in df.columns if 'Q_flow_total' in col]
        output_cols = [col for col in df.columns if 'CDU' in col and any(
            x in col for x in ['T_out', 'P_out', 'mdot']
        )]

        self.n_cdus = len(input_cols)
        print(f"Found {self.n_cdus} CDUs")

        self.data = {
            'Q_flow': df[input_cols].values,
            'outputs': df[output_cols].values,
            'output_names': output_cols
        }

        self.timesteps = len(df)
        print(f"Loaded {self.timesteps} timesteps")

    def identify_load_spikes(self, threshold_percentile: float = 95,
                            min_change: float = 1000) -> Dict[int, List[int]]:
        """
        Identify timesteps where CDUs experience significant load changes.

        Args:
            threshold_percentile: Percentile threshold for change magnitude
            min_change: Minimum absolute change in Q_flow (W)

        Returns:
            Dictionary mapping CDU index to list of spike timesteps
        """
        print("\nIdentifying load spikes...")
        Q_flow = self.data['Q_flow']

        # Calculate first difference (rate of change)
        dQ_dt = np.diff(Q_flow, axis=0)

        spikes = {}
        for cdu_idx in range(self.n_cdus):
            # Find large changes
            changes = np.abs(dQ_dt[:, cdu_idx])
            threshold = max(np.percentile(changes, threshold_percentile), min_change)

            spike_indices = np.where(changes > threshold)[0]
            spikes[cdu_idx] = spike_indices.tolist()

            print(f"CDU {cdu_idx}: {len(spike_indices)} spikes detected")

        return spikes

    def compute_lagged_correlation(self, source_cdu: int, target_cdu: int,
                                  max_lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-lagged cross-correlation between two CDUs.

        Args:
            source_cdu: Index of source CDU
            target_cdu: Index of target CDU
            max_lag: Maximum lag to consider (timesteps)

        Returns:
            lags: Array of lag values
            correlations: Correlation coefficient at each lag
        """
        source_signal = self.data['Q_flow'][:, source_cdu]
        target_signal = self.data['Q_flow'][:, target_cdu]

        lags = np.arange(-max_lag, max_lag + 1)
        correlations = np.zeros_like(lags, dtype=float)

        for i, lag in enumerate(lags):
            if lag < 0:
                corr, _ = pearsonr(source_signal[:lag], target_signal[-lag:])
            elif lag > 0:
                corr, _ = pearsonr(source_signal[lag:], target_signal[:-lag])
            else:
                corr, _ = pearsonr(source_signal, target_signal)
            correlations[i] = corr

        return lags, correlations

    def analyze_impulse_response(self, source_cdu: int, spike_idx: int,
                                window: int = 20) -> Dict:
        """
        Analyze how a load spike in source CDU affects all other CDUs.

        Args:
            source_cdu: CDU where spike occurs
            spike_idx: Timestep index of the spike
            window: Number of timesteps to analyze after spike

        Returns:
            Dictionary with response data for each CDU
        """
        if spike_idx + window >= self.timesteps:
            window = self.timesteps - spike_idx - 1

        responses = {}
        baseline_window = 10

        for target_cdu in range(self.n_cdus):
            if target_cdu == source_cdu:
                continue

            # Get baseline (before spike)
            baseline_start = max(0, spike_idx - baseline_window)
            baseline = np.mean(self.data['Q_flow'][baseline_start:spike_idx, target_cdu])

            # Get response (after spike)
            response_signal = self.data['Q_flow'][spike_idx:spike_idx+window, target_cdu]
            response_change = response_signal - baseline

            # Calculate metrics
            max_response = np.max(np.abs(response_change))
            time_to_peak = np.argmax(np.abs(response_change))

            responses[target_cdu] = {
                'signal': response_signal,
                'change': response_change,
                'max_response': max_response,
                'time_to_peak': time_to_peak,
                'distance': abs(target_cdu - source_cdu)
            }

        return responses

    def analyze_propagation_speed(self, source_cdu: int, spikes: List[int],
                                 max_lag: int = 20) -> Dict:
        """
        Analyze propagation speed and attenuation with distance using parallel processing.

        Args:
            source_cdu: Source CDU index
            spikes: List of spike timesteps
            max_lag: Maximum lag to consider

        Returns:
            Dictionary with propagation metrics
        """
        print(f"\nAnalyzing propagation from CDU {source_cdu} (parallelized)...")

        Q_flow = self.data['Q_flow']
        
        # Prepare arguments for parallel processing
        args_list = [
            (source_cdu, target_cdu, Q_flow, max_lag)
            for target_cdu in range(self.n_cdus)
            if target_cdu != source_cdu
        ]

        # Parallel propagation analysis
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_compute_propagation_for_target, args_list)

        distances = []
        optimal_lags = []
        max_correlations = []
        attenuations = []

        for result in results:
            if result is not None:
                target_cdu, distance, optimal_lag, max_corr, attenuation = result
                distances.append(distance)
                optimal_lags.append(optimal_lag)
                max_correlations.append(max_corr)
                attenuations.append(attenuation)

        return {
            'distances': np.array(distances),
            'optimal_lags': np.array(optimal_lags),
            'max_correlations': np.array(max_correlations),
            'attenuations': np.array(attenuations)
        }

    def create_propagation_map(self, source_cdu: int, spike_idx: int,
                              window: int = 20, lags: List[int] = [0, 2, 5, 10]):
        """
        Create animated/multi-panel propagation map.

        Args:
            source_cdu: Source CDU index
            spike_idx: Spike timestep
            window: Analysis window
            lags: Time lags to visualize
        """
        responses = self.analyze_impulse_response(source_cdu, spike_idx, window)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, lag in enumerate(lags):
            if lag >= window:
                continue

            ax = axes[idx]
            cdu_indices = []
            response_magnitudes = []

            for target_cdu, data in responses.items():
                if lag < len(data['change']):
                    cdu_indices.append(target_cdu)
                    response_magnitudes.append(data['change'][lag])

            # Create bar plot
            colors = ['red' if abs(r) > np.std(response_magnitudes) else 'blue'
                     for r in response_magnitudes]
            ax.bar(cdu_indices, response_magnitudes, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.axvline(x=source_cdu, color='green', linestyle='--',
                      linewidth=2, label='Source CDU')

            ax.set_xlabel('CDU Index')
            ax.set_ylabel('Load Change (W)')
            ax.set_title(f'Response at t+{lag} timesteps')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Load Propagation from CDU {source_cdu} (Spike at t={spike_idx})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'propagation_map_CDU{source_cdu}_t{spike_idx}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved propagation map to {self.output_dir}")

    def create_cross_correlation_matrix(self, lags: List[int] = [0, 1, 5, 10],
                                       max_lag: int = 20):
        """
        Create cross-correlation matrices at different lags using parallel processing.

        Args:
            lags: Specific lags to visualize
            max_lag: Maximum lag for computation
        """
        print("\nComputing cross-correlation matrices (parallelized)...")

        Q_flow = self.data['Q_flow']
        
        # Prepare all pairs for parallel computation
        all_pairs = []
        for source_cdu in range(self.n_cdus):
            for target_cdu in range(self.n_cdus):
                if source_cdu != target_cdu:
                    all_pairs.append((
                        source_cdu, target_cdu,
                        Q_flow[:, source_cdu], Q_flow[:, target_cdu],
                        max_lag
                    ))

        # Parallel correlation computation
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_compute_lagged_correlation_pair, all_pairs)

        # Build correlation lookup
        corr_lookup = {}
        for source_idx, target_idx, lags_arr, correlations in results:
            corr_lookup[(source_idx, target_idx)] = (lags_arr, correlations)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for idx, target_lag in enumerate(lags):
            corr_matrix = np.zeros((self.n_cdus, self.n_cdus))

            for source_cdu in range(self.n_cdus):
                for target_cdu in range(self.n_cdus):
                    if source_cdu == target_cdu:
                        corr_matrix[source_cdu, target_cdu] = 1.0
                    else:
                        lags_arr, correlations = corr_lookup[(source_cdu, target_cdu)]
                        lag_idx = np.where(lags_arr == target_lag)[0][0]
                        corr_matrix[source_cdu, target_cdu] = correlations[lag_idx]

            ax = axes[idx]
            sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0,
                       vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
            ax.set_xlabel('Target CDU')
            ax.set_ylabel('Source CDU')
            ax.set_title(f'Cross-Correlation at Lag={target_lag}')

        plt.suptitle('Load Propagation: Cross-Correlation Matrices',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_correlation_matrices.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-correlation matrices to {self.output_dir}")

    def analyze_attenuation_with_distance(self, source_cdus: List[int] = None):
        """
        Analyze how effect weakens with distance.

        Args:
            source_cdus: List of CDUs to analyze (default: sample of CDUs)
        """
        if source_cdus is None:
            source_cdus = [0, self.n_cdus//4, self.n_cdus//2, 3*self.n_cdus//4]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for source_cdu in source_cdus:
            metrics = self.analyze_propagation_speed(source_cdu, [], max_lag=20)

            ax1.scatter(metrics['distances'], np.abs(metrics['max_correlations']),
                       label=f'CDU {source_cdu}', alpha=0.6, s=50)

            ax2.scatter(metrics['distances'], metrics['optimal_lags'],
                       label=f'CDU {source_cdu}', alpha=0.6, s=50)

        ax1.set_xlabel('Distance (CDU units)')
        ax1.set_ylabel('Max Correlation (absolute)')
        ax1.set_title('Attenuation with Distance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Distance (CDU units)')
        ax2.set_ylabel('Optimal Lag (timesteps)')
        ax2.set_title('Propagation Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attenuation_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved attenuation analysis to {self.output_dir}")

    def run_full_analysis(self, sample_size: Optional[int] = None):
        """Run complete load propagation analysis."""
        print("="*60)
        print("LOAD PROPAGATION ANALYSIS")
        print("="*60)

        # Load data
        self.load_data(sample_size)

        # Identify spikes
        spikes = self.identify_load_spikes()

        # Select a few CDUs for detailed analysis
        analysis_cdus = [0, self.n_cdus//2, self.n_cdus-1]

        for cdu_idx in analysis_cdus:
            if spikes[cdu_idx]:
                spike_idx = spikes[cdu_idx][0]  # Use first spike
                self.create_propagation_map(cdu_idx, spike_idx)

        # Create cross-correlation matrices
        self.create_cross_correlation_matrix()

        # Analyze attenuation
        self.analyze_attenuation_with_distance()

        print("\n" + "="*60)
        print("Analysis complete! Results saved to:", self.output_dir)
        print("="*60)
