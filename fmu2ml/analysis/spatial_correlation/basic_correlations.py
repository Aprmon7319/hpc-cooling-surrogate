import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def _compute_correlation_pair(args):
    """Helper function for parallel pairwise correlation computation."""
    i, j, series_i, series_j, method = args
    
    if i == j:
        return i, j, 1.0
    
    if method == 'pearson':
        corr, _ = pearsonr(series_i, series_j)
    else:
        corr, _ = spearmanr(series_i, series_j)
    
    return i, j, corr


class SpatialCorrelationAnalyzer:
    """
    Analyzes spatial correlations between CDUs in the datacenter cooling system.
    """
    
    def __init__(self, num_cdus: int, n_workers: int = None):
        """
        Initialize the spatial correlation analyzer.
        
        Args:
            num_cdus: Number of CDUs in the datacenter
            n_workers: Number of parallel workers. If None, uses cpu_count() - 1.
        """
        self.num_cdus = num_cdus
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        self.output_metrics = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        self.correlation_matrices = {}
        self.distance_correlations = {}
        
    def compute_correlation_matrix(
        self, 
        data: pd.DataFrame, 
        metric: str,
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        Compute NxN correlation matrix for a specific metric across all CDUs using parallel processing.
        
        Args:
            data: DataFrame containing CDU data
            metric: Output metric to analyze (e.g., 'T_prim_r_C')
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            NxN correlation matrix
        """
        print(f"Computing correlation matrix using {self.n_workers} workers...")
        
        # Extract metric data for all CDUs
        cdu_data = []
        for cdu_idx in range(1, self.num_cdus + 1):
            col_name = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{metric}'
            if col_name in data.columns:
                cdu_data.append(data[col_name].values)
            else:
                raise ValueError(f"Column {col_name} not found in data")
        
        cdu_matrix = np.array(cdu_data)
        n = cdu_matrix.shape[0]
        
        # Prepare arguments for parallel processing
        args_list = []
        for i in range(n):
            for j in range(n):
                args_list.append((i, j, cdu_matrix[i], cdu_matrix[j], method))
        
        # Parallel correlation computation
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_compute_correlation_pair, args_list)
        
        # Build correlation matrix from results
        corr_matrix = np.zeros((n, n))
        for i, j, corr in results:
            corr_matrix[i, j] = corr
        
        self.correlation_matrices[metric] = corr_matrix
        return corr_matrix
    
    def compute_distance_correlation(
        self, 
        corr_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze correlation as a function of CDU distance |i - j|.
        
        Args:
            corr_matrix: NxN correlation matrix
            
        Returns:
            Tuple of (distances, correlations, distance_bins_stats)
        """
        n = corr_matrix.shape[0]
        distances = []
        correlations = []
        
        # Extract upper triangle (avoid duplicates)
        for i in range(n):
            for j in range(i + 1, n):
                distance = abs(i - j)
                distances.append(distance)
                correlations.append(corr_matrix[i, j])
        
        distances = np.array(distances)
        correlations = np.array(correlations)
        
        # Compute statistics for distance bins
        max_distance = int(distances.max())
        distance_bins = []
        
        for d in range(1, max_distance + 1):
            mask = distances == d
            if mask.sum() > 0:
                distance_bins.append({
                    'distance': d,
                    'mean_corr': correlations[mask].mean(),
                    'std_corr': correlations[mask].std(),
                    'count': mask.sum()
                })
        
        distance_bins_df = pd.DataFrame(distance_bins)
        
        return distances, correlations, distance_bins_df
    
    def test_spatial_hypothesis(
        self, 
        distance_bins_df: pd.DataFrame,
        significance_threshold: float = 0.3
    ) -> Dict:
        """
        Test hypothesis: "Nearby CDUs are more correlated than distant CDUs"
        
        Args:
            distance_bins_df: DataFrame with distance-wise correlation statistics
            significance_threshold: Correlation threshold for significance
            
        Returns:
            Dictionary with hypothesis test results
        """
        # Calculate correlation range (where correlation drops below threshold)
        significant_distances = distance_bins_df[
            distance_bins_df['mean_corr'] >= significance_threshold
        ]
        
        if len(significant_distances) > 0:
            correlation_range = significant_distances['distance'].max()
        else:
            correlation_range = 0
        
        # Calculate decay rate (linear fit)
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            distance_bins_df['distance'], 
            distance_bins_df['mean_corr']
        )
        
        # Compute clustering coefficient (nearby correlations vs distant)
        nearby_threshold = 5
        nearby_corr = distance_bins_df[
            distance_bins_df['distance'] <= nearby_threshold
        ]['mean_corr'].mean()
        distant_corr = distance_bins_df[
            distance_bins_df['distance'] > nearby_threshold
        ]['mean_corr'].mean()
        
        results = {
            'correlation_range': correlation_range,
            'decay_rate': slope,
            'decay_r_squared': r_value**2,
            'decay_p_value': p_value,
            'nearby_correlation': nearby_corr,
            'distant_correlation': distant_corr,
            'clustering_ratio': nearby_corr / distant_corr if distant_corr != 0 else np.inf,
            'hypothesis_supported': slope < 0 and p_value < 0.05
        }
        
        return results
    
    def plot_correlation_matrix(
        self, 
        corr_matrix: np.ndarray, 
        metric: str,
        save_path: Optional[str] = None
    ):
        """
        Create correlation matrix heatmap visualization.
        
        Args:
            corr_matrix: NxN correlation matrix
            metric: Name of the metric
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        # Customize
        ax.set_xlabel('CDU Index', fontsize=12)
        ax.set_ylabel('CDU Index', fontsize=12)
        ax.set_title(f'Spatial Correlation Matrix: {metric}', fontsize=14, fontweight='bold')
        
        # Set tick labels
        tick_positions = np.arange(0, self.num_cdus, max(1, self.num_cdus // 10))
        ax.set_xticks(tick_positions + 0.5)
        ax.set_yticks(tick_positions + 0.5)
        ax.set_xticklabels(tick_positions + 1)
        ax.set_yticklabels(tick_positions + 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")
        
        plt.show()
        
    def plot_correlation_vs_distance(
        self,
        distances: np.ndarray,
        correlations: np.ndarray,
        distance_bins_df: pd.DataFrame,
        metric: str,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation vs distance with confidence bands.
        
        Args:
            distances: Array of CDU distances
            correlations: Array of correlation values
            distance_bins_df: DataFrame with binned statistics
            metric: Name of the metric
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Scatter plot of all pairs
        ax.scatter(distances, correlations, alpha=0.3, s=20, label='Individual pairs')
        
        # Plot binned means with confidence bands
        ax.plot(
            distance_bins_df['distance'], 
            distance_bins_df['mean_corr'],
            'r-', 
            linewidth=2, 
            marker='o',
            label='Mean correlation'
        )
        
        # Add confidence bands (±1 std)
        ax.fill_between(
            distance_bins_df['distance'],
            distance_bins_df['mean_corr'] - distance_bins_df['std_corr'],
            distance_bins_df['mean_corr'] + distance_bins_df['std_corr'],
            alpha=0.3,
            color='red',
            label='±1 std'
        )
        
        # Customize
        ax.set_xlabel('CDU Distance |i - j|', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title(f'Spatial Correlation Decay: {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([-1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation vs distance plot to {save_path}")
        
        plt.show()
    
    def plot_correlation_decay_heatmap(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        distance_bins: List[Tuple[int, int]] = [(1, 1), (2, 5), (6, 10), (11, 999)],
        method: str = 'pearson',
        save_path: Optional[str] = None
    ):
        """
        Create correlation decay heatmap across metrics and distance bins.
        
        Args:
            data: DataFrame containing CDU data
            metrics: List of metrics to analyze
            distance_bins: List of (min, max) distance tuples
            method: Correlation method
            save_path: Optional path to save the figure
        """
        # Prepare matrix: rows = metrics, columns = distance bins
        decay_matrix = np.zeros((len(metrics), len(distance_bins)))
        bin_labels = [f"{mn}-{mx}" if mn != mx else f"{mn}" for mn, mx in distance_bins]
        
        for i, metric in enumerate(metrics):
            try:
                # Compute correlation matrix
                corr_matrix = self.compute_correlation_matrix(data, metric, method)
                distances, correlations, _ = self.compute_distance_correlation(corr_matrix)
                
                # Compute mean correlation for each distance bin
                for j, (min_dist, max_dist) in enumerate(distance_bins):
                    mask = (distances >= min_dist) & (distances <= max_dist)
                    if mask.sum() > 0:
                        decay_matrix[i, j] = correlations[mask].mean()
                    else:
                        decay_matrix[i, j] = np.nan
                        
            except Exception as e:
                print(f"Error processing {metric}: {e}")
                decay_matrix[i, :] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            decay_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=bin_labels,
            yticklabels=metrics,
            cbar_kws={'label': 'Average Correlation'},
            ax=ax
        )
        
        ax.set_xlabel('CDU Distance Range', fontsize=12)
        ax.set_ylabel('Output Metric', fontsize=12)
        ax.set_title('Correlation Decay Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation decay heatmap to {save_path}")
        
        plt.show()
        
        return decay_matrix
    
    def analyze_metric(
        self,
        data: pd.DataFrame,
        metric: str,
        method: str = 'pearson',
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Complete spatial correlation analysis for a single metric.
        
        Args:
            data: DataFrame containing CDU data
            metric: Output metric to analyze
            method: Correlation method
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {metric}")
        print(f"{'='*60}")
        
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(data, metric, method)
        
        # Compute distance-correlation relationship
        distances, correlations, distance_bins_df = self.compute_distance_correlation(corr_matrix)
        
        # Test spatial hypothesis
        hypothesis_results = self.test_spatial_hypothesis(distance_bins_df)
        
        # Print results
        print(f"\nSpatial Correlation Analysis Results:")
        print(f"  Correlation Range: {hypothesis_results['correlation_range']} CDUs")
        print(f"  Decay Rate: {hypothesis_results['decay_rate']:.4f}")
        print(f"  Decay R²: {hypothesis_results['decay_r_squared']:.4f}")
        print(f"  Nearby Correlation (≤5 CDUs): {hypothesis_results['nearby_correlation']:.4f}")
        print(f"  Distant Correlation (>5 CDUs): {hypothesis_results['distant_correlation']:.4f}")
        print(f"  Clustering Ratio: {hypothesis_results['clustering_ratio']:.4f}")
        print(f"  Hypothesis Supported: {hypothesis_results['hypothesis_supported']}")
        
        # Create visualizations
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_correlation_matrix(
                corr_matrix, 
                metric,
                save_path=f"{save_dir}/{metric}_correlation_matrix.png"
            )
            
            self.plot_correlation_vs_distance(
                distances,
                correlations,
                distance_bins_df,
                metric,
                save_path=f"{save_dir}/{metric}_correlation_vs_distance.png"
            )
        else:
            self.plot_correlation_matrix(corr_matrix, metric)
            self.plot_correlation_vs_distance(distances, correlations, distance_bins_df, metric)
        
        return {
            'correlation_matrix': corr_matrix,
            'distance_bins': distance_bins_df,
            'hypothesis_results': hypothesis_results
        }
    
    def analyze_all_metrics(
        self,
        data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        method: str = 'pearson',
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Analyze spatial correlations for all output metrics.
        
        Args:
            data: DataFrame containing CDU data
            metrics: List of metrics to analyze (defaults to all)
            method: Correlation method
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping metrics to their analysis results
        """
        if metrics is None:
            metrics = self.output_metrics
        
        results = {}
        
        for metric in metrics:
            try:
                results[metric] = self.analyze_metric(data, metric, method, save_dir)
            except Exception as e:
                print(f"Error analyzing {metric}: {e}")
                continue
        
        # Create comprehensive decay heatmap
        if save_dir:
            self.plot_correlation_decay_heatmap(
                data, 
                metrics, 
                method=method,
                save_path=f"{save_dir}/correlation_decay_heatmap.png"
            )
        else:
            self.plot_correlation_decay_heatmap(data, metrics, method=method)
        
        return results