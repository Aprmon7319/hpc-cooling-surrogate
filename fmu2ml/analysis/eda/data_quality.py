import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import scipy.stats as stats
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import the existing data loader
from fmu2ml.data.processors.data_loader import DataLoader
from fmu2ml.utils.io_utils import load_config

from .utils import _detect_outliers_for_column, _process_cdu_wrapper, _get_unit_for_metric

warnings.filterwarnings('ignore')


@dataclass
class VariableGroup:
    """Groups related variables for analysis"""
    name: str
    variables: List[str]
    unit: str
    description: str


class DataQualityAnalyzer:
    """
    Analyzes data quality and distribution characteristics for cooling model data.
    Implements Phase 1: Data Quality & Structural Assessment.
    Uses multiprocessing for parallel processing and integrates with existing data loader.
    """
    
    def __init__(
        self, 
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], str]] = None,
        output_dir: Optional[str] = None,
        n_workers: Optional[int] = None,
        use_data_loader: bool = True
    ):
        """
        Initialize the analyzer with flexible data input options.
        
        Args:
            data: Pre-loaded DataFrame (optional if data_path is provided)
            data_path: Path to data file (used if data is None)
            config: Configuration dict or path to config file
            output_dir: Directory for output files
            n_workers: Number of parallel workers
            use_data_loader: Whether to use the DataLoader for loading data
        """
        # Load configuration
        if isinstance(config, str):
            self.config = load_config(config)
        elif config is None:
            # Try to load default EDA config
            try:
                default_config_path = Path(__file__).parent.parent.parent / 'config' / 'defaults' / 'eda_config.yaml'
                if default_config_path.exists():
                    self.config = load_config(str(default_config_path))
                else:
                    self.config = {}
                    print("Warning: No config provided and default not found. Using empty config.")
            except Exception as e:
                print(f"Warning: Could not load default config: {e}")
                self.config = {}
        else:
            self.config = config
        
        # Load or use provided data
        if data is not None:
            self.data = data
            print(f"Using provided DataFrame with shape: {data.shape}")
        elif data_path is not None:
            if use_data_loader:
                print(f"Loading data using DataLoader from: {data_path}")
                self.data_loader = DataLoader(config=self.config)
                self.data = self.data_loader.load_data(
                    data_path,
                    format='auto'  # Auto-detect format
                )
            else:
                # Fallback to simple loading
                print(f"Loading data directly from: {data_path}")
                from .utils import load_simulation_data
                self.data = load_simulation_data(data_path, self.config)
        else:
            raise ValueError("Either 'data' or 'data_path' must be provided")
        
        # Extract key parameters from config
        self.num_cdus = self.config.get('NUM_CDUS', 49)
        self.num_blocks = self.config.get('NUM_BLOCKS', self.num_cdus)
        
        # Set up parallel processing
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        print(f"Using {self.n_workers} workers for parallel processing")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = self.config.get('OUTPUT_DIR', 'eda_results')
            phase_dir = self.config.get('EDA_PHASE1_DIR', 'phase1')
            self.output_dir = Path(base_dir) / phase_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.cdu_dir = self.output_dir / 'cdu_specific'
        self.cdu_dir.mkdir(exist_ok=True)
        
        self.metric_dir = self.output_dir / 'by_metric'
        self.metric_dir.mkdir(exist_ok=True)
        
        self.summary_dir = self.output_dir / 'summary'
        self.summary_dir.mkdir(exist_ok=True)
        
        # Get variable naming patterns from config
        self.input_patterns = self.config.get('INPUT_PATTERNS', {})
        self.output_patterns = self.config.get('OUTPUT_PATTERNS', {})
        
        # Results storage
        self.statistics_summary = {}
        self.normality_tests = {}
        
        print(f"Initialized DataQualityAnalyzer:")
        print(f"  - Number of CDUs: {self.num_cdus}")
        print(f"  - Data shape: {self.data.shape}")
        print(f"  - Output directory: {self.output_dir}")
    
    @classmethod
    def from_config(cls, config_path: str, data_path: str, **kwargs):
        """
        Convenience constructor to create analyzer from config file.
        
        Args:
            config_path: Path to configuration YAML file
            data_path: Path to data file
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            DataQualityAnalyzer instance
        """
        config = load_config(config_path)
        return cls(data_path=data_path, config=config, **kwargs)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dictionary with data statistics and metadata
        """
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.data.isna().sum().to_dict(),
            'num_cdus': self.num_cdus,
            'time_range': None
        }
        
        # Add time range if available
        if 'time' in self.data.columns:
            info['time_range'] = {
                'start': self.data['time'].min(),
                'end': self.data['time'].max(),
                'duration': self.data['time'].max() - self.data['time'].min()
            }
        
        return info
    
    def save_data_info(self):
        """Save data information to file."""
        info = self.get_data_info()
        
        # Convert to human-readable format
        with open(self.summary_dir / 'data_info.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA INFORMATION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Shape: {info['shape']}\n")
            f.write(f"Number of CDUs: {info['num_cdus']}\n")
            f.write(f"Memory Usage: {info['memory_usage_mb']:.2f} MB\n\n")
            
            if info['time_range']:
                f.write("Time Range:\n")
                f.write(f"  Start: {info['time_range']['start']}\n")
                f.write(f"  End: {info['time_range']['end']}\n")
                f.write(f"  Duration: {info['time_range']['duration']}\n\n")
            
            f.write(f"Columns ({len(info['columns'])}):\n")
            for col in info['columns'][:20]:  # Show first 20
                f.write(f"  - {col}\n")
            if len(info['columns']) > 20:
                f.write(f"  ... and {len(info['columns']) - 20} more\n")
        
        print(f"Data info saved to: {self.summary_dir / 'data_info.txt'}")
    
    def run_full_analysis(self):
        """Execute complete Phase 1 analysis with parallel processing."""
        print("="*60)
        print("Phase 1: Data Quality & Structural Assessment")
        print("="*60)
        
        # Save data information
        print("\n0. Saving Data Information")
        self.save_data_info()
        
        print("\n1.1 CDU-Specific Analysis")
        self.analyze_cdu_specific()
        
        print("\n1.2 Metric-Based Analysis")
        self.analyze_by_metric()
                
        print("\n1.4 Normality Assessment")
        self.assess_normality()
        
        print("\n1.5 Correlation Analysis")
        self.analyze_correlations()
        
        print("\n1.6 Outlier Detection")
        self.detect_outliers()
        
        print("\n1.7 Missing Data Analysis")
        self.analyze_missing_data()
                
        print(f"\nAll results saved to: {self.output_dir}")
    
    
    def analyze_cdu_specific(self):
        """Generate CDU-specific analysis with separate plots for each variable."""
        print(f"Generating CDU-specific visualizations for {self.num_cdus} CDUs...")
        
        # Prepare data dictionary for multiprocessing
        data_dict = {}
        for col in self.data.columns:
            if col not in ['time', 'hue']:
                data_dict[col] = self.data[col].dropna()
        
        # Add time data if available
        if 'time' in self.data.columns:
            data_dict['time'] = self.data['time']
        
        # Prepare arguments for parallel processing
        process_args = [
            (cdu_idx, data_dict, self.output_patterns, str(self.cdu_dir))
            for cdu_idx in range(1, self.num_cdus + 1)
        ]
        
        # Use parallel processing for CDU analysis
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for args in process_args:
                future = executor.submit(_process_cdu_wrapper, args)
                futures.append(future)
            
            # Monitor progress
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0 or completed == self.num_cdus:
                    print(f"  Processed {completed}/{self.num_cdus} CDUs")
    
    def analyze_by_metric(self):
        """Generate metric-based comparison plots across all CDUs."""
        print(f"Generating metric-based comparison plots...")
        
        # Get all unique metrics
        cdu_metrics = self.output_patterns.get(
            'cdu_metrics',
            ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
             'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
             'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        )
        
        # Process each metric sequentially (they're already quite fast)
        for metric in cdu_metrics:
            self._process_metric_comparison(metric)
    
    def _process_metric_comparison(self, metric: str):
        """Create comparison plots for a specific metric across all CDUs."""
        # Collect data for this metric from all CDUs
        metric_data = {}
        cdu_prefix_pattern = self.output_patterns.get(
            'cdu_summary_prefix',
            'simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.'
        )
        
        for cdu_idx in range(1, self.num_cdus + 1):
            var_name = cdu_prefix_pattern.format(cdu_idx) + metric
            if var_name in self.data.columns:
                metric_data[cdu_idx] = self.data[var_name].dropna()
        
        if not metric_data:
            return
        
        # Create comparison plots
        self._create_metric_comparison_plots(metric, metric_data)
    
    def _create_metric_comparison_plots(self, metric: str, metric_data: Dict[int, pd.Series]):
        """Create comprehensive comparison plots for a metric across CDUs."""
        safe_metric_name = metric.replace('/', '_').replace('.', '_')
        
        try:
            # 1. Time series comparison (sample of CDUs if too many)
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            
            # Plot subset if too many CDUs
            cdu_indices = list(metric_data.keys())
            if len(cdu_indices) > 20:
                # Plot every nth CDU to avoid overcrowding
                step = max(1, len(cdu_indices) // 20)
                plot_indices = cdu_indices[::step]
            else:
                plot_indices = cdu_indices
            
            # Time series
            ax1 = axes[0]
            x_axis = self.data['time'] if 'time' in self.data.columns else range(len(self.data))
            
            for cdu_idx in plot_indices:
                ax1.plot(x_axis, metric_data[cdu_idx], alpha=0.6, linewidth=0.8, label=f'CDU {cdu_idx}')
            
            ax1.set_xlabel('Time' if 'time' in self.data.columns else 'Sample')
            ax1.set_ylabel(_get_unit_for_metric(metric))
            ax1.set_title(f'{metric} - Time Series Comparison (Showing {len(plot_indices)}/{len(cdu_indices)} CDUs)', 
                         fontsize=12, fontweight='bold')
            ax1.legend(ncol=min(10, len(plot_indices)), fontsize=7, loc='upper right')
            ax1.grid(alpha=0.3)
            
            # Box plot comparison (all CDUs)
            ax2 = axes[1]
            box_data = [metric_data[cdu_idx] for cdu_idx in sorted(metric_data.keys())]
            box_positions = list(sorted(metric_data.keys()))
            
            bp = ax2.boxplot(box_data, positions=box_positions, patch_artist=True,
                            widths=0.6, showfliers=False)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax2.set_xlabel('CDU Number')
            ax2.set_ylabel(_get_unit_for_metric(metric))
            ax2.set_title(f'{metric} - Distribution Comparison (All CDUs)', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3, axis='y')
            
            # Set x-axis ticks appropriately
            if len(box_positions) > 50:
                # Show every 5th CDU label
                tick_positions = [pos for pos in box_positions if pos % 5 == 1 or pos == box_positions[-1]]
                ax2.set_xticks(tick_positions)
            
            plt.tight_layout()
            plt.savefig(self.metric_dir / f'{safe_metric_name}_comparison.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Statistical summary heatmap
            self._create_metric_stats_heatmap(metric, metric_data, safe_metric_name)
            
            # 3. Distribution comparison (violin plot)
            self._create_metric_violin_plot(metric, metric_data, safe_metric_name)
            
        except Exception as e:
            print(f"Error creating metric comparison for {metric}: {e}")
            plt.close('all')
    
    def _create_metric_stats_heatmap(self, metric: str, metric_data: Dict[int, pd.Series], safe_name: str):
        """Create heatmap showing statistics for each CDU."""
        try:
            stats_list = []
            
            for cdu_idx in sorted(metric_data.keys()):
                data = metric_data[cdu_idx]
                stats_list.append({
                    'CDU': cdu_idx,
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Median': data.median()
                })
            
            stats_df = pd.DataFrame(stats_list)
            stats_df.set_index('CDU', inplace=True)
            
            # Normalize each column for better visualization
            stats_norm = (stats_df - stats_df.min()) / (stats_df.max() - stats_df.min())
            
            fig, ax = plt.subplots(figsize=(12, max(8, self.num_cdus * 0.15)))
            
            sns.heatmap(stats_norm.T, annot=False, cmap='YlOrRd', 
                       cbar_kws={'label': 'Normalized Value'}, ax=ax)
            
            ax.set_title(f'{metric} - Statistical Summary Heatmap', fontsize=12, fontweight='bold')
            ax.set_xlabel('CDU Number')
            ax.set_ylabel('Statistic')
            
            plt.tight_layout()
            plt.savefig(self.metric_dir / f'{safe_name}_stats_heatmap.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating stats heatmap for {metric}: {e}")
            plt.close('all')
    
    def _create_metric_violin_plot(self, metric: str, metric_data: Dict[int, pd.Series], safe_name: str):
        """Create violin plot comparing distributions."""
        try:
            # If too many CDUs, create multiple plots
            cdu_indices = sorted(metric_data.keys())
            cdus_per_plot = 25
            
            for plot_idx, start_idx in enumerate(range(0, len(cdu_indices), cdus_per_plot)):
                end_idx = min(start_idx + cdus_per_plot, len(cdu_indices))
                plot_cdus = cdu_indices[start_idx:end_idx]
                
                fig, ax = plt.subplots(figsize=(max(12, len(plot_cdus) * 0.5), 6))
                
                plot_data = [metric_data[cdu_idx] for cdu_idx in plot_cdus]
                positions = list(range(len(plot_cdus)))
                
                parts = ax.violinplot(plot_data, positions=positions, 
                                     showmeans=True, showmedians=True)
                
                for pc in parts['bodies']:
                    pc.set_facecolor('lightgreen')
                    pc.set_alpha(0.7)
                
                ax.set_xticks(positions)
                ax.set_xticklabels([f'CDU{cdu}' for cdu in plot_cdus], rotation=45, ha='right')
                ax.set_ylabel(_get_unit_for_metric(metric))
                ax.set_title(f'{metric} - Distribution Comparison (CDU {plot_cdus[0]}-{plot_cdus[-1]})', 
                            fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                
                plt.tight_layout()
                suffix = f'_part{plot_idx+1}' if len(cdu_indices) > cdus_per_plot else ''
                plt.savefig(self.metric_dir / f'{safe_name}_violin{suffix}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error creating violin plot for {metric}: {e}")
            plt.close('all')
    
    def assess_normality(self):
        """Assess normality for key variables."""
        print("Assessing normality...")
        
        normality_results = []
        
        # Get all output metrics
        cdu_metrics = self.output_patterns.get('cdu_metrics', [])
        cdu_prefix_pattern = self.output_patterns.get('cdu_summary_prefix', '')
        
        # Sample first 10 CDUs and all metrics
        for cdu_idx in range(1, min(self.num_cdus + 1, 11)):
            for metric in cdu_metrics:
                var_name = cdu_prefix_pattern.format(cdu_idx) + metric
                
                if var_name not in self.data.columns:
                    continue
                
                data_clean = self.data[var_name].dropna()
                
                if len(data_clean) == 0:
                    continue
                
                try:
                    # Shapiro-Wilk test (sample if too large)
                    sample_size = min(5000, len(data_clean))
                    sample_data = data_clean.sample(n=sample_size, random_state=42) if len(data_clean) > sample_size else data_clean
                    
                    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                    
                    normality_results.append({
                        'Variable': var_name,
                        'Shapiro_Statistic': shapiro_stat,
                        'Shapiro_p_value': shapiro_p,
                        'Is_Normal': shapiro_p > 0.05
                    })
                except Exception as e:
                    print(f"Error in normality test for {var_name}: {e}")
        
        self.normality_tests = pd.DataFrame(normality_results)
        self.normality_tests.to_csv(self.summary_dir / 'normality_tests.csv', index=False)
        
        print(f"Normality assessment completed for {len(normality_results)} variables")
    
    def analyze_correlations(self):
        """Analyze correlations with focus on input-output relationships."""
        print("Analyzing correlations...")
        
        # Create correlations subdirectory
        corr_dir = self.summary_dir / 'correlations'
        corr_dir.mkdir(exist_ok=True)
        
        # Identify input and output columns
        input_cols = self._get_input_columns()
        output_cols = self._get_output_columns()
        
        print(f"  Found {len(input_cols)} input variables")
        print(f"  Found {len(output_cols)} output variables")
        
        # 1. Input-to-Output Correlation Analysis (Most Important)
        print("  Computing input-output correlations...")
        self._analyze_input_output_correlations(input_cols, output_cols, corr_dir)
        
        # 2. Output-to-Output Correlation Analysis
        print("  Computing output-output correlations...")
        self._analyze_output_output_correlations(output_cols, corr_dir)
        
        # 3. Full correlation matrix (sampled if too large)
        print("  Computing full correlation matrix...")
        self._analyze_full_correlations(input_cols, output_cols, corr_dir)
        
        print(f"Correlation analysis completed. Results saved to: {corr_dir}")

    def _get_input_columns(self):
        """Get all input column names."""
        input_cols = []
        
        # CDU-specific inputs
        for i in range(1, self.num_cdus + 1):
            q_flow_col = f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total'
            t_air_col = f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air'
            
            if q_flow_col in self.data.columns:
                input_cols.append(q_flow_col)
            if t_air_col in self.data.columns:
                input_cols.append(t_air_col)
        
        # Global input
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        if t_ext_col in self.data.columns:
            input_cols.append(t_ext_col)
        
        return input_cols

    def _get_output_columns(self):
        """Get all output column names."""
        output_cols = []
        
        # CDU output metrics
        cdu_metrics = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                    'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                    'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        
        for i in range(1, self.num_cdus + 1):
            for metric in cdu_metrics:
                col = f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{metric}'
                if col in self.data.columns:
                    output_cols.append(col)
            
            # HTC
            htc_col = f'simulator[1].datacenter[1].computeBlock[{i}].cabinet[1].summary.htc'
            if htc_col in self.data.columns:
                output_cols.append(htc_col)
        
        # Datacenter-level outputs
        dc_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        pue_col = 'pue'
        
        if dc_flow_col in self.data.columns:
            output_cols.append(dc_flow_col)
        if pue_col in self.data.columns:
            output_cols.append(pue_col)
        
        return output_cols

    def _analyze_input_output_correlations(self, input_cols, output_cols, corr_dir):
        """Analyze correlations between inputs and outputs with significance testing."""
        try:
            # Sample outputs if too many (keep all inputs)
            sampled_output_cols = self._smart_sample_outputs(output_cols, max_outputs=100)
            
            # Compute correlation matrix
            combined_data = self.data[input_cols + sampled_output_cols].dropna()
            
            if len(combined_data) == 0:
                print("  Warning: No valid data for input-output correlation")
                return
            
            # Calculate correlations and p-values
            n = len(combined_data)
            correlations = []
            
            for input_col in input_cols:
                for output_col in sampled_output_cols:
                    try:
                        r, p_value = stats.pearsonr(combined_data[input_col], combined_data[output_col])
                        correlations.append({
                            'Input': self._simplify_column_name(input_col),
                            'Output': self._simplify_column_name(output_col),
                            'Correlation': r,
                            'P_value': p_value,
                            'Significant': p_value < 0.05,
                            'Abs_Correlation': abs(r)
                        })
                    except Exception as e:
                        continue
            
            corr_df = pd.DataFrame(correlations)
            
            if corr_df.empty:
                print("  Warning: No correlations computed")
                return
            
            # Save full results
            corr_df_sorted = corr_df.sort_values('Abs_Correlation', ascending=False)
            corr_df_sorted.to_csv(corr_dir / 'input_output_correlations_full.csv', index=False)
            
            # Save top 20 correlations
            top_20 = corr_df_sorted.head(20)
            top_20.to_csv(corr_dir / 'input_output_top20.csv', index=False)
            
            print(f"    Computed {len(corr_df)} input-output correlations")
            print(f"    Significant correlations: {corr_df['Significant'].sum()} ({100*corr_df['Significant'].mean():.1f}%)")
            print(f"    Top correlation: {top_20.iloc[0]['Correlation']:.3f} ({top_20.iloc[0]['Input']} -> {top_20.iloc[0]['Output']})")
            
            # Create visualization
            self._plot_input_output_correlations(input_cols, sampled_output_cols, combined_data, corr_dir)
            
            # Create histogram of correlation strengths
            self._plot_correlation_histogram(corr_df, corr_dir, 'input_output')
            
        except Exception as e:
            print(f"  Error in input-output correlation analysis: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_output_output_correlations(self, output_cols, corr_dir):
        """Analyze correlations between output variables."""
        try:
            # Sample outputs if too many
            sampled_output_cols = self._smart_sample_outputs(output_cols, max_outputs=150)
            
            # Compute correlation matrix
            output_data = self.data[sampled_output_cols].dropna()
            
            if len(output_data) == 0:
                print("  Warning: No valid data for output-output correlation")
                return
            
            corr_matrix = output_data.corr()
            
            # Extract upper triangle (avoid duplicates)
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    r = corr_matrix.iloc[i, j]
                    
                    # Calculate p-value
                    n = len(output_data[[col1, col2]].dropna())
                    if n > 2:
                        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    else:
                        p_value = 1.0
                    
                    correlations.append({
                        'Variable_1': self._simplify_column_name(col1),
                        'Variable_2': self._simplify_column_name(col2),
                        'Correlation': r,
                        'P_value': p_value,
                        'Significant': p_value < 0.05,
                        'Abs_Correlation': abs(r)
                    })
            
            corr_df = pd.DataFrame(correlations)
            corr_df_sorted = corr_df.sort_values('Abs_Correlation', ascending=False)
            
            # Save results
            corr_df_sorted.to_csv(corr_dir / 'output_output_correlations_full.csv', index=False)
            top_20 = corr_df_sorted.head(20)
            top_20.to_csv(corr_dir / 'output_output_top20.csv', index=False)
            
            print(f"    Computed {len(corr_df)} output-output correlations")
            print(f"    Significant correlations: {corr_df['Significant'].sum()} ({100*corr_df['Significant'].mean():.1f}%)")
            
            # Create visualization
            self._plot_output_output_heatmap(corr_matrix, sampled_output_cols, corr_dir)
            
            # Create histogram
            self._plot_correlation_histogram(corr_df, corr_dir, 'output_output')
            
        except Exception as e:
            print(f"  Error in output-output correlation analysis: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_full_correlations(self, input_cols, output_cols, corr_dir):
        """Analyze full correlation matrix with smart sampling."""
        try:
            # Smart sampling
            sampled_inputs = input_cols  # Keep all inputs
            sampled_outputs = self._smart_sample_outputs(output_cols, max_outputs=100)
            all_sampled = sampled_inputs + sampled_outputs
            
            # Compute correlation matrix
            corr_data = self.data[all_sampled].dropna()
            
            if len(corr_data) == 0:
                print("  Warning: No valid data for full correlation matrix")
                return
            
            corr_matrix = corr_data.corr()
            
            # Save full matrix
            corr_matrix.to_csv(corr_dir / 'correlation_matrix_full.csv')
            
            # Create enhanced visualization with labels
            self._plot_full_correlation_matrix(corr_matrix, sampled_inputs, sampled_outputs, corr_dir)
            
            print(f"    Full correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]} variables")
            
        except Exception as e:
            print(f"  Error in full correlation analysis: {e}")
            import traceback
            traceback.print_exc()

    def _smart_sample_outputs(self, output_cols, max_outputs=100):
        """Smart sampling of output variables ensuring representation from all CDUs."""
        if len(output_cols) <= max_outputs:
            return output_cols
        
        sampled = []
        
        # Always include datacenter-level variables
        dc_vars = [col for col in output_cols if 'datacenter[1].summary' in col or col == 'pue']
        sampled.extend(dc_vars)
        
        # Get CDU-specific variables
        cdu_vars = [col for col in output_cols if col not in dc_vars]
        
        # Sample CDUs evenly
        cdus_to_sample = min(15, self.num_cdus)  # Sample up to 15 CDUs
        if self.num_cdus <= cdus_to_sample:
            cdu_indices = list(range(1, self.num_cdus + 1))
        else:
            step = self.num_cdus // cdus_to_sample
            cdu_indices = list(range(1, self.num_cdus + 1, step))[:cdus_to_sample]
        
        # Get all variables for sampled CDUs
        for cdu_idx in cdu_indices:
            cdu_specific = [col for col in cdu_vars if f'Block[{cdu_idx}]' in col or f'computeBlock_{cdu_idx}' in col]
            sampled.extend(cdu_specific)
        
        # If still too many, sample metrics
        if len(sampled) > max_outputs:
            # Keep all from first 10 CDUs, sample rest
            priority_cdus = cdu_indices[:10]
            priority_vars = [col for col in sampled if any(f'Block[{i}]' in col for i in priority_cdus)]
            other_vars = [col for col in sampled if col not in priority_vars]
            
            remaining = max_outputs - len(priority_vars)
            if remaining > 0:
                sampled = priority_vars + other_vars[:remaining]
            else:
                sampled = priority_vars[:max_outputs]
        
        return sampled

    def _simplify_column_name(self, col_name):
        """Simplify column names for readability."""
        # Extract CDU number
        if 'Block[' in col_name or 'computeBlock_' in col_name:
            import re
            match = re.search(r'(?:Block\[|computeBlock_)(\d+)', col_name)
            if match:
                cdu_num = match.group(1)
                
                # Extract metric name
                if '.summary.' in col_name:
                    metric = col_name.split('.summary.')[-1]
                    return f"CDU{cdu_num}_{metric}"
                elif 'sources_' in col_name:
                    if 'Q_flow' in col_name:
                        return f"CDU{cdu_num}_Q_flow"
                    elif 'T_Air' in col_name:
                        return f"CDU{cdu_num}_T_Air"
        
        # Global variables
        if 'T_ext' in col_name:
            return 'T_ext'
        elif col_name == 'pue':
            return 'PUE'
        elif 'datacenter[1].summary' in col_name:
            return 'DC_' + col_name.split('.')[-1]
        
        return col_name

    def _plot_input_output_correlations(self, input_cols, output_cols, data, corr_dir):
        """Plot input-output correlation heatmap."""
        try:
            # Compute correlation matrix
            input_data = data[input_cols]
            output_data = data[output_cols]
            
            # Calculate correlations
            corr_matrix = pd.DataFrame(
                index=[self._simplify_column_name(col) for col in input_cols],
                columns=[self._simplify_column_name(col) for col in output_cols]
            )
            
            for inp in input_cols:
                for out in output_cols:
                    corr_matrix.loc[self._simplify_column_name(inp), self._simplify_column_name(out)] = \
                        data[inp].corr(data[out])
            
            corr_matrix = corr_matrix.astype(float)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(output_cols) * 0.15), 
                                            max(8, len(input_cols) * 0.15)))
            
            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='right', fontsize=7)
            ax.set_yticklabels(corr_matrix.index, fontsize=8)
            
            # Color-code y-axis labels (inputs)
            for i, label in enumerate(ax.get_yticklabels()):
                if 'Q_flow' in label.get_text():
                    label.set_color('red')
                    label.set_fontweight('bold')
                elif 'T_Air' in label.get_text():
                    label.set_color('green')
                    label.set_fontweight('bold')
                elif 'T_ext' in label.get_text():
                    label.set_color('purple')
                    label.set_fontweight('bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=10)
            
            # Add title
            ax.set_title('Input-Output Correlation Matrix\n' + 
                        '(Red=Q_flow, Green=T_air, Purple=T_ext)', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.set_xlabel('Output Variables', fontsize=10, fontweight='bold')
            ax.set_ylabel('Input Variables', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(corr_dir / 'input_output_correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    Error creating input-output heatmap: {e}")
            plt.close('all')

    def _plot_output_output_heatmap(self, corr_matrix, output_cols, corr_dir):
        """Plot output-output correlation heatmap."""
        try:
            fig, ax = plt.subplots(figsize=(16, 14))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Simplify labels
            simplified_labels = [self._simplify_column_name(col) for col in output_cols]
            corr_matrix_plot = corr_matrix.copy()
            corr_matrix_plot.index = simplified_labels
            corr_matrix_plot.columns = simplified_labels
            
            # Plot
            sns.heatmap(corr_matrix_plot, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0, cbar_kws={"shrink": 0.8},
                    xticklabels=False, yticklabels=False, ax=ax,
                    vmin=-1, vmax=1)
            
            ax.set_title(f'Output-Output Correlation Matrix ({len(output_cols)} variables)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(corr_dir / 'output_output_correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    Error creating output-output heatmap: {e}")
            plt.close('all')

    def _plot_full_correlation_matrix(self, corr_matrix, input_cols, output_cols, corr_dir):
        """Plot full correlation matrix with variable type labels."""
        try:
            fig, ax = plt.subplots(figsize=(18, 16))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Plot heatmap
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0, cbar_kws={"shrink": 0.8},
                    xticklabels=False, yticklabels=False, ax=ax,
                    vmin=-1, vmax=1)
            
            # Add separating lines between input and output sections
            n_inputs = len(input_cols)
            if n_inputs > 0 and n_inputs < len(corr_matrix):
                ax.axhline(y=n_inputs, color='black', linewidth=2)
                ax.axvline(x=n_inputs, color='black', linewidth=2)
            
            # Add labels showing sections
            ax.text(n_inputs/2, -1, 'INPUTS', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='red')
            ax.text(n_inputs + len(output_cols)/2, -1, 'OUTPUTS', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='blue')
            
            ax.set_title(f'Full Correlation Matrix\n({n_inputs} inputs, {len(output_cols)} outputs)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(corr_dir / 'full_correlation_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    Error creating full correlation matrix: {e}")
            plt.close('all')

    def _plot_correlation_histogram(self, corr_df, corr_dir, prefix):
        """Plot histogram of correlation strengths."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # All correlations
            ax1 = axes[0]
            ax1.hist(corr_df['Correlation'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
            ax1.set_xlabel('Correlation Coefficient', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title('Distribution of All Correlations', fontsize=11, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3, axis='y')
            
            # Absolute correlations
            ax2 = axes[1]
            ax2.hist(corr_df['Abs_Correlation'], bins=50, alpha=0.7, edgecolor='black', color='coral')
            ax2.axvline(corr_df['Abs_Correlation'].median(), color='green', linestyle='--', 
                    linewidth=2, label=f"Median: {corr_df['Abs_Correlation'].median():.3f}")
            ax2.set_xlabel('Absolute Correlation', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title('Distribution of Correlation Strengths', fontsize=11, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3, axis='y')
            
            # Overall title
            sig_pct = 100 * corr_df['Significant'].mean()
            plt.suptitle(f'Correlation Distribution - {prefix.replace("_", "-").title()}\n' + 
                        f'({len(corr_df)} pairs, {sig_pct:.1f}% significant at p<0.05)',
                        fontsize=13, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            plt.savefig(corr_dir / f'{prefix}_correlation_histogram.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    Error creating correlation histogram: {e}")
            plt.close('all')

    def _sample_variables_for_correlation(self, numeric_cols: List[str]) -> List[str]:
        """Sample variables intelligently for correlation analysis (legacy method - kept for compatibility)."""
        # This method is now replaced by _smart_sample_outputs and _get_input_columns
        # but kept for backward compatibility if called elsewhere
        
        # Include all input variables
        input_patterns = ['Q_flow_total', 'T_Air', 'T_ext']
        input_vars = [col for col in numeric_cols 
                    if any(pattern in col for pattern in input_patterns)]
        
        sampled = input_vars.copy()
        
        # Include variables from first 10-20 CDUs
        cdu_count = 0
        target_cdus = 15
        
        for cdu_idx in range(1, self.num_cdus + 1):
            if cdu_count >= target_cdus:
                break
            
            cdu_vars = [col for col in numeric_cols 
                    if (f'Block[{cdu_idx}]' in col or f'computeBlock_{cdu_idx}' in col) 
                    and col not in sampled]
            
            if cdu_vars:
                sampled.extend(cdu_vars)
                cdu_count += 1
        
        # Add datacenter-level variables
        dc_vars = [col for col in numeric_cols 
                if 'datacenter' in col.lower() and col not in sampled]
        sampled.extend(dc_vars)
        
        return sampled[:200]  # Cap at 200
    def _sample_variables_for_correlation(self, numeric_cols: List[str]) -> List[str]:
        """Sample variables intelligently for correlation analysis."""
        # Include variables from first 10-20 CDUs
        sampled = []
        cdu_count = 0
        target_cdus = 15
        
        for cdu_idx in range(1, self.num_cdus + 1):
            if cdu_count >= target_cdus:
                break
            
            cdu_vars = [col for col in numeric_cols 
                       if f'Block[{cdu_idx}]' in col or f'computeBlock_{cdu_idx}' in col]
            
            if cdu_vars:
                sampled.extend(cdu_vars)
                cdu_count += 1
        
        # Add datacenter-level variables
        dc_vars = [col for col in numeric_cols if 'datacenter' in col.lower()]
        sampled.extend(dc_vars)
        
        return sampled[:200]  # Cap at 200
    
    def detect_outliers(self):
        """Detect outliers using parallel processing."""
        print("Detecting outliers...")
        
        # Prepare column data as tuples (column_name, series)
        columns_to_process = [
            (col, self.data[col]) 
            for col in self.data.columns 
            if col not in ['time', 'hue'] and self.data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        
        # Process in parallel using the global function
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(_detect_outliers_for_column, columns_to_process))
        
        # Filter out None results
        outlier_summary = [r for r in results if r is not None]
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df.to_csv(self.summary_dir / 'outlier_detection.csv', index=False)
            print(f"Outlier detection completed for {len(outlier_summary)} variables")
        else:
            print("No outliers detected or all processing failed")
    
    def analyze_missing_data(self):
        """Analyze missing data patterns."""
        print("Analyzing missing data...")
        
        missing_summary = []
        
        for col in self.data.columns:
            total = len(self.data)
            missing = self.data[col].isna().sum()
            missing_pct = 100 * missing / total
            
            missing_summary.append({
                'Variable': col,
                'Total_Points': total,
                'Missing_Count': int(missing),
                'Missing_%': missing_pct
            })
        
        missing_df = pd.DataFrame(missing_summary)
        missing_df = missing_df.sort_values('Missing_%', ascending=False)
        missing_df.to_csv(self.summary_dir / 'missing_data_analysis.csv', index=False)
        
        # Plot missing data
        vars_with_missing = missing_df[missing_df['Missing_%'] > 0]
        
        if not vars_with_missing.empty:
            try:
                fig, ax = plt.subplots(figsize=(12, max(6, len(vars_with_missing) * 0.2)))
                ax.barh(range(len(vars_with_missing)), vars_with_missing['Missing_%'])
                ax.set_yticks(range(len(vars_with_missing)))
                ax.set_yticklabels([v[:40] for v in vars_with_missing['Variable']], fontsize=8)
                ax.set_xlabel('Missing Data (%)')
                ax.set_title('Missing Data Analysis', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3, axis='x')
                plt.tight_layout()
                plt.savefig(self.summary_dir / 'missing_data.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating missing data plot: {e}")
                plt.close('all')
        
        print(f"Missing data analysis completed")
    

