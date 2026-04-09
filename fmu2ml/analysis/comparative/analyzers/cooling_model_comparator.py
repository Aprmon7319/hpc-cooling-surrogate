"""
Cooling Model Comparator.

Main analyzer for comparing cooling system behaviors across
different HPC data center configurations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from raps.config import ConfigManager

from .system_profiler import SystemProfiler
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class CoolingModelComparator:
    """
    Compares cooling model behaviors across multiple data center systems.
    
    Provides unified analysis framework for understanding differences
    in cooling efficiency, thermal dynamics, and scaling characteristics.
    """
    
    def __init__(
        self,
        system_names: List[str],
        n_workers: int = 4,
        **config_overrides
    ):
        """
        Initialize the cooling model comparator.
        
        Args:
            system_names: List of system names to compare
            n_workers: Number of parallel workers for data processing
            **config_overrides: Additional configuration overrides
        """
        self.system_names = system_names
        self.n_workers = n_workers
        self.config_overrides = config_overrides
        
        # Initialize profilers for each system
        self.profilers = {
            name: SystemProfiler(name, **config_overrides)
            for name in system_names
        }
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Storage for loaded data and computed metrics
        self.system_data: Dict[str, pd.DataFrame] = {}
        self.system_metrics: Dict[str, Dict] = {}
        self.comparison_results: Dict[str, Any] = {}
        
        logger.info(f"CoolingModelComparator initialized for systems: {system_names}")
    
    def load_data(
        self,
        data_paths: Dict[str, str],
        sample_size: Optional[int] = None
    ) -> None:
        """
        Load simulation data for each system.
        
        Args:
            data_paths: Dictionary mapping system names to data file paths
            sample_size: Optional limit on number of samples per system
        """
        logger.info("Loading data for all systems...")
        
        for system_name in self.system_names:
            if system_name not in data_paths:
                logger.warning(f"No data path provided for {system_name}, skipping")
                continue
            
            data_path = Path(data_paths[system_name])
            
            if not data_path.exists():
                logger.warning(f"Data file not found for {system_name}: {data_path}")
                continue
            
            logger.info(f"Loading data for {system_name} from {data_path}")
            
            if data_path.suffix == '.parquet':
                data = pd.read_parquet(data_path, engine='pyarrow')
            else:
                data = pd.read_csv(data_path, engine='c')
            
            if sample_size and len(data) > sample_size:
                # Sample uniformly to preserve distribution
                indices = np.linspace(0, len(data) - 1, sample_size, dtype=int)
                data = data.iloc[indices].reset_index(drop=True)
            
            self.system_data[system_name] = data
            logger.info(f"  Loaded {len(data)} samples for {system_name}")
    
    def generate_data(
        self,
        n_samples: int = 500,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        transition_steps: int = 60,
        stabilization_hours: int = 2
    ) -> None:
        """
        Generate simulation data for all systems.
        
        Args:
            n_samples: Number of operating conditions to sample
            input_ranges: Custom input ranges for data generation
            transition_steps: Steps for smooth transitions
            stabilization_hours: Hours for stabilization phase
        """
        from fmu2ml.analysis.input_output_relations.analyzers import DataGenerator
        
        logger.info(f"Generating {n_samples} samples for each system...")
        
        for system_name in self.system_names:
            logger.info(f"Generating data for {system_name}...")
            
            generator = DataGenerator(
                system_name=system_name,
                n_workers=self.n_workers,
                **self.config_overrides
            )
            
            data = generator.generate_sensitivity_data(
                n_samples=n_samples,
                input_ranges=input_ranges,
                transition_steps=transition_steps,
                stabilization_hours=stabilization_hours
            )
            
            self.system_data[system_name] = data
            logger.info(f"  Generated {len(data)} samples for {system_name}")
    
    def compute_metrics(self, time_step: float = 1.0) -> Dict[str, Dict]:
        """
        Compute comprehensive metrics for all loaded systems.
        
        Args:
            time_step: Time step in seconds for dynamic metrics
            
        Returns:
            Dictionary of metrics per system
        """
        logger.info("Computing metrics for all systems...")
        
        for system_name in self.system_names:
            if system_name not in self.system_data:
                logger.warning(f"No data loaded for {system_name}, skipping metrics")
                continue
            
            data = self.system_data[system_name]
            config = self.profilers[system_name].config
            
            logger.info(f"Computing metrics for {system_name}...")
            
            metrics = self.metrics_calculator.compute_all_metrics(
                data, config, time_step
            )
            
            # Add system profile information
            metrics['profile'] = self.profilers[system_name].get_system_profile()
            
            self.system_metrics[system_name] = metrics
            
        return self.system_metrics
    
    def compare_efficiency(self) -> pd.DataFrame:
        """
        Compare cooling efficiency across systems.
        
        Returns:
            DataFrame with efficiency comparison
        """
        efficiency_data = []
        
        for system_name, metrics in self.system_metrics.items():
            row = {
                'system': system_name,
                'num_cdus': metrics['system_info']['num_cdus'],
                'cooling_efficiency': metrics['system_info']['cooling_efficiency']
            }
            row.update(metrics.get('efficiency', {}))
            efficiency_data.append(row)
        
        df = pd.DataFrame(efficiency_data)
        self.comparison_results['efficiency'] = df
        return df
    
    def compare_thermal_performance(self) -> pd.DataFrame:
        """
        Compare thermal performance across systems.
        
        Returns:
            DataFrame with thermal comparison
        """
        thermal_data = []
        
        for system_name, metrics in self.system_metrics.items():
            row = {
                'system': system_name,
                'num_cdus': metrics['system_info']['num_cdus']
            }
            row.update(metrics.get('thermal', {}))
            thermal_data.append(row)
        
        df = pd.DataFrame(thermal_data)
        self.comparison_results['thermal'] = df
        return df
    
    def compare_flow_characteristics(self) -> pd.DataFrame:
        """
        Compare flow characteristics across systems.
        
        Returns:
            DataFrame with flow comparison
        """
        flow_data = []
        
        for system_name, metrics in self.system_metrics.items():
            row = {
                'system': system_name,
                'num_cdus': metrics['system_info']['num_cdus']
            }
            row.update(metrics.get('flow', {}))
            flow_data.append(row)
        
        df = pd.DataFrame(flow_data)
        self.comparison_results['flow'] = df
        return df
    
    def compare_dynamic_response(self) -> pd.DataFrame:
        """
        Compare dynamic response characteristics across systems.
        
        Returns:
            DataFrame with dynamic response comparison
        """
        dynamic_data = []
        
        for system_name, metrics in self.system_metrics.items():
            row = {
                'system': system_name,
                'num_cdus': metrics['system_info']['num_cdus']
            }
            row.update(metrics.get('dynamic', {}))
            dynamic_data.append(row)
        
        df = pd.DataFrame(dynamic_data)
        self.comparison_results['dynamic'] = df
        return df
    
    def compare_sensitivity(self) -> pd.DataFrame:
        """
        Compare input-output sensitivities across systems.
        
        Returns:
            DataFrame with sensitivity comparison
        """
        sensitivity_data = []
        
        for system_name, metrics in self.system_metrics.items():
            row = {
                'system': system_name,
                'num_cdus': metrics['system_info']['num_cdus']
            }
            row.update(metrics.get('sensitivity', {}))
            sensitivity_data.append(row)
        
        df = pd.DataFrame(sensitivity_data)
        self.comparison_results['sensitivity'] = df
        return df
    
    def compute_normalized_comparison(self) -> pd.DataFrame:
        """
        Compute normalized metrics for fair comparison across scales.
        
        Returns:
            DataFrame with normalized comparison metrics
        """
        normalized_data = []
        
        for system_name, metrics in self.system_metrics.items():
            num_cdus = metrics['system_info']['num_cdus']
            
            row = {
                'system': system_name,
                'num_cdus': num_cdus,
                # Normalized efficiency metrics
                'cdup_power_per_cdu_kw': metrics.get('efficiency', {}).get(
                    'cdup_power_per_cdu_kw', np.nan
                ),
                'heat_load_per_cdu_kw': metrics.get('efficiency', {}).get(
                    'heat_load_per_cdu_kw', np.nan
                ),
                'cooling_power_ratio': metrics.get('efficiency', {}).get(
                    'cooling_power_ratio', np.nan
                ),
                # Normalized flow metrics
                'sec_flow_per_cdu_gpm': metrics.get('flow', {}).get(
                    'sec_flow_per_cdu_gpm', np.nan
                ),
                'prim_flow_per_cdu_gpm': metrics.get('flow', {}).get(
                    'prim_flow_per_cdu_gpm', np.nan
                ),
                # Thermal metrics (already comparable)
                'mean_delta_t_c': metrics.get('thermal', {}).get(
                    'mean_delta_t_c', np.nan
                ),
                'mean_rack_return_temp_c': metrics.get('thermal', {}).get(
                    'mean_rack_return_temp_c', np.nan
                ),
                # Dynamic metrics
                'mean_temp_rate_c_per_s': metrics.get('dynamic', {}).get(
                    'mean_temp_rate_c_per_s', np.nan
                ),
                'thermal_time_constant_approx_s': metrics.get('dynamic', {}).get(
                    'thermal_time_constant_approx_s', np.nan
                )
            }
            normalized_data.append(row)
        
        df = pd.DataFrame(normalized_data)
        self.comparison_results['normalized'] = df
        return df
    
    def run_full_comparison(
        self,
        data_paths: Optional[Dict[str, str]] = None,
        generate_data: bool = False,
        n_samples: int = 500,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete comparison analysis.
        
        Args:
            data_paths: Dictionary mapping system names to data paths
            generate_data: Whether to generate new simulation data
            n_samples: Number of samples for data generation
            input_ranges: Input ranges for data generation
            
        Returns:
            Dictionary of comparison DataFrames
        """
        logger.info("=" * 60)
        logger.info("Running Full Cooling Model Comparison")
        logger.info("=" * 60)
        
        # Load or generate data
        if generate_data:
            self.generate_data(n_samples=n_samples, input_ranges=input_ranges)
        elif data_paths:
            self.load_data(data_paths)
        else:
            logger.error("Either data_paths or generate_data=True must be provided")
            return {}
        
        # Compute metrics
        self.compute_metrics()
        
        # Run all comparisons
        results = {
            'efficiency': self.compare_efficiency(),
            'thermal': self.compare_thermal_performance(),
            'flow': self.compare_flow_characteristics(),
            'dynamic': self.compare_dynamic_response(),
            'sensitivity': self.compare_sensitivity(),
            'normalized': self.compute_normalized_comparison()
        }
        
        logger.info("=" * 60)
        logger.info("Comparison complete!")
        logger.info("=" * 60)
        
        return results
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        include_data: bool = False
    ) -> None:
        """
        Save comparison results to disk.
        
        Args:
            output_dir: Directory to save results
            include_data: Whether to also save raw data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison DataFrames
        for name, df in self.comparison_results.items():
            df.to_csv(output_dir / f"comparison_{name}_{timestamp}.csv", index=False)
        
        # Save metrics as JSON
        metrics_file = output_dir / f"system_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj
            
            json.dump(self.system_metrics, f, default=convert, indent=2)
        
        # Save system profiles
        for system_name, profiler in self.profilers.items():
            profile = profiler.get_system_profile()
            profile_file = output_dir / f"profile_{system_name}_{timestamp}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile, f, default=convert, indent=2)
        
        # Optionally save raw data
        if include_data:
            data_dir = output_dir / "data"
            data_dir.mkdir(exist_ok=True)
            for system_name, data in self.system_data.items():
                data.to_parquet(
                    data_dir / f"{system_name}_data_{timestamp}.parquet",
                    index=False
                )
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the comparison.
        
        Returns:
            Formatted summary report string
        """
        lines = [
            "=" * 80,
            "COOLING MODEL COMPARISON SUMMARY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Systems Compared: {', '.join(self.system_names)}",
            ""
        ]
        
        # System overview
        lines.append("SYSTEM OVERVIEW")
        lines.append("-" * 40)
        for system_name in self.system_names:
            if system_name in self.system_metrics:
                info = self.system_metrics[system_name]['system_info']
                lines.append(f"  {system_name}:")
                lines.append(f"    - CDUs: {info['num_cdus']}")
                lines.append(f"    - Nodes/Rack: {info['nodes_per_rack']}")
                lines.append(f"    - GPUs/Node: {info['gpus_per_node']}")
                lines.append(f"    - Cooling Efficiency: {info['cooling_efficiency']}")
        lines.append("")
        
        # Efficiency comparison
        if 'efficiency' in self.comparison_results:
            lines.append("EFFICIENCY COMPARISON")
            lines.append("-" * 40)
            df = self.comparison_results['efficiency']
            for _, row in df.iterrows():
                lines.append(f"  {row['system']}:")
                if 'mean_cdup_power_kw' in row:
                    lines.append(f"    - Mean CDUP Power: {row['mean_cdup_power_kw']:.2f} kW")
                if 'cooling_power_ratio' in row:
                    lines.append(f"    - Cooling Power Ratio: {row['cooling_power_ratio']:.4f}")
            lines.append("")
        
        # Thermal comparison
        if 'thermal' in self.comparison_results:
            lines.append("THERMAL PERFORMANCE")
            lines.append("-" * 40)
            df = self.comparison_results['thermal']
            for _, row in df.iterrows():
                lines.append(f"  {row['system']}:")
                if 'mean_rack_return_temp_c' in row:
                    lines.append(f"    - Mean Return Temp: {row['mean_rack_return_temp_c']:.2f} °C")
                if 'mean_delta_t_c' in row:
                    lines.append(f"    - Mean ΔT: {row['mean_delta_t_c']:.2f} °C")
            lines.append("")
        
        # Normalized comparison
        if 'normalized' in self.comparison_results:
            lines.append("NORMALIZED COMPARISON (Per-CDU Metrics)")
            lines.append("-" * 40)
            df = self.comparison_results['normalized']
            for _, row in df.iterrows():
                lines.append(f"  {row['system']}:")
                if 'cdup_power_per_cdu_kw' in row and not pd.isna(row['cdup_power_per_cdu_kw']):
                    lines.append(f"    - CDUP Power/CDU: {row['cdup_power_per_cdu_kw']:.2f} kW")
                if 'sec_flow_per_cdu_gpm' in row and not pd.isna(row['sec_flow_per_cdu_gpm']):
                    lines.append(f"    - Flow/CDU: {row['sec_flow_per_cdu_gpm']:.2f} GPM")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
