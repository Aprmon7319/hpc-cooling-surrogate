"""
Rate of Change Analyzer - Temporal Derivatives Analysis

Analyzes how outputs respond to:
1. Input levels (absolute values)
2. Input rates (derivatives/changes)
3. Comparison of level vs rate effects
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.metrics import r2_score, mutual_info_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import KBinsDiscretizer
from typing import Dict, List, Tuple, Optional
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class RateOfChangeAnalyzer:
    """
    Analyzes temporal derivative effects of inputs on outputs.
    Compares whether outputs respond more to input levels or input changes.
    """
    
    def __init__(
        self,
        system_name: str = 'marconi100',
        n_workers: int = 8,
        threads_per_worker: int = 1,
        memory_limit: str = '5GB',
        **config_overrides
    ):
        """
        Initialize the rate of change analyzer.
        
        Args:
            system_name: System configuration name (e.g., 'marconi100', 'leonardo')
            n_workers: Number of Dask workers for parallel analysis
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            **config_overrides: Additional configuration overrides
        """
        self.system_name = system_name
        
        # Load system configuration
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        # Get number of CDUs from config
        self.num_cdus = self.config.get('NUM_CDUS', self.config.get('num_cdus', 49))
        
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        # Additional datacenter-level outputs
        self.datacenter_outputs = [
            'V_flow_prim_GPM_dc',  # Datacenter total
            'htc'  # Heat transfer coefficient per CDU
        ]
        
        self.client = None
        
        logger.info(f"RateOfChangeAnalyzer initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client for analysis tasks."""
        if self.client is None:
            logger.info("Initializing Dask cluster for rate of change analysis...")
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                processes=True,
                silence_logs=logging.WARNING
            )
            self.client = Client(cluster)
            logger.info(f"Dask client initialized: {self.client.dashboard_link}")
        return self.client
    
    def _close_dask_client(self):
        """Close Dask distributed client."""
        if self.client is not None:
            logger.info("Closing Dask cluster...")
            self.client.close()
            self.client = None
            logger.info("Dask client closed")
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare data by aggregating CDU inputs and outputs.
        
        Args:
            data: Raw simulation data with all CDU columns
            
        Returns:
            Dictionary with aggregated data per input-output pair
        """
        logger.info("Preparing data for rate of change analysis...")
        
        prepared_data = {
            'inputs': {},
            'outputs': {},
            'time': None
        }
        
        # Extract time if available
        if 'time' in data.columns:
            prepared_data['time'] = data['time'].values
        else:
            # Create time index
            prepared_data['time'] = np.arange(len(data))
        
        # Aggregate inputs across CDUs
        for cdu_idx in range(1, self.num_cdus + 1):
            q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
            
            if q_flow_col in data.columns:
                if cdu_idx == 1:
                    prepared_data['inputs']['Q_flow'] = data[q_flow_col].values.reshape(-1, 1)
                    prepared_data['inputs']['T_Air'] = data[t_air_col].values.reshape(-1, 1)
                else:
                    prepared_data['inputs']['Q_flow'] = np.hstack([
                        prepared_data['inputs']['Q_flow'],
                        data[q_flow_col].values.reshape(-1, 1)
                    ])
                    prepared_data['inputs']['T_Air'] = np.hstack([
                        prepared_data['inputs']['T_Air'],
                        data[t_air_col].values.reshape(-1, 1)
                    ])
        
        # T_ext is common for all CDUs
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        if t_ext_col in data.columns:
            prepared_data['inputs']['T_ext'] = data[t_ext_col].values
        
        # Aggregate CDU-level outputs
        for cdu_idx in range(1, self.num_cdus + 1):
            for output_var in self.output_vars:
                output_col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{output_var}'
                
                if output_col in data.columns:
                    if output_var not in prepared_data['outputs']:
                        prepared_data['outputs'][output_var] = data[output_col].values.reshape(-1, 1)
                    else:
                        prepared_data['outputs'][output_var] = np.hstack([
                            prepared_data['outputs'][output_var],
                            data[output_col].values.reshape(-1, 1)
                        ])
            
            # Heat transfer coefficient per CDU
            htc_col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cabinet[1].summary.htc'
            if htc_col in data.columns:
                if 'htc' not in prepared_data['outputs']:
                    prepared_data['outputs']['htc'] = data[htc_col].values.reshape(-1, 1)
                else:
                    prepared_data['outputs']['htc'] = np.hstack([
                        prepared_data['outputs']['htc'],
                        data[htc_col].values.reshape(-1, 1)
                    ])
        
        # Datacenter-level outputs
        dc_v_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if dc_v_flow_col in data.columns:
            prepared_data['outputs']['V_flow_prim_GPM_dc'] = data[dc_v_flow_col].values
        
        # PUE if available
        pue_col = 'pue'
        if pue_col in data.columns:
            prepared_data['outputs']['pue'] = data[pue_col].values
        
        logger.info(f"Data prepared: {len(prepared_data['inputs'])} inputs, {len(prepared_data['outputs'])} outputs")
        return prepared_data
    
    def compute_derivatives(
        self,
        prepared_data: Dict[str, np.ndarray],
        dt: float = 1.0,
        smooth_window: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Compute temporal derivatives (rates of change) for inputs and outputs.
        
        Args:
            prepared_data: Prepared data dictionary
            dt: Time step for derivative computation
            smooth_window: Window size for smoothing before differentiation
            
        Returns:
            Dictionary with derivatives for all variables
        """
        logger.info("Computing temporal derivatives...")
        
        derivatives = {
            'inputs': {},
            'outputs': {},
            'time': prepared_data['time'][1:] if prepared_data['time'] is not None else None
        }
        
        def compute_derivative(data: np.ndarray, dt: float, smooth_window: int) -> np.ndarray:
            """Compute smoothed derivative."""
            if data.ndim == 1:
                # Smooth the data first
                if smooth_window > 1:
                    kernel = np.ones(smooth_window) / smooth_window
                    smoothed = np.convolve(data, kernel, mode='same')
                else:
                    smoothed = data
                # Compute derivative using central differences
                derivative = np.gradient(smoothed, dt)
                return derivative[:-1]  # Align with other arrays
            else:
                # Handle 2D arrays (CDU-wise data)
                derivatives = []
                for i in range(data.shape[1]):
                    if smooth_window > 1:
                        kernel = np.ones(smooth_window) / smooth_window
                        smoothed = np.convolve(data[:, i], kernel, mode='same')
                    else:
                        smoothed = data[:, i]
                    derivative = np.gradient(smoothed, dt)
                    derivatives.append(derivative[:-1])
                return np.column_stack(derivatives)
        
        # Compute input derivatives
        for input_name, input_data in prepared_data['inputs'].items():
            derivatives['inputs'][input_name] = compute_derivative(input_data, dt, smooth_window)
            # Also keep original (level) data aligned
            if input_data.ndim == 1:
                derivatives['inputs'][f'{input_name}_level'] = input_data[:-1]
            else:
                derivatives['inputs'][f'{input_name}_level'] = input_data[:-1, :]
        
        # Compute output derivatives
        for output_name, output_data in prepared_data['outputs'].items():
            derivatives['outputs'][output_name] = compute_derivative(output_data, dt, smooth_window)
            # Also keep original (level) data aligned
            if output_data.ndim == 1:
                derivatives['outputs'][f'{output_name}_level'] = output_data[:-1]
            else:
                derivatives['outputs'][f'{output_name}_level'] = output_data[:-1, :]
        
        logger.info(f"Computed derivatives for {len(derivatives['inputs'])} inputs, {len(derivatives['outputs'])} outputs")
        return derivatives
    
    @staticmethod
    def _compute_level_rate_metrics_static(
        input_level: np.ndarray,
        input_rate: np.ndarray,
        output_level: np.ndarray,
        output_rate: np.ndarray,
        input_name: str,
        output_name: str
    ) -> Dict[str, float]:
        """
        Compute metrics comparing level vs rate effects.
        Static method for parallel execution.
        """
        def flatten_and_clean(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Flatten arrays and remove NaN values."""
            if arr1.ndim > 1:
                flat1 = arr1.flatten()
            else:
                flat1 = arr1
            
            if arr2.ndim > 1:
                flat2 = arr2.flatten()
            else:
                flat2 = arr2
            
            # Repeat 1D arrays to match 2D if needed
            if len(flat1) != len(flat2):
                if len(flat1) < len(flat2):
                    flat1 = np.repeat(flat1, len(flat2) // len(flat1))
                else:
                    flat2 = np.repeat(flat2, len(flat1) // len(flat2))
            
            # Remove NaN
            mask = ~(np.isnan(flat1) | np.isnan(flat2) | np.isinf(flat1) | np.isinf(flat2))
            return flat1[mask], flat2[mask]
        
        try:
            # Level effect: corr(Input[t], Output[t])
            input_level_clean, output_level_clean = flatten_and_clean(input_level, output_level)
            
            if len(input_level_clean) < 10:
                return {}
            
            level_pearson_r, level_pearson_p = stats.pearsonr(input_level_clean, output_level_clean)
            level_spearman_r, level_spearman_p = stats.spearmanr(input_level_clean, output_level_clean)
            
            # Rate effect: corr(ΔInput[t], Output[t])
            input_rate_clean, output_level_clean2 = flatten_and_clean(input_rate, output_level)
            
            if len(input_rate_clean) < 10:
                rate_level_pearson_r = np.nan
                rate_level_spearman_r = np.nan
            else:
                rate_level_pearson_r, _ = stats.pearsonr(input_rate_clean, output_level_clean2)
                rate_level_spearman_r, _ = stats.spearmanr(input_rate_clean, output_level_clean2)
            
            # Rate-Rate effect: corr(ΔInput[t], ΔOutput[t])
            input_rate_clean2, output_rate_clean = flatten_and_clean(input_rate, output_rate)
            
            if len(input_rate_clean2) < 10:
                rate_rate_pearson_r = np.nan
                rate_rate_spearman_r = np.nan
            else:
                rate_rate_pearson_r, _ = stats.pearsonr(input_rate_clean2, output_rate_clean)
                rate_rate_spearman_r, _ = stats.spearmanr(input_rate_clean2, output_rate_clean)
            
            # Compute R² scores for level and rate models
            lr_level = LinearRegression()
            lr_level.fit(input_level_clean.reshape(-1, 1), output_level_clean)
            r2_level = r2_score(output_level_clean, lr_level.predict(input_level_clean.reshape(-1, 1)))
            
            if len(input_rate_clean) >= 10:
                lr_rate = LinearRegression()
                lr_rate.fit(input_rate_clean.reshape(-1, 1), output_level_clean2)
                r2_rate = r2_score(output_level_clean2, lr_rate.predict(input_rate_clean.reshape(-1, 1)))
            else:
                r2_rate = np.nan
            
            # Mutual information for level and rate
            discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
            
            input_level_discrete = discretizer.fit_transform(input_level_clean.reshape(-1, 1)).flatten()
            output_level_discrete = discretizer.fit_transform(output_level_clean.reshape(-1, 1)).flatten()
            mi_level = mutual_info_score(input_level_discrete, output_level_discrete)
            
            if len(input_rate_clean) >= 10:
                input_rate_discrete = discretizer.fit_transform(input_rate_clean.reshape(-1, 1)).flatten()
                output_level_discrete2 = discretizer.fit_transform(output_level_clean2.reshape(-1, 1)).flatten()
                mi_rate = mutual_info_score(input_rate_discrete, output_level_discrete2)
            else:
                mi_rate = np.nan
            
            # Determine dominant effect
            level_strength = abs(level_pearson_r) if not np.isnan(level_pearson_r) else 0
            rate_strength = abs(rate_level_pearson_r) if not np.isnan(rate_level_pearson_r) else 0
            
            if level_strength > rate_strength * 1.2:
                dominant_effect = 'level'
            elif rate_strength > level_strength * 1.2:
                dominant_effect = 'rate'
            else:
                dominant_effect = 'mixed'
            
            return {
                'input': input_name,
                'output': output_name,
                # Level effects
                'level_pearson_r': level_pearson_r,
                'level_pearson_p': level_pearson_p,
                'level_spearman_r': level_spearman_r,
                'level_r2': r2_level,
                'level_mi': mi_level,
                # Rate effects on output level
                'rate_level_pearson_r': rate_level_pearson_r,
                'rate_level_spearman_r': rate_level_spearman_r,
                'rate_level_r2': r2_rate,
                'rate_level_mi': mi_rate,
                # Rate-Rate effects
                'rate_rate_pearson_r': rate_rate_pearson_r,
                'rate_rate_spearman_r': rate_rate_spearman_r,
                # Summary
                'level_strength': level_strength,
                'rate_strength': rate_strength,
                'dominant_effect': dominant_effect,
                'effect_ratio': rate_strength / level_strength if level_strength > 0 else np.inf
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics for {input_name} -> {output_name}: {e}")
            return {}
    
    def analyze_all_level_rate_effects(
        self,
        derivatives_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze level vs rate effects for all input-output pairs using Dask.
        
        Args:
            derivatives_data: Data with computed derivatives
            
        Returns:
            DataFrame with level and rate effect metrics for all pairs
        """
        logger.info("Analyzing level vs rate effects for all pairs...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        task_info = []
        
        for input_name in self.input_vars:
            input_level = derivatives_data['inputs'].get(f'{input_name}_level')
            input_rate = derivatives_data['inputs'].get(input_name)
            
            if input_level is None or input_rate is None:
                continue
            
            for output_name in list(self.output_vars) + self.datacenter_outputs:
                output_level = derivatives_data['outputs'].get(f'{output_name}_level')
                output_rate = derivatives_data['outputs'].get(output_name)
                
                if output_level is None or output_rate is None:
                    continue
                
                task = delayed(self._compute_level_rate_metrics_static)(
                    input_level, input_rate, output_level, output_rate,
                    input_name, output_name
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        logger.info(f"Executing {len(delayed_tasks)} level-rate analysis tasks in parallel...")
        results_list = dask.compute(*delayed_tasks)
        
        results = [r for r in results_list if r]
        results_df = pd.DataFrame(results)
        
        logger.info(f"Computed level-rate metrics for {len(results_df)} pairs")
        return results_df
    
    @staticmethod
    def _compute_lagged_correlation_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        max_lag: int,
        input_name: str,
        output_name: str
    ) -> Dict:
        """
        Compute lagged cross-correlation to find response time.
        Static method for parallel execution.
        """
        def flatten_and_clean(arr: np.ndarray) -> np.ndarray:
            if arr.ndim > 1:
                return arr.mean(axis=1)  # Average across CDUs
            return arr
        
        try:
            input_clean = flatten_and_clean(input_data)
            output_clean = flatten_and_clean(output_data)
            
            # Remove NaN
            mask = ~(np.isnan(input_clean) | np.isnan(output_clean))
            input_clean = input_clean[mask]
            output_clean = output_clean[mask]
            
            if len(input_clean) < max_lag * 2:
                return {}
            
            # Normalize
            input_norm = (input_clean - np.mean(input_clean)) / (np.std(input_clean) + 1e-8)
            output_norm = (output_clean - np.mean(output_clean)) / (np.std(output_clean) + 1e-8)
            
            # Compute cross-correlation for different lags
            correlations = []
            lags = range(-max_lag, max_lag + 1)
            
            for lag in lags:
                if lag < 0:
                    corr = np.corrcoef(input_norm[:lag], output_norm[-lag:])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(input_norm[lag:], output_norm[:-lag])[0, 1]
                else:
                    corr = np.corrcoef(input_norm, output_norm)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            correlations = np.array(correlations)
            
            # Find optimal lag (max absolute correlation)
            best_idx = np.argmax(np.abs(correlations))
            optimal_lag = list(lags)[best_idx]
            max_correlation = correlations[best_idx]
            
            # Zero-lag correlation for reference
            zero_lag_idx = list(lags).index(0)
            zero_lag_corr = correlations[zero_lag_idx]
            
            return {
                'input': input_name,
                'output': output_name,
                'optimal_lag': optimal_lag,
                'max_correlation': max_correlation,
                'zero_lag_correlation': zero_lag_corr,
                'lag_improvement': abs(max_correlation) - abs(zero_lag_corr),
                'correlations': correlations.tolist(),
                'lags': list(lags)
            }
            
        except Exception as e:
            logger.error(f"Error computing lagged correlation for {input_name} -> {output_name}: {e}")
            return {}
    
    def analyze_response_lags(
        self,
        prepared_data: Dict[str, np.ndarray],
        max_lag: int = 30
    ) -> pd.DataFrame:
        """
        Analyze response time lags between inputs and outputs.
        
        Args:
            prepared_data: Prepared data dictionary
            max_lag: Maximum lag to consider
            
        Returns:
            DataFrame with lag analysis results
        """
        logger.info(f"Analyzing response lags (max_lag={max_lag})...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'].get(input_name)
            if input_data is None:
                continue
            
            for output_name in list(self.output_vars) + self.datacenter_outputs:
                output_data = prepared_data['outputs'].get(output_name)
                if output_data is None:
                    continue
                
                task = delayed(self._compute_lagged_correlation_static)(
                    input_data, output_data, max_lag, input_name, output_name
                )
                delayed_tasks.append(task)
        
        logger.info(f"Executing {len(delayed_tasks)} lag analysis tasks...")
        results_list = dask.compute(*delayed_tasks)
        
        # Separate correlation arrays from main results
        results = []
        correlation_data = []
        
        for r in results_list:
            if r:
                corr_data = {
                    'input': r['input'],
                    'output': r['output'],
                    'correlations': r.pop('correlations'),
                    'lags': r.pop('lags')
                }
                correlation_data.append(corr_data)
                results.append(r)
        
        results_df = pd.DataFrame(results)
        
        # Store correlation data as attribute for plotting
        self._lag_correlation_data = correlation_data
        
        logger.info(f"Computed lag analysis for {len(results_df)} pairs")
        return results_df
    
    def get_lag_correlation_data(self) -> List[Dict]:
        """Return detailed lag correlation data for visualization."""
        return getattr(self, '_lag_correlation_data', [])
    
    def summarize_dynamic_effects(
        self,
        level_rate_df: pd.DataFrame,
        lag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create summary of dynamic effects combining level/rate and lag analysis.
        
        Args:
            level_rate_df: Level vs rate analysis results
            lag_df: Lag analysis results
            
        Returns:
            Combined summary DataFrame
        """
        logger.info("Creating dynamic effects summary...")
        
        # Merge the dataframes
        summary = level_rate_df.merge(
            lag_df[['input', 'output', 'optimal_lag', 'max_correlation', 'lag_improvement']],
            on=['input', 'output'],
            how='left'
        )
        
        # Add interpretation columns
        summary['response_type'] = summary.apply(
            lambda row: 'fast' if abs(row['optimal_lag']) <= 5 else 
                       ('medium' if abs(row['optimal_lag']) <= 15 else 'slow'),
            axis=1
        )
        
        summary['dynamics_summary'] = summary.apply(
            lambda row: f"{row['dominant_effect']}_effect, {row['response_type']}_response, lag={row['optimal_lag']}",
            axis=1
        )
        
        return summary
