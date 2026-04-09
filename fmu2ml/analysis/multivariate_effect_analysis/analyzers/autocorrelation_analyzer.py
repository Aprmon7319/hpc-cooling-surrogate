"""
Autocorrelation Analysis for FMU input-output relationships.

Provides methods to analyze temporal dependencies in outputs,
including ACF, PACF, and cross-correlation with inputs.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import acf, pacf, ccf
from typing import Dict, List, Tuple, Optional
import logging
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class AutocorrelationAnalyzer:
    """
    Analyzes autocorrelation and cross-correlation patterns in
    FMU input-output time series data.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        max_lag: int = 50,
        confidence_level: float = 0.95,
        compute_pacf: bool = True,
        compute_ccf: bool = True,
        **config_overrides
    ):
        """
        Initialize the autocorrelation analyzer.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            max_lag: Maximum lag for autocorrelation
            confidence_level: Confidence level for significance bands
            compute_pacf: Whether to compute partial autocorrelation
            compute_ccf: Whether to compute cross-correlation
            **config_overrides: Additional configuration overrides
        """
        self.system_name = system_name
        
        # Load system configuration
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        self.num_cdus = self.config.get('NUM_CDUS', self.config.get('num_cdus', 49))
        
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.max_lag = max_lag
        self.confidence_level = confidence_level
        self.compute_pacf = compute_pacf
        self.compute_ccf = compute_ccf
        
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        self.datacenter_output_vars = [
            'V_flow_prim_GPM_datacenter',
            'pue'
        ]
        
        self.client = None
        
        logger.info(f"AutocorrelationAnalyzer initialized for system: {system_name}")
        logger.info(f"Max lag: {max_lag}, Confidence level: {confidence_level}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client."""
        if self.client is None:
            logger.info("Initializing Dask cluster for autocorrelation analysis...")
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
            Dictionary with aggregated input and output arrays
        """
        logger.info("Preparing data for autocorrelation analysis...")
        
        n_samples = len(data)
        prepared = {'inputs': {}, 'outputs': {}, 'datacenter_outputs': {}}
        
        # Aggregate inputs
        for input_var in self.input_vars:
            if input_var == 'Q_flow':
                q_flow_cols = [
                    col for col in data.columns 
                    if 'Q_flow_total' in col or 'Q_flow' in col.split('_')[-1]
                ]
                if q_flow_cols:
                    prepared['inputs']['Q_flow'] = data[q_flow_cols].mean(axis=1).values / 1000
                    
            elif input_var == 'T_Air':
                t_air_cols = [col for col in data.columns if 'T_Air' in col]
                if t_air_cols:
                    prepared['inputs']['T_Air'] = data[t_air_cols].mean(axis=1).values
                    
            elif input_var == 'T_ext':
                t_ext_cols = [col for col in data.columns if 'T_ext' in col]
                if t_ext_cols:
                    prepared['inputs']['T_ext'] = data[t_ext_cols].mean(axis=1).values
        
        # Map output variable patterns
        output_patterns = {
            'V_flow_prim_GPM': 'V_flow_prim_GPM',
            'V_flow_sec_GPM': 'V_flow_sec_GPM', 
            'W_flow_CDUP_kW': 'W_flow_CDUP_kW',
            'T_prim_s_C': 'T_prim_s_C',
            'T_prim_r_C': 'T_prim_r_C',
            'T_sec_s_C': 'T_sec_s_C',
            'T_sec_r_C': 'T_sec_r_C',
            'p_prim_s_psig': 'p_prim_s_psig',
            'p_prim_r_psig': 'p_prim_r_psig',
            'p_sec_s_psig': 'p_sec_s_psig',
            'p_sec_r_psig': 'p_sec_r_psig'
        }
        
        # Aggregate outputs (mean across CDUs)
        for output_var, pattern in output_patterns.items():
            matching_cols = [col for col in data.columns if pattern in col]
            if matching_cols:
                prepared['outputs'][output_var] = data[matching_cols].mean(axis=1).values
        
        # Datacenter-level outputs
        for dc_output in self.datacenter_output_vars:
            if dc_output in data.columns:
                prepared['datacenter_outputs'][dc_output] = data[dc_output].values
        
        prepared['n_samples'] = n_samples
        
        logger.info(f"Prepared data: {len(prepared['inputs'])} inputs, "
                   f"{len(prepared['outputs'])} outputs")
        
        return prepared
    
    @staticmethod
    def _compute_acf_pacf_static(
        var_name: str,
        var_type: str,
        data: np.ndarray,
        max_lag: int,
        confidence_level: float,
        compute_pacf: bool
    ) -> Dict:
        """
        Static method for parallel computation of ACF and PACF.
        
        Args:
            var_name: Variable name
            var_type: Variable type ('input' or 'output')
            data: Time series data
            max_lag: Maximum lag
            confidence_level: Confidence level for significance
            compute_pacf: Whether to compute PACF
            
        Returns:
            Dictionary with ACF/PACF results
        """
        result = {
            'variable': var_name,
            'type': var_type,
            'acf_values': [],
            'acf_conf_int': [],
            'pacf_values': [],
            'pacf_conf_int': [],
            'significant_acf_lags': [],
            'significant_pacf_lags': [],
            'first_insignificant_lag': np.nan,
            'persistence': np.nan,
            'ar_order_suggestion': 0
        }
        
        # Validate data
        mask = ~np.isnan(data)
        x = data[mask]
        
        n = len(x)
        if n < max_lag + 10:
            return result
        
        try:
            # Compute ACF with confidence intervals
            acf_values, acf_conf_int = acf(
                x, 
                nlags=max_lag, 
                alpha=1 - confidence_level,
                fft=True
            )
            
            result['acf_values'] = acf_values.tolist()
            result['acf_conf_int'] = acf_conf_int.tolist()
            
            # Compute significance threshold
            # Under null hypothesis, acf ~ N(0, 1/sqrt(n))
            z = stats.norm.ppf((1 + confidence_level) / 2)
            threshold = z / np.sqrt(n)
            
            # Find significant ACF lags
            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) > threshold:
                    result['significant_acf_lags'].append(lag)
            
            # Find first insignificant lag
            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) <= threshold:
                    result['first_insignificant_lag'] = lag
                    break
            
            # Compute persistence (sum of ACF values)
            result['persistence'] = np.sum(np.abs(acf_values[1:]))
            
            # Compute PACF
            if compute_pacf:
                try:
                    pacf_values, pacf_conf_int = pacf(
                        x,
                        nlags=min(max_lag, n // 2 - 1),
                        alpha=1 - confidence_level,
                        method='ywm'  # Yule-Walker with MLE
                    )
                    
                    result['pacf_values'] = pacf_values.tolist()
                    result['pacf_conf_int'] = pacf_conf_int.tolist()
                    
                    # Find significant PACF lags (for AR order)
                    for lag in range(1, len(pacf_values)):
                        if abs(pacf_values[lag]) > threshold:
                            result['significant_pacf_lags'].append(lag)
                    
                    # Suggest AR order based on PACF
                    if result['significant_pacf_lags']:
                        result['ar_order_suggestion'] = max(result['significant_pacf_lags'])
                    
                except Exception as e:
                    logger.warning(f"PACF computation failed for {var_name}: {e}")
            
        except Exception as e:
            logger.warning(f"ACF computation failed for {var_name}: {e}")
        
        return result
    
    @staticmethod
    def _compute_ccf_static(
        input_name: str,
        output_name: str,
        input_data: np.ndarray,
        output_data: np.ndarray,
        max_lag: int,
        confidence_level: float
    ) -> Dict:
        """
        Static method for parallel computation of cross-correlation.
        
        Args:
            input_name: Input variable name
            output_name: Output variable name
            input_data: Input time series
            output_data: Output time series
            max_lag: Maximum lag
            confidence_level: Confidence level
            
        Returns:
            Dictionary with CCF results
        """
        result = {
            'input': input_name,
            'output': output_name,
            'ccf_values': [],
            'lags': [],
            'peak_lag': np.nan,
            'peak_ccf': np.nan,
            'significant_lags': [],
            'lead_lag_relationship': 'none'
        }
        
        # Validate data
        mask = ~(np.isnan(input_data) | np.isnan(output_data))
        x = input_data[mask]
        y = output_data[mask]
        
        n = len(x)
        if n < max_lag + 10:
            return result
        
        try:
            # Standardize
            x = (x - np.mean(x)) / np.std(x)
            y = (y - np.mean(y)) / np.std(y)
            
            # Compute CCF for both positive and negative lags
            lags = list(range(-max_lag, max_lag + 1))
            ccf_values = []
            
            for lag in lags:
                if lag < 0:
                    # Negative lag: input leads output
                    ccf_val = np.corrcoef(x[:lag], y[-lag:])[0, 1]
                elif lag > 0:
                    # Positive lag: output leads input
                    ccf_val = np.corrcoef(x[lag:], y[:-lag])[0, 1]
                else:
                    # Lag 0: contemporaneous
                    ccf_val = np.corrcoef(x, y)[0, 1]
                
                ccf_values.append(ccf_val if not np.isnan(ccf_val) else 0)
            
            result['ccf_values'] = ccf_values
            result['lags'] = lags
            
            # Find peak CCF
            abs_ccf = [abs(c) for c in ccf_values]
            peak_idx = np.argmax(abs_ccf)
            result['peak_lag'] = lags[peak_idx]
            result['peak_ccf'] = ccf_values[peak_idx]
            
            # Significance threshold
            z = stats.norm.ppf((1 + confidence_level) / 2)
            threshold = z / np.sqrt(n)
            
            # Find significant lags
            for i, (lag, ccf_val) in enumerate(zip(lags, ccf_values)):
                if abs(ccf_val) > threshold:
                    result['significant_lags'].append(lag)
            
            # Determine lead-lag relationship
            if result['peak_lag'] < 0:
                result['lead_lag_relationship'] = 'input_leads'
            elif result['peak_lag'] > 0:
                result['lead_lag_relationship'] = 'output_leads'
            else:
                result['lead_lag_relationship'] = 'contemporaneous'
            
        except Exception as e:
            logger.warning(f"CCF computation failed for {input_name} -> {output_name}: {e}")
        
        return result
    
    def analyze_autocorrelation(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze autocorrelation for all variables.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            Tuple of (input ACF DataFrame, output ACF DataFrame)
        """
        logger.info("Analyzing autocorrelation patterns...")
        
        self._init_dask_client()
        
        inputs = prepared_data['inputs']
        outputs = prepared_data['outputs']
        all_outputs = {**outputs, **prepared_data.get('datacenter_outputs', {})}
        
        # Create delayed tasks for inputs
        input_tasks = []
        for input_name, input_data in inputs.items():
            task = delayed(self._compute_acf_pacf_static)(
                input_name,
                'input',
                input_data,
                self.max_lag,
                self.confidence_level,
                self.compute_pacf
            )
            input_tasks.append(task)
        
        # Create delayed tasks for outputs
        output_tasks = []
        for output_name, output_data in all_outputs.items():
            task = delayed(self._compute_acf_pacf_static)(
                output_name,
                'output',
                output_data,
                self.max_lag,
                self.confidence_level,
                self.compute_pacf
            )
            output_tasks.append(task)
        
        # Execute in parallel
        logger.info(f"Executing {len(input_tasks) + len(output_tasks)} ACF/PACF analyses...")
        input_results = self.client.compute(input_tasks, sync=True)
        output_results = self.client.compute(output_tasks, sync=True)
        
        # Store full results for visualization
        self._acf_results = {
            'inputs': input_results,
            'outputs': output_results
        }
        
        # Create summary DataFrames
        input_summary = []
        for r in input_results:
            input_summary.append({
                'variable': r['variable'],
                'first_insignificant_lag': r['first_insignificant_lag'],
                'persistence': r['persistence'],
                'ar_order_suggestion': r['ar_order_suggestion'],
                'n_significant_acf_lags': len(r['significant_acf_lags']),
                'n_significant_pacf_lags': len(r['significant_pacf_lags'])
            })
        
        output_summary = []
        for r in output_results:
            output_summary.append({
                'variable': r['variable'],
                'first_insignificant_lag': r['first_insignificant_lag'],
                'persistence': r['persistence'],
                'ar_order_suggestion': r['ar_order_suggestion'],
                'n_significant_acf_lags': len(r['significant_acf_lags']),
                'n_significant_pacf_lags': len(r['significant_pacf_lags'])
            })
        
        input_df = pd.DataFrame(input_summary)
        output_df = pd.DataFrame(output_summary)
        
        logger.info(f"Completed ACF/PACF analysis for {len(input_df)} inputs, {len(output_df)} outputs")
        
        return input_df, output_df
    
    def analyze_cross_correlation(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze cross-correlation between inputs and outputs.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            DataFrame with CCF results
        """
        logger.info("Analyzing cross-correlation patterns...")
        
        self._init_dask_client()
        
        inputs = prepared_data['inputs']
        outputs = prepared_data['outputs']
        all_outputs = {**outputs, **prepared_data.get('datacenter_outputs', {})}
        
        # Create delayed tasks
        tasks = []
        for input_name, input_data in inputs.items():
            for output_name, output_data in all_outputs.items():
                task = delayed(self._compute_ccf_static)(
                    input_name,
                    output_name,
                    input_data,
                    output_data,
                    self.max_lag,
                    self.confidence_level
                )
                tasks.append(task)
        
        # Execute in parallel
        logger.info(f"Executing {len(tasks)} CCF analyses...")
        results = self.client.compute(tasks, sync=True)
        
        # Store full results for visualization
        self._ccf_results = results
        
        # Create summary DataFrame
        summary = []
        for r in results:
            summary.append({
                'input': r['input'],
                'output': r['output'],
                'peak_lag': r['peak_lag'],
                'peak_ccf': r['peak_ccf'],
                'lead_lag_relationship': r['lead_lag_relationship'],
                'n_significant_lags': len(r['significant_lags'])
            })
        
        ccf_df = pd.DataFrame(summary)
        ccf_df = ccf_df.sort_values('peak_ccf', key=abs, ascending=False)
        
        logger.info(f"Completed CCF analysis for {len(ccf_df)} input-output pairs")
        
        return ccf_df
    
    def get_acf_data(self, variable: str, var_type: str = 'output') -> Dict:
        """
        Get ACF/PACF data for a specific variable.
        
        Args:
            variable: Variable name
            var_type: Variable type ('input' or 'output')
            
        Returns:
            Dictionary with ACF/PACF values and confidence intervals
        """
        if not hasattr(self, '_acf_results'):
            logger.warning("No ACF results available. Run analyze_autocorrelation first.")
            return {}
        
        results = self._acf_results.get('inputs' if var_type == 'input' else 'outputs', [])
        
        for r in results:
            if r['variable'] == variable:
                return {
                    'acf_values': r['acf_values'],
                    'acf_conf_int': r['acf_conf_int'],
                    'pacf_values': r['pacf_values'],
                    'pacf_conf_int': r['pacf_conf_int'],
                    'lags': list(range(len(r['acf_values'])))
                }
        
        return {}
    
    def get_ccf_data(self, input_name: str, output_name: str) -> Dict:
        """
        Get CCF data for a specific input-output pair.
        
        Args:
            input_name: Input variable name
            output_name: Output variable name
            
        Returns:
            Dictionary with CCF values and lags
        """
        if not hasattr(self, '_ccf_results'):
            logger.warning("No CCF results available. Run analyze_cross_correlation first.")
            return {}
        
        for r in self._ccf_results:
            if r['input'] == input_name and r['output'] == output_name:
                return {
                    'ccf_values': r['ccf_values'],
                    'lags': r['lags'],
                    'peak_lag': r['peak_lag'],
                    'peak_ccf': r['peak_ccf']
                }
        
        return {}
    
    def compute_temporal_recommendations(
        self,
        acf_output_df: pd.DataFrame,
        ccf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate recommendations for temporal modeling based on analysis.
        
        Args:
            acf_output_df: Output ACF summary DataFrame
            ccf_df: Cross-correlation DataFrame
            
        Returns:
            DataFrame with temporal modeling recommendations
        """
        logger.info("Computing temporal modeling recommendations...")
        
        recommendations = []
        
        # Analyze each output
        for _, output_row in acf_output_df.iterrows():
            output_name = output_row['variable']
            
            # Get relevant CCF data
            output_ccf = ccf_df[ccf_df['output'] == output_name]
            
            # Determine if temporal modeling is needed
            high_persistence = output_row['persistence'] > 10
            has_ar_component = output_row['ar_order_suggestion'] > 0
            has_lagged_inputs = any(output_ccf['peak_lag'].abs() > 0)
            
            # Recommendations
            rec = {
                'output': output_name,
                'needs_temporal_model': high_persistence or has_ar_component,
                'suggested_ar_order': output_row['ar_order_suggestion'],
                'persistence_score': output_row['persistence'],
                'has_lagged_input_effects': has_lagged_inputs,
                'recommendation': ''
            }
            
            # Generate recommendation text
            if high_persistence and has_ar_component:
                rec['recommendation'] = (
                    f"Use LSTM or sequence model with {output_row['ar_order_suggestion']} lags. "
                    f"High persistence ({output_row['persistence']:.1f}) indicates strong temporal dependencies."
                )
            elif has_ar_component:
                rec['recommendation'] = (
                    f"Consider AR({output_row['ar_order_suggestion']}) terms or short sequence input."
                )
            elif has_lagged_inputs:
                max_lag = output_ccf['peak_lag'].abs().max()
                rec['recommendation'] = (
                    f"Include lagged inputs up to lag {max_lag:.0f}. "
                    f"Standard feedforward with lagged features may suffice."
                )
            else:
                rec['recommendation'] = (
                    "No strong temporal dependencies detected. "
                    "Feedforward network should be sufficient."
                )
            
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
