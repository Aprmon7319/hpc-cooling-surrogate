"""
Lagged Effect Analysis (Distributed Lag Models) for FMU input-output relationships.

Provides methods to analyze how past input values affect current outputs,
determining optimal lag structure and memory length of the system.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import logging
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class LaggedEffectAnalyzer:
    """
    Analyzes lagged effects using distributed lag models to understand
    how past input values influence current outputs.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        max_lag: int = 30,
        min_lag: int = 0,
        lag_step: int = 1,
        cv_folds: int = 5,
        significance_threshold: float = 0.05,
        **config_overrides
    ):
        """
        Initialize the lagged effect analyzer.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            max_lag: Maximum lag to consider
            min_lag: Minimum lag to consider
            lag_step: Step size for lag analysis
            cv_folds: Cross-validation folds for optimal lag selection
            significance_threshold: p-value threshold for significance
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
        self.min_lag = min_lag
        self.lag_step = lag_step
        self.cv_folds = cv_folds
        self.significance_threshold = significance_threshold
        
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
        
        logger.info(f"LaggedEffectAnalyzer initialized for system: {system_name}")
        logger.info(f"Max lag: {max_lag}, Lag step: {lag_step}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client."""
        if self.client is None:
            logger.info("Initializing Dask cluster for lagged effect analysis...")
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
        logger.info("Preparing data for lagged effect analysis...")
        
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
    def _create_lagged_features(
        x: np.ndarray,
        max_lag: int,
        min_lag: int = 0,
        lag_step: int = 1
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Create lagged features from input array.
        
        Args:
            x: Input time series
            max_lag: Maximum lag
            min_lag: Minimum lag
            lag_step: Step between lags
            
        Returns:
            Tuple of (lagged feature matrix, list of lag values)
        """
        n = len(x)
        lags = list(range(min_lag, max_lag + 1, lag_step))
        
        # Initialize lagged matrix
        X_lagged = np.zeros((n - max_lag, len(lags)))
        
        for i, lag in enumerate(lags):
            X_lagged[:, i] = x[max_lag - lag:n - lag]
        
        return X_lagged, lags
    
    @staticmethod
    def _fit_distributed_lag_model_static(
        input_name: str,
        output_name: str,
        input_data: np.ndarray,
        output_data: np.ndarray,
        max_lag: int,
        min_lag: int,
        lag_step: int,
        cv_folds: int
    ) -> Dict:
        """
        Static method for parallel execution of distributed lag model fitting.
        
        Args:
            input_name: Name of input variable
            output_name: Name of output variable
            input_data: Input time series
            output_data: Output time series
            max_lag: Maximum lag
            min_lag: Minimum lag
            lag_step: Step between lags
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with lag model results
        """
        result = {
            'input': input_name,
            'output': output_name,
            'optimal_lag': np.nan,
            'r2_no_lag': np.nan,
            'r2_with_lags': np.nan,
            'r2_improvement': np.nan,
            'cumulative_effect': np.nan,
            'memory_length': np.nan,
            'lag_coefficients': {},
            'lag_pvalues': {},
            'significant_lags': [],
            'needs_sequence_input': False
        }
        
        # Validate data
        mask = ~(np.isnan(input_data) | np.isnan(output_data))
        x = input_data[mask]
        y = output_data[mask]
        
        n = len(x)
        if n < max_lag + 50:
            return result
        
        try:
            # Create lagged features
            X_lagged, lags = LaggedEffectAnalyzer._create_lagged_features(
                x, max_lag, min_lag, lag_step
            )
            y_aligned = y[max_lag:]
            
            if len(y_aligned) < 50:
                return result
            
            # Model without lags (just contemporaneous)
            X_no_lag = x[max_lag:].reshape(-1, 1)
            model_no_lag = LinearRegression()
            model_no_lag.fit(X_no_lag, y_aligned)
            r2_no_lag = model_no_lag.score(X_no_lag, y_aligned)
            result['r2_no_lag'] = r2_no_lag
            
            # Model with all lags
            model_lags = LinearRegression()
            model_lags.fit(X_lagged, y_aligned)
            r2_with_lags = model_lags.score(X_lagged, y_aligned)
            result['r2_with_lags'] = r2_with_lags
            
            # R² improvement
            result['r2_improvement'] = r2_with_lags - r2_no_lag
            
            # Store lag coefficients
            for i, lag in enumerate(lags):
                coef = model_lags.coef_[i]
                result['lag_coefficients'][lag] = coef
                
                # Compute p-value using bootstrap
                n_bootstrap = 100
                coef_samples = []
                for _ in range(n_bootstrap):
                    idx = np.random.choice(len(y_aligned), len(y_aligned), replace=True)
                    try:
                        m = LinearRegression()
                        m.fit(X_lagged[idx], y_aligned[idx])
                        coef_samples.append(m.coef_[i])
                    except:
                        pass
                
                if len(coef_samples) > 10:
                    std = np.std(coef_samples)
                    if std > 0:
                        t_stat = coef / std
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y_aligned) - len(lags) - 1))
                        result['lag_pvalues'][lag] = p_value
                        
                        if p_value < 0.05:
                            result['significant_lags'].append(lag)
            
            # Cumulative effect (sum of all lag coefficients)
            result['cumulative_effect'] = sum(result['lag_coefficients'].values())
            
            # Find optimal lag using cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            best_lag = 0
            best_cv_score = -np.inf
            
            for test_max_lag in range(min_lag, max_lag + 1, lag_step):
                if test_max_lag > 0:
                    X_test, _ = LaggedEffectAnalyzer._create_lagged_features(
                        x, test_max_lag, min_lag, lag_step
                    )
                    y_test = y[test_max_lag:]
                else:
                    X_test = x[max_lag:].reshape(-1, 1)
                    y_test = y[max_lag:]
                
                if len(y_test) < cv_folds * 10:
                    continue
                
                try:
                    cv_scores = cross_val_score(
                        LinearRegression(), X_test, y_test, 
                        cv=tscv, scoring='r2'
                    )
                    mean_cv_score = np.mean(cv_scores)
                    
                    if mean_cv_score > best_cv_score:
                        best_cv_score = mean_cv_score
                        best_lag = test_max_lag
                except:
                    pass
            
            result['optimal_lag'] = best_lag
            
            # Memory length (last significant lag)
            if result['significant_lags']:
                result['memory_length'] = max(result['significant_lags'])
            else:
                result['memory_length'] = 0
            
            # Recommendation for sequence inputs
            result['needs_sequence_input'] = (
                result['r2_improvement'] > 0.05 and 
                len(result['significant_lags']) > 1
            )
            
        except Exception as e:
            logger.warning(f"Lag analysis failed for {input_name} -> {output_name}: {e}")
        
        return result
    
    def analyze_lagged_effects(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze lagged effects for all input-output pairs.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            DataFrame with lag analysis results
        """
        logger.info("Analyzing lagged effects...")
        
        self._init_dask_client()
        
        inputs = prepared_data['inputs']
        outputs = prepared_data['outputs']
        
        # Include datacenter outputs
        all_outputs = {**outputs, **prepared_data.get('datacenter_outputs', {})}
        
        # Create delayed tasks
        tasks = []
        for input_name, input_data in inputs.items():
            for output_name, output_data in all_outputs.items():
                task = delayed(self._fit_distributed_lag_model_static)(
                    input_name,
                    output_name,
                    input_data,
                    output_data,
                    self.max_lag,
                    self.min_lag,
                    self.lag_step,
                    self.cv_folds
                )
                tasks.append(task)
        
        # Execute in parallel
        logger.info(f"Executing {len(tasks)} lag analyses...")
        results = self.client.compute(tasks, sync=True)
        
        # Create DataFrame (excluding nested dicts for main df)
        results_simple = []
        for r in results:
            simple_r = {k: v for k, v in r.items() 
                       if k not in ['lag_coefficients', 'lag_pvalues', 'significant_lags']}
            simple_r['n_significant_lags'] = len(r['significant_lags'])
            simple_r['significant_lags_str'] = ','.join(map(str, r['significant_lags']))
            results_simple.append(simple_r)
        
        results_df = pd.DataFrame(results_simple)
        
        # Store full results for visualization
        self._full_lag_results = results
        
        # Sort by R² improvement
        results_df = results_df.sort_values('r2_improvement', ascending=False)
        
        logger.info(f"Completed {len(results_df)} lag analyses")
        
        return results_df
    
    def get_lag_coefficients(
        self,
        input_name: str,
        output_name: str
    ) -> Dict[int, float]:
        """
        Get lag coefficients for a specific input-output pair.
        
        Args:
            input_name: Input variable name
            output_name: Output variable name
            
        Returns:
            Dictionary mapping lag to coefficient
        """
        if not hasattr(self, '_full_lag_results'):
            logger.warning("No lag results available. Run analyze_lagged_effects first.")
            return {}
        
        for r in self._full_lag_results:
            if r['input'] == input_name and r['output'] == output_name:
                return r['lag_coefficients']
        
        return {}
    
    def compute_cumulative_effects(
        self,
        lag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute cumulative effect summary for visualization.
        
        Args:
            lag_df: DataFrame with lag analysis results
            
        Returns:
            DataFrame with cumulative effects by input-output pair
        """
        logger.info("Computing cumulative effects...")
        
        cumulative_data = []
        
        if not hasattr(self, '_full_lag_results'):
            return pd.DataFrame()
        
        for r in self._full_lag_results:
            lag_coeffs = r.get('lag_coefficients', {})
            if not lag_coeffs:
                continue
            
            lags = sorted(lag_coeffs.keys())
            cumulative = 0
            cumulative_by_lag = []
            
            for lag in lags:
                cumulative += lag_coeffs[lag]
                cumulative_by_lag.append({
                    'input': r['input'],
                    'output': r['output'],
                    'lag': lag,
                    'coefficient': lag_coeffs[lag],
                    'cumulative_effect': cumulative
                })
            
            cumulative_data.extend(cumulative_by_lag)
        
        return pd.DataFrame(cumulative_data)
    
    def get_memory_length_summary(
        self,
        lag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Summarize memory length (system response time) by input.
        
        Args:
            lag_df: DataFrame with lag analysis results
            
        Returns:
            DataFrame with memory length statistics by input
        """
        logger.info("Computing memory length summary...")
        
        summary = []
        for input_name in self.input_vars:
            input_data = lag_df[lag_df['input'] == input_name]
            
            if input_data.empty:
                continue
            
            summary.append({
                'input': input_name,
                'mean_memory_length': input_data['memory_length'].mean(),
                'max_memory_length': input_data['memory_length'].max(),
                'mean_optimal_lag': input_data['optimal_lag'].mean(),
                'pct_needs_sequence': (input_data['needs_sequence_input'].sum() / 
                                       len(input_data) * 100),
                'mean_r2_improvement': input_data['r2_improvement'].mean()
            })
        
        return pd.DataFrame(summary)
