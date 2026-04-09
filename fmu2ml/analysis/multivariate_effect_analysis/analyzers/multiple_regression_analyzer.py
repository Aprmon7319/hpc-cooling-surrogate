"""
Multiple Regression Analysis for FMU input-output relationships.

Provides standardized beta coefficients, significance testing,
variance inflation factors, and variance explained metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import logging
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class MultipleRegressionAnalyzer:
    """
    Performs multiple regression analysis to understand the relative importance
    of each input in predicting outputs.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        vif_threshold: float = 10.0,
        significance_threshold: float = 0.05,
        standardize: bool = True,
        bootstrap_samples: int = 1000,
        **config_overrides
    ):
        """
        Initialize the multiple regression analyzer.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            vif_threshold: VIF threshold for multicollinearity warning
            significance_threshold: p-value threshold for significance
            standardize: Whether to standardize inputs for beta coefficients
            bootstrap_samples: Number of bootstrap samples for confidence intervals
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
        self.vif_threshold = vif_threshold
        self.significance_threshold = significance_threshold
        self.standardize = standardize
        self.bootstrap_samples = bootstrap_samples
        
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
        
        logger.info(f"MultipleRegressionAnalyzer initialized for system: {system_name}")
        logger.info(f"VIF threshold: {vif_threshold}, Standardize: {standardize}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client."""
        if self.client is None:
            logger.info("Initializing Dask cluster for multiple regression analysis...")
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
        logger.info("Preparing data for multiple regression analysis...")
        
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
                   f"{len(prepared['outputs'])} outputs, "
                   f"{len(prepared['datacenter_outputs'])} datacenter outputs")
        
        return prepared
    
    @staticmethod
    def _compute_vif(X: np.ndarray, feature_idx: int) -> float:
        """
        Compute Variance Inflation Factor for a single feature.
        
        Args:
            X: Feature matrix
            feature_idx: Index of feature to compute VIF for
            
        Returns:
            VIF value
        """
        y = X[:, feature_idx]
        X_others = np.delete(X, feature_idx, axis=1)
        
        if X_others.shape[1] == 0:
            return 1.0
        
        try:
            model = LinearRegression()
            model.fit(X_others, y)
            r2 = model.score(X_others, y)
            
            if r2 >= 1.0:
                return np.inf
            
            vif = 1 / (1 - r2)
            return vif
            
        except Exception:
            return np.inf
    
    @staticmethod
    def _fit_regression_static(
        output_name: str,
        output_data: np.ndarray,
        input_matrix: np.ndarray,
        input_names: List[str],
        standardize: bool = True,
        bootstrap_samples: int = 1000
    ) -> Dict:
        """
        Static method for parallel execution of regression fitting.
        
        Args:
            output_name: Name of output variable
            output_data: Output variable data
            input_matrix: Matrix of input variables
            input_names: Names of input variables
            standardize: Whether to standardize inputs
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with regression results
        """
        result = {
            'output': output_name,
            'r2': np.nan,
            'adj_r2': np.nan,
            'f_statistic': np.nan,
            'f_pvalue': np.nan,
            'rmse': np.nan
        }
        
        # Add placeholders for each input
        for input_name in input_names:
            result[f'beta_{input_name}'] = np.nan
            result[f'beta_std_{input_name}'] = np.nan
            result[f'pvalue_{input_name}'] = np.nan
            result[f'vif_{input_name}'] = np.nan
            result[f'is_significant_{input_name}'] = False
        
        # Validate data
        mask = ~np.isnan(output_data)
        for i in range(input_matrix.shape[1]):
            mask &= ~np.isnan(input_matrix[:, i])
        
        X = input_matrix[mask]
        y = output_data[mask]
        
        n = len(y)
        p = X.shape[1]
        
        if n < p + 10:
            return result
        
        try:
            # Standardize if requested
            if standardize:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                X_scaled = X
                y_scaled = y
            
            # Fit regression
            model = LinearRegression()
            model.fit(X_scaled, y_scaled)
            
            # Predictions and residuals
            y_pred = model.predict(X_scaled)
            residuals = y_scaled - y_pred
            
            # R² and adjusted R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_scaled - np.mean(y_scaled)) ** 2)
            
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
            
            result['r2'] = r2
            result['adj_r2'] = adj_r2
            
            # RMSE
            result['rmse'] = np.sqrt(ss_res / n)
            
            # F-statistic
            if n > p + 1:
                ms_reg = (ss_tot - ss_res) / p if p > 0 else 0
                ms_res = ss_res / (n - p - 1)
                
                if ms_res > 0:
                    f_stat = ms_reg / ms_res
                    f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1)
                    result['f_statistic'] = f_stat
                    result['f_pvalue'] = f_pvalue
            
            # Beta coefficients and statistics
            for i, input_name in enumerate(input_names):
                beta = model.coef_[i]
                result[f'beta_{input_name}'] = beta
                
                # Standard error via bootstrap
                beta_samples = []
                for _ in range(min(bootstrap_samples, 100)):  # Limit for speed
                    idx = np.random.choice(n, n, replace=True)
                    try:
                        m = LinearRegression()
                        m.fit(X_scaled[idx], y_scaled[idx])
                        beta_samples.append(m.coef_[i])
                    except:
                        pass
                
                if len(beta_samples) > 10:
                    beta_std = np.std(beta_samples)
                    result[f'beta_std_{input_name}'] = beta_std
                    
                    # t-test for significance
                    if beta_std > 0:
                        t_stat = beta / beta_std
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
                        result[f'pvalue_{input_name}'] = p_value
                        result[f'is_significant_{input_name}'] = p_value < 0.05
                
                # VIF
                vif = MultipleRegressionAnalyzer._compute_vif(X, i)
                result[f'vif_{input_name}'] = vif
            
        except Exception as e:
            logger.warning(f"Regression failed for {output_name}: {e}")
        
        return result
    
    def analyze_multiple_regression(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Perform multiple regression analysis for all outputs.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            DataFrame with regression results for each output
        """
        logger.info("Performing multiple regression analysis...")
        
        self._init_dask_client()
        
        inputs = prepared_data['inputs']
        outputs = prepared_data['outputs']
        
        # Include datacenter outputs
        all_outputs = {**outputs, **prepared_data.get('datacenter_outputs', {})}
        
        # Build input matrix
        input_names = list(inputs.keys())
        input_matrix = np.column_stack([inputs[name] for name in input_names])
        
        # Create delayed tasks
        tasks = []
        for output_name, output_data in all_outputs.items():
            task = delayed(self._fit_regression_static)(
                output_name,
                output_data,
                input_matrix,
                input_names,
                self.standardize,
                self.bootstrap_samples
            )
            tasks.append(task)
        
        # Execute in parallel
        logger.info(f"Executing {len(tasks)} regression analyses...")
        results = self.client.compute(tasks, sync=True)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by R²
        results_df = results_df.sort_values('r2', ascending=False)
        
        logger.info(f"Completed {len(results_df)} regression analyses")
        
        return results_df
    
    def compute_beta_coefficient_matrix(
        self,
        regression_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract beta coefficient matrix for visualization.
        
        Args:
            regression_df: DataFrame with regression results
            
        Returns:
            DataFrame with beta coefficients (rows=outputs, cols=inputs)
        """
        logger.info("Extracting beta coefficient matrix...")
        
        beta_cols = [col for col in regression_df.columns if col.startswith('beta_') and not col.startswith('beta_std_')]
        
        beta_data = {}
        for output in regression_df['output']:
            row_data = regression_df[regression_df['output'] == output]
            for col in beta_cols:
                input_name = col.replace('beta_', '')
                if input_name not in beta_data:
                    beta_data[input_name] = {}
                beta_data[input_name][output] = row_data[col].values[0]
        
        beta_matrix = pd.DataFrame(beta_data).T
        
        return beta_matrix
    
    def compute_significance_matrix(
        self,
        regression_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract p-value matrix for significance heatmap.
        
        Args:
            regression_df: DataFrame with regression results
            
        Returns:
            DataFrame with p-values (rows=inputs, cols=outputs)
        """
        logger.info("Extracting significance matrix...")
        
        pvalue_cols = [col for col in regression_df.columns if col.startswith('pvalue_')]
        
        pvalue_data = {}
        for output in regression_df['output']:
            row_data = regression_df[regression_df['output'] == output]
            for col in pvalue_cols:
                input_name = col.replace('pvalue_', '')
                if input_name not in pvalue_data:
                    pvalue_data[input_name] = {}
                pvalue_data[input_name][output] = row_data[col].values[0]
        
        pvalue_matrix = pd.DataFrame(pvalue_data).T
        
        return pvalue_matrix
    
    def compute_vif_summary(
        self,
        regression_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract VIF summary for multicollinearity check.
        
        Args:
            regression_df: DataFrame with regression results
            
        Returns:
            DataFrame with VIF values and multicollinearity warnings
        """
        logger.info("Computing VIF summary...")
        
        vif_cols = [col for col in regression_df.columns if col.startswith('vif_')]
        
        vif_summary = []
        for col in vif_cols:
            input_name = col.replace('vif_', '')
            vif_values = regression_df[col].dropna()
            
            if len(vif_values) > 0:
                mean_vif = vif_values.mean()
                max_vif = vif_values.max()
                
                vif_summary.append({
                    'input': input_name,
                    'mean_vif': mean_vif,
                    'max_vif': max_vif,
                    'multicollinearity_warning': max_vif > self.vif_threshold
                })
        
        return pd.DataFrame(vif_summary)
    
    def compute_variance_explained(
        self,
        regression_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract R² values for variance explained bar chart.
        
        Args:
            regression_df: DataFrame with regression results
            
        Returns:
            DataFrame with R² and adjusted R² for each output
        """
        logger.info("Computing variance explained summary...")
        
        variance_df = regression_df[['output', 'r2', 'adj_r2', 'f_statistic', 'f_pvalue']].copy()
        variance_df = variance_df.sort_values('r2', ascending=False)
        variance_df['model_significant'] = variance_df['f_pvalue'] < self.significance_threshold
        
        return variance_df
