"""
Partial Correlation Analysis for FMU input-output relationships.

Provides methods to compute partial correlations, removing confounding
effects from other variables to identify direct relationships.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class PartialCorrelationAnalyzer:
    """
    Analyzes partial correlations between inputs and outputs, controlling
    for confounding variables to identify direct effects.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        significance_threshold: float = 0.05,
        edge_threshold: float = 0.1,
        method: str = 'pearson',
        **config_overrides
    ):
        """
        Initialize the partial correlation analyzer.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers for parallel analysis
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            significance_threshold: p-value threshold for significance
            edge_threshold: Minimum partial correlation for network edges
            method: Correlation method ('pearson', 'spearman')
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
        self.significance_threshold = significance_threshold
        self.edge_threshold = edge_threshold
        self.method = method
        
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
        
        logger.info(f"PartialCorrelationAnalyzer initialized for system: {system_name}")
        logger.info(f"Method: {method}, Significance threshold: {significance_threshold}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client for analysis tasks."""
        if self.client is None:
            logger.info("Initializing Dask cluster for partial correlation analysis...")
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
        logger.info("Preparing data for partial correlation analysis...")
        
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
        
        # Store number of samples
        prepared['n_samples'] = n_samples
        
        logger.info(f"Prepared data: {len(prepared['inputs'])} inputs, "
                   f"{len(prepared['outputs'])} outputs, "
                   f"{len(prepared['datacenter_outputs'])} datacenter outputs")
        
        return prepared
    
    @staticmethod
    def _compute_partial_correlation(
        x: np.ndarray,
        y: np.ndarray,
        controls: List[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Compute partial correlation between x and y controlling for other variables.
        
        Uses residualization method: regress out controls from both x and y,
        then compute correlation of residuals.
        
        Args:
            x: First variable
            y: Second variable
            controls: List of control variables to partial out
            
        Returns:
            Tuple of (partial correlation, p-value)
        """
        if len(controls) == 0:
            # No controls, just compute regular correlation
            r, p = stats.pearsonr(x, y)
            return r, p
        
        # Build control matrix
        n = len(x)
        Z = np.column_stack(controls)
        
        # Add intercept
        Z = np.column_stack([np.ones(n), Z])
        
        # Compute residuals using least squares
        try:
            # Use pseudo-inverse for numerical stability
            Z_pinv = pinv(Z)
            
            # Residualize x
            beta_x = Z_pinv @ x
            x_resid = x - Z @ beta_x
            
            # Residualize y
            beta_y = Z_pinv @ y
            y_resid = y - Z @ beta_y
            
            # Compute correlation of residuals
            r, p = stats.pearsonr(x_resid, y_resid)
            
            # Adjust degrees of freedom for controls
            n_controls = len(controls)
            df = n - n_controls - 2
            if df > 0:
                t_stat = r * np.sqrt(df / (1 - r**2 + 1e-10))
                p = 2 * stats.t.sf(abs(t_stat), df)
            
            return r, p
            
        except Exception as e:
            logger.warning(f"Partial correlation computation failed: {e}")
            return np.nan, 1.0
    
    @staticmethod
    def _analyze_partial_correlation_static(
        input_name: str,
        output_name: str,
        input_data: np.ndarray,
        output_data: np.ndarray,
        other_inputs: Dict[str, np.ndarray],
        method: str = 'pearson'
    ) -> Dict:
        """
        Static method for parallel execution of partial correlation analysis.
        
        Args:
            input_name: Name of input variable
            output_name: Name of output variable
            input_data: Input variable data
            output_data: Output variable data
            other_inputs: Dictionary of other input variables to control for
            method: Correlation method
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'input': input_name,
            'output': output_name,
            'pearson_corr': np.nan,
            'pearson_pvalue': 1.0,
            'partial_corr': np.nan,
            'partial_pvalue': 1.0,
            'corr_difference': np.nan,
            'is_direct_effect': False
        }
        
        # Validate data
        mask = ~(np.isnan(input_data) | np.isnan(output_data))
        for name, arr in other_inputs.items():
            mask &= ~np.isnan(arr)
        
        x = input_data[mask]
        y = output_data[mask]
        controls = [arr[mask] for arr in other_inputs.values()]
        
        if len(x) < 10:
            return result
        
        try:
            # Compute Pearson correlation
            if method == 'spearman':
                pearson_r, pearson_p = stats.spearmanr(x, y)
            else:
                pearson_r, pearson_p = stats.pearsonr(x, y)
            
            result['pearson_corr'] = pearson_r
            result['pearson_pvalue'] = pearson_p
            
            # Compute partial correlation
            partial_r, partial_p = PartialCorrelationAnalyzer._compute_partial_correlation(
                x, y, controls
            )
            
            result['partial_corr'] = partial_r
            result['partial_pvalue'] = partial_p
            
            # Compute difference
            result['corr_difference'] = abs(pearson_r) - abs(partial_r)
            
            # Determine if this is a direct effect
            result['is_direct_effect'] = (
                partial_p < 0.05 and abs(partial_r) > 0.1
            )
            
        except Exception as e:
            logger.warning(f"Analysis failed for {input_name} -> {output_name}: {e}")
        
        return result
    
    def analyze_partial_correlations(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute partial correlations for all input-output pairs.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            DataFrame with partial correlation results
        """
        logger.info("Computing partial correlations...")
        
        self._init_dask_client()
        
        inputs = prepared_data['inputs']
        outputs = prepared_data['outputs']
        
        # Include datacenter outputs
        all_outputs = {**outputs, **prepared_data.get('datacenter_outputs', {})}
        
        # Create delayed tasks
        tasks = []
        for input_name in self.input_vars:
            if input_name not in inputs:
                continue
                
            input_data = inputs[input_name]
            
            # Get other inputs as controls
            other_inputs = {
                name: arr for name, arr in inputs.items() 
                if name != input_name
            }
            
            for output_name, output_data in all_outputs.items():
                task = delayed(self._analyze_partial_correlation_static)(
                    input_name,
                    output_name,
                    input_data,
                    output_data,
                    other_inputs,
                    self.method
                )
                tasks.append(task)
        
        # Execute in parallel
        logger.info(f"Executing {len(tasks)} partial correlation analyses...")
        results = self.client.compute(tasks, sync=True)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by absolute partial correlation
        results_df['abs_partial_corr'] = results_df['partial_corr'].abs()
        results_df = results_df.sort_values('abs_partial_corr', ascending=False)
        results_df = results_df.drop(columns=['abs_partial_corr'])
        
        logger.info(f"Computed {len(results_df)} partial correlations")
        
        return results_df
    
    def compute_correlation_comparison(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute comparison between Pearson and partial correlations.
        
        Returns matrices for visualization.
        
        Args:
            prepared_data: Dictionary with prepared input/output data
            
        Returns:
            Dictionary with Pearson and partial correlation matrices
        """
        logger.info("Computing correlation comparison matrices...")
        
        partial_df = self.analyze_partial_correlations(prepared_data)
        
        # Create pivot tables
        pearson_pivot = partial_df.pivot(
            index='input',
            columns='output',
            values='pearson_corr'
        )
        
        partial_pivot = partial_df.pivot(
            index='input',
            columns='output',
            values='partial_corr'
        )
        
        difference_pivot = partial_df.pivot(
            index='input',
            columns='output',
            values='corr_difference'
        )
        
        direct_effect_pivot = partial_df.pivot(
            index='input',
            columns='output',
            values='is_direct_effect'
        )
        
        return {
            'pearson': pearson_pivot,
            'partial': partial_pivot,
            'difference': difference_pivot,
            'direct_effect': direct_effect_pivot,
            'full_results': partial_df
        }
    
    def build_network_data(
        self,
        partial_df: pd.DataFrame
    ) -> Dict:
        """
        Build network diagram data from partial correlation results.
        
        Args:
            partial_df: DataFrame with partial correlation results
            
        Returns:
            Dictionary with nodes and edges for network visualization
        """
        logger.info("Building network diagram data...")
        
        nodes = []
        edges = []
        
        # Add input nodes
        for input_name in self.input_vars:
            nodes.append({
                'id': input_name,
                'type': 'input',
                'label': input_name
            })
        
        # Add output nodes
        output_names = partial_df['output'].unique()
        for output_name in output_names:
            nodes.append({
                'id': output_name,
                'type': 'output',
                'label': output_name
            })
        
        # Add edges for significant partial correlations
        for _, row in partial_df.iterrows():
            if abs(row['partial_corr']) > self.edge_threshold and row['partial_pvalue'] < self.significance_threshold:
                edges.append({
                    'source': row['input'],
                    'target': row['output'],
                    'weight': row['partial_corr'],
                    'abs_weight': abs(row['partial_corr']),
                    'p_value': row['partial_pvalue'],
                    'is_direct': row['is_direct_effect']
                })
        
        logger.info(f"Network: {len(nodes)} nodes, {len(edges)} significant edges")
        
        return {
            'nodes': nodes,
            'edges': edges
        }
