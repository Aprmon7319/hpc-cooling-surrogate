import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mutual_info_score, normalized_mutual_info_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from typing import Dict, List, Tuple, Optional
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class DirectEffectAnalyzer:
    """
    Analyzes direct effects of inputs on outputs using correlation,
    regression, and mutual information metrics.
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
        Initialize the analyzer.
        
        Args:
            system_name: System configuration name (e.g., 'marconi100', 'leonardo')
            n_workers: Number of Dask workers (for analysis, not simulation)
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
        self.client = None
        
        logger.info(f"DirectEffectAnalyzer initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
        
        
    def _init_dask_client(self):
        """Initialize Dask distributed client for analysis tasks only."""
        if self.client is None:
            logger.info("Initializing Dask cluster for analysis tasks...")
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
        logger.info("Preparing data for direct effect analysis...")
        
        prepared_data = {
            'inputs': {},
            'outputs': {}
        }
        
        # Aggregate inputs across CDUs
        for cdu_idx in range(1, self.num_cdus + 1):
            q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'

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
        prepared_data['inputs']['T_ext'] = data[t_ext_col].values
        
        # Aggregate outputs across CDUs
        for cdu_idx in range(1, self.num_cdus + 1):
            for output_var in self.output_vars:
                output_col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{output_var}'
                
                if output_var not in prepared_data['outputs']:
                    prepared_data['outputs'][output_var] = data[output_col].values.reshape(-1, 1)
                else:
                    prepared_data['outputs'][output_var] = np.hstack([
                        prepared_data['outputs'][output_var],
                        data[output_col].values.reshape(-1, 1)
                    ])
        
        logger.info(f"Data prepared: {len(prepared_data['inputs'])} inputs, {len(prepared_data['outputs'])} outputs")
        return prepared_data
    
    @staticmethod
    def _compute_univariate_metrics_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str
    ) -> Dict[str, float]:
        """
        Static method for computing univariate sensitivity metrics for input-output pair.
        Used for parallel execution with Dask.
        """
        # Flatten data if multi-dimensional
        if input_data.ndim > 1:
            input_flat = input_data.flatten()
        else:
            # Repeat for each CDU if single dimension
            input_flat = np.repeat(input_data, output_data.shape[1])
        
        output_flat = output_data.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        input_clean = input_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input_clean) < 10:
            return {}
        
        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(input_clean, output_clean)
        
        # Spearman correlation (monotonic)
        spearman_r, spearman_p = stats.spearmanr(input_clean, output_clean)
        
        # Mutual information (non-linear) - use normalized version for [0, 1] range
        discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        input_discrete = discretizer.fit_transform(input_clean.reshape(-1, 1)).flatten().astype(int)
        output_discrete = discretizer.fit_transform(output_clean.reshape(-1, 1)).flatten().astype(int)
        
        # Use normalized mutual info score which returns values in [0, 1]
        mi_normalized = normalized_mutual_info_score(input_discrete, output_discrete)
        
        # Also compute raw mutual info for reference
        mi_raw = mutual_info_score(input_discrete, output_discrete)
        
        # Linear regression R²
        lr = LinearRegression()
        lr.fit(input_clean.reshape(-1, 1), output_clean)
        r2 = r2_score(output_clean, lr.predict(input_clean.reshape(-1, 1)))
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mutual_info': mi_normalized,  # Now normalized to [0, 1]
            'mutual_info_raw': mi_raw,     # Raw value in nats for reference
            'r2_score': r2,
            'linear_coef': lr.coef_[0],
            'linear_intercept': lr.intercept_
        }
    
    def compute_univariate_metrics(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str
    ) -> Dict[str, float]:
        """Compute univariate sensitivity metrics for input-output pair."""
        return self._compute_univariate_metrics_static(
            input_data, output_data, input_name, output_name
        )
    
    def analyze_all_pairs(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze all input-output pairs using Dask parallel processing.
        """
        logger.info("Computing univariate metrics for all input-output pairs using Dask...")
        
        self._init_dask_client()
        
        # Create list of delayed tasks
        delayed_tasks = []
        task_info = []
        
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'][input_name]
            
            for output_name in self.output_vars:
                output_data = prepared_data['outputs'][output_name]
                
                # Create delayed task
                task = delayed(self._compute_univariate_metrics_static)(
                    input_data, output_data, input_name, output_name
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        # Compute all tasks in parallel
        logger.info(f"Executing {len(delayed_tasks)} analysis tasks in parallel...")
        results_list = dask.compute(*delayed_tasks)
        
        # Combine results
        results = []
        for (input_name, output_name), metrics in zip(task_info, results_list):
            if metrics:
                metrics['input'] = input_name
                metrics['output'] = output_name
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Computed metrics for {len(results_df)} input-output pairs")
        
        return results_df
    
    def rank_input_importance(
        self,
        metrics_df: pd.DataFrame,
        metric: str = 'mutual_info'
    ) -> pd.DataFrame:
        """Rank inputs by importance for each output."""
        rankings = []
        
        for output in self.output_vars:
            output_metrics = metrics_df[metrics_df['output'] == output].copy()
            output_metrics = output_metrics.sort_values(metric, ascending=False)
            
            for rank, (_, row) in enumerate(output_metrics.iterrows(), 1):
                rankings.append({
                    'output': output,
                    'input': row['input'],
                    'rank': rank,
                    'metric': metric,
                    'value': row[metric]
                })
        
        return pd.DataFrame(rankings)
    
    @staticmethod
    def _compute_response_surface_static(
        input1_data: np.ndarray,
        input2_data: np.ndarray,
        output_data: np.ndarray,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Static method for computing response surface."""
        # Get the expected flat length from output
        output_flat = output_data.flatten()
        expected_length = len(output_flat)
        
        # Flatten input1
        if input1_data.ndim > 1:
            input1_flat = input1_data.flatten()
        else:
            # T_ext case: repeat for each CDU
            n_repeats = expected_length // len(input1_data)
            input1_flat = np.repeat(input1_data, n_repeats)
            # Handle any length mismatch
            if len(input1_flat) < expected_length:
                input1_flat = np.concatenate([
                    input1_flat, 
                    np.full(expected_length - len(input1_flat), input1_flat[-1])
                ])
            elif len(input1_flat) > expected_length:
                input1_flat = input1_flat[:expected_length]
        
        # Flatten input2
        if input2_data.ndim > 1:
            input2_flat = input2_data.flatten()
        else:
            # T_ext case: repeat for each CDU
            n_repeats = expected_length // len(input2_data)
            input2_flat = np.repeat(input2_data, n_repeats)
            # Handle any length mismatch
            if len(input2_flat) < expected_length:
                input2_flat = np.concatenate([
                    input2_flat,
                    np.full(expected_length - len(input2_flat), input2_flat[-1])
                ])
            elif len(input2_flat) > expected_length:
                input2_flat = input2_flat[:expected_length]
        
        # Remove NaN
        mask = ~(np.isnan(input1_flat) | np.isnan(input2_flat) | np.isnan(output_flat))
        input1_clean = input1_flat[mask]
        input2_clean = input2_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input1_clean) < 10:
            # Return flat surface if insufficient data
            input1_range = np.linspace(0, 1, n_points)
            input2_range = np.linspace(0, 1, n_points)
            input1_grid, input2_grid = np.meshgrid(input1_range, input2_range)
            return input1_grid, input2_grid, np.zeros_like(input1_grid)
        
        # Create grid
        input1_range = np.linspace(input1_clean.min(), input1_clean.max(), n_points)
        input2_range = np.linspace(input2_clean.min(), input2_clean.max(), n_points)
        input1_grid, input2_grid = np.meshgrid(input1_range, input2_range)
        
        # Fit polynomial model - use degree=3 for more curvature if present
        poly = PolynomialFeatures(degree=3, include_bias=True)
        X = poly.fit_transform(np.column_stack([input1_clean, input2_clean]))
        
        # Use lower regularization to capture more curvature
        model = Ridge(alpha=0.1)
        model.fit(X, output_clean)
        
        # Predict on grid
        X_grid = poly.transform(np.column_stack([
            input1_grid.flatten(),
            input2_grid.flatten()
        ]))
        output_grid = model.predict(X_grid).reshape(input1_grid.shape)
        
        return input1_grid, input2_grid, output_grid
    
    # def compute_response_surface(
    #     self,
    #     prepared_data: Dict[str, np.ndarray],
    #     input1: str,
    #     input2: str,
    #     output: str,
    #     n_points: int = 50
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """Compute response surface for two inputs and one output."""
    #     logger.info(f"Computing response surface: {output} vs {input1} x {input2}")
        
    #     input1_data = prepared_data['inputs'][input1]
    #     input2_data = prepared_data['inputs'][input2]
    #     output_data = prepared_data['outputs'][output]
        
    #     return self._compute_response_surface_static(
    #         input1_data, input2_data, output_data, n_points
    #     )
    def compute_all_response_surfaces(
        self,
        prepared_data: Dict[str, np.ndarray],
        input_pairs: Optional[List[Tuple[str, str]]] = None,
        n_points: int = 50
    ) -> Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute response surfaces for all input-output combinations using Dask parallel processing.
        
        Args:
            prepared_data: Prepared data dictionary
            input_pairs: List of (input1, input2) pairs to analyze. 
                        If None, uses all combinations.
            n_points: Number of points per dimension
            
        Returns:
            Dictionary mapping (output, input1, input2) to (X, Y, Z) grid arrays
        """
        logger.info("Computing all response surfaces in parallel with Dask...")
        
        self._init_dask_client()
        
        if input_pairs is None:
            # All pairwise combinations of inputs
            input_pairs = [
                ('Q_flow', 'T_Air'),
                ('Q_flow', 'T_ext'),
                ('T_Air', 'T_ext')
            ]
        
        # Create delayed tasks for all combinations
        delayed_tasks = []
        task_info = []
        
        for output_name in self.output_vars:
            output_data = prepared_data['outputs'][output_name]
            
            for input1, input2 in input_pairs:
                input1_data = prepared_data['inputs'][input1]
                input2_data = prepared_data['inputs'][input2]
                
                task = delayed(self._compute_response_surface_static)(
                    input1_data, input2_data, output_data, n_points
                )
                delayed_tasks.append(task)
                task_info.append((output_name, input1, input2))
        
        logger.info(f"Executing {len(delayed_tasks)} response surface computations...")
        results_list = dask.compute(*delayed_tasks)
        
        # Combine results into dictionary
        surfaces = {}
        for (output_name, input1, input2), (X, Y, Z) in zip(task_info, results_list):
            surfaces[(output_name, input1, input2)] = (X, Y, Z)
        
        logger.info(f"Computed {len(surfaces)} response surfaces")
        return surfaces
    
    @staticmethod
    def _analyze_interaction_effects_static(
        inputs_list: List[np.ndarray],
        output_data: np.ndarray,
        output_name: str
    ) -> Dict[str, float]:
        """Static method for analyzing interaction effects."""
        X = np.column_stack(inputs_list)
        y = output_data.flatten()
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Fit linear model (no interactions)
        lr_simple = Ridge(alpha=1.0)
        lr_simple.fit(X_clean, y_clean)
        r2_simple = r2_score(y_clean, lr_simple.predict(X_clean))
        
        # Fit polynomial model (with interactions)
        poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
        X_poly = poly.fit_transform(X_clean)
        
        lr_poly = Ridge(alpha=1.0)
        lr_poly.fit(X_poly, y_clean)
        r2_poly = r2_score(y_clean, lr_poly.predict(X_poly))
        
        interaction_effect = r2_poly - r2_simple
        
        return {
            'output': output_name,
            'r2_linear': r2_simple,
            'r2_polynomial': r2_poly,
            'interaction_effect': interaction_effect
        }
    
    def analyze_all_interaction_effects(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Analyze interaction effects for all outputs using Dask."""
        logger.info("Analyzing interaction effects for all outputs in parallel...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        
        for output in self.output_vars:
            inputs_list = []
            for input_name in self.input_vars:
                input_data = prepared_data['inputs'][input_name]
                if input_data.ndim > 1:
                    inputs_list.append(input_data.flatten())
                else:
                    output_data = prepared_data['outputs'][output]
                    inputs_list.append(np.repeat(input_data, output_data.shape[1]))
            
            output_data = prepared_data['outputs'][output]
            
            task = delayed(self._analyze_interaction_effects_static)(
                inputs_list, output_data, output
            )
            delayed_tasks.append(task)
        
        logger.info(f"Executing {len(delayed_tasks)} interaction analysis tasks...")
        results_list = dask.compute(*delayed_tasks)
        
        return pd.DataFrame(results_list)
    
    def analyze_isolated_effects(
        self,
        prepared_data: Dict[str, np.ndarray],
        tolerance: float = 0.1
    ) -> pd.DataFrame:
        """Analyze effects of single input with others held near mean."""
        logger.info("Analyzing isolated input effects (others held at mean)...")
        
        results = []
        
        for target_input in self.input_vars:
            input_means = {}
            input_stds = {}
            for inp in self.input_vars:
                data = prepared_data['inputs'][inp]
                if data.ndim > 1:
                    data = data.flatten()
                input_means[inp] = np.nanmean(data)
                input_stds[inp] = np.nanstd(data)
            
            target_data = prepared_data['inputs'][target_input]
            if target_data.ndim > 1:
                target_data_flat = target_data.flatten()
            else:
                target_data_flat = target_data.copy()
            
            sample_output = prepared_data['outputs'][self.output_vars[0]]
            expected_length = sample_output.flatten().shape[0]
            
            mask = np.ones(expected_length, dtype=bool)
            
            for inp in self.input_vars:
                if inp != target_input:
                    data = prepared_data['inputs'][inp]
                    
                    if data.ndim > 1:
                        data_flat = data.flatten()
                    else:
                        data_flat = data.copy()
                    
                    if len(data_flat) != expected_length:
                        num_repeats = expected_length // len(data_flat)
                        data_flat = np.repeat(data_flat, num_repeats)
                        
                        if len(data_flat) < expected_length:
                            data_flat = np.concatenate([
                                data_flat,
                                np.full(expected_length - len(data_flat), data_flat[-1])
                            ])
                        elif len(data_flat) > expected_length:
                            data_flat = data_flat[:expected_length]
                    
                    deviation = np.abs(data_flat - input_means[inp])
                    threshold = tolerance * input_stds[inp]
                    mask &= (deviation < threshold)
            
            if len(target_data_flat) != expected_length:
                num_repeats = expected_length // len(target_data_flat)
                target_data_flat = np.repeat(target_data_flat, num_repeats)
                
                if len(target_data_flat) < expected_length:
                    target_data_flat = np.concatenate([
                        target_data_flat,
                        np.full(expected_length - len(target_data_flat), target_data_flat[-1])
                    ])
                elif len(target_data_flat) > expected_length:
                    target_data_flat = target_data_flat[:expected_length]
            
            logger.info(f"  {target_input}: {np.sum(mask)} samples with others at mean")
            
            if np.sum(mask) < 100:
                logger.warning(f"  Too few samples for {target_input}, skipping")
                continue
            
            target_data_filtered = target_data_flat[mask]
            
            for output_name in self.output_vars:
                output_data = prepared_data['outputs'][output_name]
                
                if output_data.ndim > 1:
                    output_data_flat = output_data.flatten()
                else:
                    output_data_flat = output_data.copy()
                
                output_data_filtered = output_data_flat[mask]
                
                valid_mask = ~(np.isnan(target_data_filtered) | np.isnan(output_data_filtered))
                
                if np.sum(valid_mask) < 50:
                    continue
                
                target_valid = target_data_filtered[valid_mask].reshape(-1, 1)
                output_valid = output_data_filtered[valid_mask].reshape(-1, 1)
                
                metrics = self._compute_univariate_metrics_static(
                    target_valid,
                    output_valid,
                    target_input,
                    output_name
                )
                
                if metrics:
                    metrics['input'] = target_input
                    metrics['output'] = output_name
                    metrics['analysis_type'] = 'isolated'
                    metrics['n_samples'] = len(target_valid)
                    results.append(metrics)
        
        if not results:
            logger.warning("No isolated effects could be computed")
            return pd.DataFrame()
        
        return pd.DataFrame(results)