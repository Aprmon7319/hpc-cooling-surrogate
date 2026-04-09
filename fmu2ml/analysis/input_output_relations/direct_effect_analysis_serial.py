import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DirectEffectAnalyzer:
    """
    Analyzes direct effects of inputs on outputs using correlation,
    regression, and mutual information metrics.
    """
    
    def __init__(self, num_cdus: int):
        """
        Initialize the analyzer.
        
        Args:
            num_cdus: Number of CDUs in the datacenter
        """
        self.num_cdus = num_cdus
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
    
    def compute_univariate_metrics(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str
    ) -> Dict[str, float]:
        """
        Compute univariate sensitivity metrics for input-output pair.
        
        Args:
            input_data: Input variable data (n_samples, n_cdus) or (n_samples,)
            output_data: Output variable data (n_samples, n_cdus) or (n_samples,)
            input_name: Name of input variable
            output_name: Name of output variable
            
        Returns:
            Dictionary of metrics
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
            logger.warning(f"Insufficient data for {input_name} vs {output_name}")
            return {}
        
        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(input_clean, output_clean)
        
        # Spearman correlation (monotonic)
        spearman_r, spearman_p = stats.spearmanr(input_clean, output_clean)
        
        # Mutual information (non-linear)
        # Discretize continuous variables for MI calculation
        discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        input_discrete = discretizer.fit_transform(input_clean.reshape(-1, 1)).flatten()
        output_discrete = discretizer.fit_transform(output_clean.reshape(-1, 1)).flatten()
        mi = mutual_info_score(input_discrete, output_discrete)
        
        # Linear regression R²
        lr = LinearRegression()
        lr.fit(input_clean.reshape(-1, 1), output_clean)
        r2 = r2_score(output_clean, lr.predict(input_clean.reshape(-1, 1)))
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mutual_info': mi,
            'r2_score': r2,
            'linear_coef': lr.coef_[0],
            'linear_intercept': lr.intercept_
        }
    
    def analyze_all_pairs(
        self,
        prepared_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Analyze all input-output pairs.
        
        Args:
            prepared_data: Prepared data dictionary
            
        Returns:
            DataFrame with metrics for all pairs
        """
        logger.info("Computing univariate metrics for all input-output pairs...")
        
        results = []
        
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'][input_name]
            
            for output_name in self.output_vars:
                output_data = prepared_data['outputs'][output_name]
                
                metrics = self.compute_univariate_metrics(
                    input_data, output_data, input_name, output_name
                )
                
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
        """
        Rank inputs by importance for each output.
        
        Args:
            metrics_df: DataFrame with computed metrics
            metric: Metric to use for ranking ('mutual_info', 'pearson_r', etc.)
            
        Returns:
            DataFrame with ranked inputs per output
        """
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
    
    def compute_response_surface(
        self,
        prepared_data: Dict[str, pd.DataFrame],
        input1: str,
        input2: str,
        output: str,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute response surface for two inputs and one output.
        
        Args:
            prepared_data: Prepared data dictionary
            input1: First input variable name
            input2: Second input variable name
            output: Output variable name
            n_points: Number of points per dimension
            
        Returns:
            Tuple of (input1_grid, input2_grid, output_grid)
        """
        logger.info(f"Computing response surface: {output} vs {input1} x {input2}")
        
        # Get data
        input1_data = prepared_data['inputs'][input1]
        input2_data = prepared_data['inputs'][input2]
        output_data = prepared_data['outputs'][output]
        
        # Flatten data
        if input1_data.ndim > 1:
            input1_flat = input1_data.flatten()
        else:
            input1_flat = np.repeat(input1_data, output_data.shape[1])
            
        if input2_data.ndim > 1:
            input2_flat = input2_data.flatten()
        else:
            input2_flat = np.repeat(input2_data, output_data.shape[1])
            
        output_flat = output_data.flatten()
        
        # Remove NaN
        mask = ~(np.isnan(input1_flat) | np.isnan(input2_flat) | np.isnan(output_flat))
        input1_clean = input1_flat[mask]
        input2_clean = input2_flat[mask]
        output_clean = output_flat[mask]
        
        # Create grid
        input1_range = np.linspace(input1_clean.min(), input1_clean.max(), n_points)
        input2_range = np.linspace(input2_clean.min(), input2_clean.max(), n_points)
        input1_grid, input2_grid = np.meshgrid(input1_range, input2_range)
        
        # Fit model (polynomial regression)
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X = poly.fit_transform(np.column_stack([input1_clean, input2_clean]))
        
        model = Ridge(alpha=1.0)
        model.fit(X, output_clean)
        
        # Predict on grid
        X_grid = poly.transform(np.column_stack([
            input1_grid.flatten(),
            input2_grid.flatten()
        ]))
        output_grid = model.predict(X_grid).reshape(input1_grid.shape)
        
        return input1_grid, input2_grid, output_grid
    
    def analyze_interaction_effects(
        self,
        prepared_data: Dict[str, pd.DataFrame],
        output: str
    ) -> Dict[str, float]:
        """
        Analyze interaction effects between inputs for a given output.
        
        Args:
            prepared_data: Prepared data dictionary
            output: Output variable name
            
        Returns:
            Dictionary with interaction metrics
        """
        logger.info(f"Analyzing interaction effects for {output}")
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        
        # Prepare input matrix
        inputs_list = []
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'][input_name]
            if input_data.ndim > 1:
                inputs_list.append(input_data.flatten())
            else:
                output_data = prepared_data['outputs'][output]
                inputs_list.append(np.repeat(input_data, output_data.shape[1]))
        
        X = np.column_stack(inputs_list)
        y = prepared_data['outputs'][output].flatten()
        
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
        
        # Interaction effect = improvement from adding interactions
        interaction_effect = r2_poly - r2_simple
        
        return {
            'output': output,
            'r2_linear': r2_simple,
            'r2_polynomial': r2_poly,
            'interaction_effect': interaction_effect
        }


class DataGenerator:
    """
    Generates simulation data suitable for direct effect analysis.
    """
    
    def __init__(self, simulator, num_cdus: int, config: Dict):
        """
        Initialize data generator.
        
        Args:
            simulator: FMU simulator instance
            num_cdus: Number of CDUs
            config: Configuration dictionary
        """
        self.simulator = simulator
        self.num_cdus = num_cdus
        self.config = config
        
    def generate_sensitivity_data(
        self,
        n_samples: int = 1000,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        transition_steps: int = 60  # Smooth transitions between samples
    ) -> pd.DataFrame:
        """
        Generate data for sensitivity analysis with varied inputs.
        Creates continuous time-series with smooth transitions between conditions.
        
        Args:
            n_samples: Number of different operating conditions to sample
            input_ranges: Custom ranges for inputs (min, max)
            transition_steps: Number of steps for smooth transition between conditions
            
        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Generating {n_samples} operating conditions with continuous time-series...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),  # kW
                'T_Air': (288.15, 308.15),     # K
                'T_ext': (283.15, 313.15)      # K
            }
        
        # Use Latin Hypercube Sampling for better coverage of operating conditions
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_samples)
        
        # Scale to input ranges - these are the target operating points
        q_flow_targets = qmc.scale(
            samples[:, 0:1],
            input_ranges['Q_flow'][0]*1000,
            input_ranges['Q_flow'][1]*1000
        ).flatten()
        
        t_air_targets = qmc.scale(
            samples[:, 1:2],
            input_ranges['T_Air'][0],
            input_ranges['T_Air'][1]
        ).flatten()
        
        t_ext_targets = qmc.scale(
            samples[:, 2:3],
            input_ranges['T_ext'][0],
            input_ranges['T_ext'][1]
        ).flatten()
        
        # Create continuous time-series by interpolating between target points
        total_steps = n_samples * transition_steps
        time_steps = np.arange(total_steps)
        
        # Interpolate smoothly between target values
        target_indices = np.linspace(0, n_samples - 1, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, total_steps)
        
        q_flow_series = np.interp(sample_indices, target_indices, q_flow_targets)
        t_air_series = np.interp(sample_indices, target_indices, t_air_targets)
        t_ext_series = np.interp(sample_indices, target_indices, t_ext_targets)
        
        # Create input DataFrame with proper FMU column names
        input_data_rows = []
        for i in range(total_steps):
            row = {
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext': t_ext_series[i]
            }
            
            # Add CDU-specific inputs with full FMU column names
            for cdu_idx in range(1, self.num_cdus + 1):
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = q_flow_series[i]
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = t_air_series[i]
            
            input_data_rows.append(row)
        
        input_df = pd.DataFrame(input_data_rows)
        
        logger.info(f"Created continuous time-series with {total_steps} steps")
        logger.info(f"Q_flow range: [{q_flow_series.min():.1f}, {q_flow_series.max():.1f}] W")
        logger.info(f"T_Air range: [{t_air_series.min():.1f}, {t_air_series.max():.1f}] K")
        logger.info(f"T_ext range: [{t_ext_series.min():.1f}, {t_ext_series.max():.1f}] K")
        
        
        try:
            results_df = self.simulator.run_simulation(
                input_data=input_df,
                stabilization_hours=3,  
                step_size=1,
                save_history=False
            )
            
            logger.info(f"Successfully generated {len(results_df)} samples")
            logger.info(f"Result columns: {list(results_df.columns)[:10]}...")  # Show first 10 columns
            
            return results_df
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    def generate_response_surface_data(
        self,
        input1: str,
        input2: str,
        input1_range: Tuple[float, float],
        input2_range: Tuple[float, float],
        n_points_per_dim: int = 20,
        fixed_inputs: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate data for response surface analysis.
        
        Args:
            input1: First input variable name
            input2: Second input variable name
            input1_range: Range for input1 (min, max)
            input2_range: Range for input2 (min, max)
            n_points_per_dim: Number of points per dimension
            fixed_inputs: Values for other inputs
            
        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Generating response surface data: {input1} x {input2}")
        
        # Create grid
        input1_vals = np.linspace(input1_range[0], input1_range[1], n_points_per_dim)
        input2_vals = np.linspace(input2_range[0], input2_range[1], n_points_per_dim)
        
        all_results = []
        
        for val1 in input1_vals:
            for val2 in input2_vals:
                # Create input dictionary
                inputs = fixed_inputs.copy() if fixed_inputs else {}
                
                # Set common input (T_ext)
                if input1 == 'T_ext':
                    inputs['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = val1
                if input2 == 'T_ext':
                    inputs['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = val2
                
                # Set CDU-specific inputs
                for cdu_idx in range(1, self.num_cdus + 1):
                    if input1 == 'Q_flow':
                        inputs[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = val1
                    if input1 == 'T_Air':
                        inputs[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = val1
                    
                    if input2 == 'Q_flow':
                        inputs[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = val2
                    if input2 == 'T_Air':
                        inputs[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = val2
                
                # Run simulation
                try:
                    result = self.simulator.simulate(inputs)
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Simulation failed: {e}")
                    continue
        
        return pd.DataFrame(all_results)