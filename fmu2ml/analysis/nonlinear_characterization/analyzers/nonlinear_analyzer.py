"""
Non-linear relationship analyzer for FMU input-output characterization.

Provides polynomial/spline fitting, model comparison, and non-linearity 
strength metrics for understanding input-output relationships.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class NonlinearAnalyzer:
    """
    Analyzes non-linear relationships between inputs and outputs using
    polynomial/spline fitting and model comparison.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        max_polynomial_degree: int = 5,
        **config_overrides
    ):
        """
        Initialize the non-linear analyzer.
        
        Args:
            system_name: System configuration name (e.g., 'marconi100', 'leonardo')
            n_workers: Number of Dask workers for parallel analysis
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            max_polynomial_degree: Maximum polynomial degree to test
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
        self.max_polynomial_degree = max_polynomial_degree
        
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        # Additional datacenter-level outputs
        self.datacenter_output_vars = [
            'V_flow_prim_GPM_datacenter',
            'pue'
        ]
        
        # CDU-level additional outputs
        self.cdu_additional_vars = [
            'htc'
        ]
        
        self.client = None
        
        logger.info(f"NonlinearAnalyzer initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
        logger.info(f"Max polynomial degree: {max_polynomial_degree}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client for analysis tasks."""
        if self.client is None:
            logger.info("Initializing Dask cluster for non-linear analysis...")
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
        logger.info("Preparing data for non-linear analysis...")
        
        prepared_data = {
            'inputs': {},
            'outputs': {},
            'datacenter_outputs': {},
            'time': None
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
            
            # CDU-level additional outputs (htc)
            for add_var in self.cdu_additional_vars:
                htc_col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cabinet[1].summary.{add_var}'
                
                if htc_col in data.columns:
                    if add_var not in prepared_data['outputs']:
                        prepared_data['outputs'][add_var] = data[htc_col].values.reshape(-1, 1)
                    else:
                        prepared_data['outputs'][add_var] = np.hstack([
                            prepared_data['outputs'][add_var],
                            data[htc_col].values.reshape(-1, 1)
                        ])
        
        # Datacenter-level outputs
        dc_vflow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if dc_vflow_col in data.columns:
            prepared_data['datacenter_outputs']['V_flow_prim_GPM_datacenter'] = data[dc_vflow_col].values
        
        # PUE if available
        pue_col = 'pue'
        if pue_col in data.columns:
            prepared_data['datacenter_outputs']['pue'] = data[pue_col].values
        
        # Time column
        if 'time' in data.columns:
            prepared_data['time'] = data['time'].values
        
        logger.info(f"Data prepared: {len(prepared_data['inputs'])} inputs, "
                   f"{len(prepared_data['outputs'])} CDU outputs, "
                   f"{len(prepared_data['datacenter_outputs'])} datacenter outputs")
        
        return prepared_data
    @staticmethod
    def _fit_polynomial_models_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str,
        max_degree: int = 5,
        cv_folds: int = 5
    ) -> Dict:
        """
        Static method for fitting polynomial models of increasing degree.
        Used for parallel execution with Dask.
        
        Returns:
            Dictionary with model fits and metrics for each degree
        """
        # Handle different input/output dimensionalities
        # Input can be 1D (shared across CDUs like T_ext) or 2D (per-CDU like Q_flow)
        # Output is typically 2D (per-CDU)
        
        if output_data.ndim == 1:
            output_flat = output_data.flatten()
            n_samples = len(output_flat)
            
            if input_data.ndim == 1:
                input_flat = input_data.flatten()
            else:
                # Average across CDUs if input is 2D but output is 1D
                input_flat = np.mean(input_data, axis=1) if input_data.shape[0] == n_samples else input_data.flatten()
        else:
            # Output is 2D: (n_timesteps, n_cdus)
            n_timesteps, n_cdus = output_data.shape
            output_flat = output_data.flatten()
            
            if input_data.ndim == 1:
                # Input is 1D (like T_ext): repeat for each CDU
                # Shape should be (n_timesteps,), need to tile for each CDU
                input_flat = np.tile(input_data, n_cdus)
            elif input_data.ndim == 2:
                # Input is 2D (like Q_flow per CDU): flatten directly
                input_flat = input_data.flatten()
            else:
                input_flat = input_data.flatten()
        
        # Ensure shapes match
        if len(input_flat) != len(output_flat):
            # Try to match shapes by repeating or truncating
            if len(input_flat) < len(output_flat):
                # Repeat input to match output length
                repeat_factor = len(output_flat) // len(input_flat)
                if repeat_factor * len(input_flat) == len(output_flat):
                    input_flat = np.tile(input_flat, repeat_factor)
                else:
                    # Cannot match shapes, return empty
                    return {}
            else:
                # Truncate or subsample
                return {}
        
        # Remove NaN values
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        input_clean = input_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input_clean) < 50:
            return {}
        
        # Standardize inputs for numerical stability
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_clean.reshape(-1, 1))
        
        results = {
            'input': input_name,
            'output': output_name,
            'n_samples': len(input_clean),
            'models': {},
            'best_degree': 1,
            'best_r2': 0,
            'nonlinearity_strength': 0
        }
        
        linear_r2 = None
        best_r2 = 0
        best_degree = 1
        
        for degree in range(1, max_degree + 1):
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                X_poly = poly.fit_transform(input_scaled)
                
                model = Ridge(alpha=1.0)
                model.fit(X_poly, output_clean)
                
                y_pred = model.predict(X_poly)
                r2 = r2_score(output_clean, y_pred)
                rmse = np.sqrt(mean_squared_error(output_clean, y_pred))
                
                # Cross-validation for model selection
                cv_splits = min(cv_folds, len(input_clean) // 10)
                if cv_splits >= 2:
                    cv_scores = cross_val_score(
                        model, X_poly, output_clean,
                        cv=cv_splits,
                        scoring='r2'
                    )
                    cv_r2_mean = cv_scores.mean()
                    cv_r2_std = cv_scores.std()
                else:
                    cv_r2_mean = r2
                    cv_r2_std = 0.0
                
                # Compute AIC and BIC for model selection
                n = len(output_clean)
                k = X_poly.shape[1]  # number of parameters
                residuals = output_clean - y_pred
                mse = np.mean(residuals ** 2)
                
                # Log-likelihood (assuming Gaussian errors)
                log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
                
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                # Store coefficients
                coef = model.coef_
                intercept = model.intercept_
                
                results['models'][degree] = {
                    'r2': r2,
                    'rmse': rmse,
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': cv_r2_std,
                    'aic': aic,
                    'bic': bic,
                    'coefficients': coef.tolist(),
                    'intercept': intercept,
                    'n_parameters': k
                }
                
                if degree == 1:
                    linear_r2 = r2
                
                if cv_r2_mean > best_r2:
                    best_r2 = cv_r2_mean
                    best_degree = degree
                    
            except Exception as e:
                logger.warning(f"Failed to fit degree {degree} polynomial: {e}")
                continue
        
        results['best_degree'] = best_degree
        results['best_r2'] = best_r2
        
        # Compute non-linearity strength as absolute R² improvement (percentage points)
        # Clamp R² values to [0, 1] to avoid inflated values when linear R² is very negative
        # (which happens when a single input explains little variance in multi-CDU data)
        if linear_r2 is not None:
            linear_r2_clamped = max(linear_r2, 0.0)
            best_nonlinear_r2 = max(
                [m['r2'] for d, m in results['models'].items() if d > 1],
                default=linear_r2
            )
            best_nonlinear_r2_clamped = max(best_nonlinear_r2, 0.0)
            results['nonlinearity_strength'] = (best_nonlinear_r2_clamped - linear_r2_clamped) * 100  # percentage points
            results['linear_r2_raw'] = linear_r2
            results['best_nonlinear_r2_raw'] = best_nonlinear_r2
        else:
            results['nonlinearity_strength'] = 0
        
        # Residual analysis for the best model
        if best_degree in results['models']:
            poly = PolynomialFeatures(degree=best_degree, include_bias=True)
            X_poly = poly.fit_transform(input_scaled)
            model = Ridge(alpha=1.0)
            model.fit(X_poly, output_clean)
            residuals = output_clean - model.predict(X_poly)
            
            # Residual statistics
            results['residual_stats'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'normality_pvalue': float(stats.normaltest(residuals)[1]) if len(residuals) >= 20 else None
            }
        
        return results
    

    @staticmethod
    def _fit_spline_model_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str,
        smoothing_factors: List[float] = None
    ) -> Dict:
        """
        Static method for fitting spline models with different smoothing factors.
        """
        if smoothing_factors is None:
            smoothing_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        # Handle different input/output dimensionalities
        if output_data.ndim == 1:
            output_flat = output_data.flatten()
            n_samples = len(output_flat)
            
            if input_data.ndim == 1:
                input_flat = input_data.flatten()
            else:
                input_flat = np.mean(input_data, axis=1) if input_data.shape[0] == n_samples else input_data.flatten()
        else:
            n_timesteps, n_cdus = output_data.shape
            output_flat = output_data.flatten()
            
            if input_data.ndim == 1:
                input_flat = np.tile(input_data, n_cdus)
            elif input_data.ndim == 2:
                input_flat = input_data.flatten()
            else:
                input_flat = input_data.flatten()
        
        # Ensure shapes match
        if len(input_flat) != len(output_flat):
            if len(input_flat) < len(output_flat):
                repeat_factor = len(output_flat) // len(input_flat)
                if repeat_factor * len(input_flat) == len(output_flat):
                    input_flat = np.tile(input_flat, repeat_factor)
                else:
                    return {}
            else:
                return {}
        
        # Remove NaN and sort
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        input_clean = input_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input_clean) < 50:
            return {}
        
        # Sort by input for spline fitting
        sort_idx = np.argsort(input_clean)
        input_sorted = input_clean[sort_idx]
        output_sorted = output_clean[sort_idx]
        
        # Aggregate points with same x value
        unique_x, idx = np.unique(input_sorted, return_inverse=True)
        unique_y = np.array([output_sorted[idx == i].mean() for i in range(len(unique_x))])
        
        if len(unique_x) < 10:
            return {}
        
        results = {
            'input': input_name,
            'output': output_name,
            'splines': {},
            'best_smoothing': None,
            'best_r2': 0
        }
        
        best_r2 = 0
        best_smoothing = None
        
        for s in smoothing_factors:
            try:
                spline = UnivariateSpline(unique_x, unique_y, s=s * len(unique_x))
                
                y_pred = spline(input_clean)
                r2 = r2_score(output_clean, y_pred)
                rmse = np.sqrt(mean_squared_error(output_clean, y_pred))
                
                results['splines'][s] = {
                    'r2': r2,
                    'rmse': rmse,
                    'knots': spline.get_knots().tolist(),
                    'n_knots': len(spline.get_knots())
                }
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_smoothing = s
                    
            except Exception as e:
                continue
        
        results['best_smoothing'] = best_smoothing
        results['best_r2'] = best_r2
        
        return results
    def analyze_polynomial_fits(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze all input-output pairs with polynomial fitting.
        
        Returns:
            DataFrame with polynomial fit results for all pairs
        """
        logger.info("Computing polynomial fits for all input-output pairs...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        task_info = []
        
        # Get all output variables (CDU-level and datacenter-level)
        all_outputs = list(prepared_data['outputs'].keys())
        all_outputs.extend(prepared_data.get('datacenter_outputs', {}).keys())
        
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'][input_name]
            
            for output_name in all_outputs:
                if output_name in prepared_data['outputs']:
                    output_data = prepared_data['outputs'][output_name]
                else:
                    output_data = prepared_data['datacenter_outputs'][output_name]
                
                task = delayed(self._fit_polynomial_models_static)(
                    input_data, output_data, input_name, output_name,
                    self.max_polynomial_degree
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        logger.info(f"Executing {len(delayed_tasks)} polynomial fit tasks in parallel...")
        results_list = dask.compute(*delayed_tasks)
        
        # Flatten results for DataFrame
        flat_results = []
        full_results = []
        
        for (input_name, output_name), result in zip(task_info, results_list):
            if result:
                full_results.append(result)
                
                # Flatten for summary DataFrame
                flat_result = {
                    'input': input_name,
                    'output': output_name,
                    'n_samples': result.get('n_samples', 0),
                    'best_degree': result.get('best_degree', 1),
                    'best_r2': result.get('best_r2', 0),
                    'nonlinearity_strength': result.get('nonlinearity_strength', 0)
                }
                
                # Add R² for each degree
                for deg in range(1, self.max_polynomial_degree + 1):
                    if deg in result.get('models', {}):
                        flat_result[f'r2_degree_{deg}'] = result['models'][deg]['r2']
                        flat_result[f'aic_degree_{deg}'] = result['models'][deg]['aic']
                        flat_result[f'bic_degree_{deg}'] = result['models'][deg]['bic']
                
                flat_results.append(flat_result)
        
        logger.info(f"Computed polynomial fits for {len(flat_results)} pairs")
        
        return pd.DataFrame(flat_results), full_results
    
    def analyze_spline_fits(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze all input-output pairs with spline fitting.
        """
        logger.info("Computing spline fits for all input-output pairs...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        task_info = []
        
        all_outputs = list(prepared_data['outputs'].keys())
        all_outputs.extend(prepared_data.get('datacenter_outputs', {}).keys())
        
        for input_name in self.input_vars:
            input_data = prepared_data['inputs'][input_name]
            
            for output_name in all_outputs:
                if output_name in prepared_data['outputs']:
                    output_data = prepared_data['outputs'][output_name]
                else:
                    output_data = prepared_data['datacenter_outputs'][output_name]
                
                task = delayed(self._fit_spline_model_static)(
                    input_data, output_data, input_name, output_name
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        logger.info(f"Executing {len(delayed_tasks)} spline fit tasks in parallel...")
        results_list = dask.compute(*delayed_tasks)
        
        results = []
        for (input_name, output_name), result in zip(task_info, results_list):
            if result:
                results.append({
                    'input': input_name,
                    'output': output_name,
                    'best_smoothing': result.get('best_smoothing'),
                    'best_r2': result.get('best_r2', 0)
                })
        
        logger.info(f"Computed spline fits for {len(results)} pairs")
        
        return pd.DataFrame(results)
    
    def compute_model_comparison(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare linear vs. non-linear model performance across all pairs.
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Computing model comparison (linear vs. non-linear)...")
        
        poly_df, full_results = self.analyze_polynomial_fits(prepared_data)
        
        comparison_data = []
        
        for result in full_results:
            if not result or 'models' not in result:
                continue
            
            models = result['models']
            if 1 not in models:
                continue
            
            linear_r2 = models[1]['r2']
            linear_aic = models[1]['aic']
            linear_bic = models[1]['bic']
            
            # Find best non-linear model
            best_nonlinear_r2 = linear_r2
            best_nonlinear_degree = 1
            best_nonlinear_aic = linear_aic
            
            for deg in range(2, self.max_polynomial_degree + 1):
                if deg in models:
                    if models[deg]['aic'] < best_nonlinear_aic:
                        best_nonlinear_aic = models[deg]['aic']
                        best_nonlinear_r2 = models[deg]['r2']
                        best_nonlinear_degree = deg
            
            # Clamp R² values to [-1, 1] range for meaningful comparison
            # Very negative R² indicates the model is worse than predicting the mean,
            # which is common when a single input explains little variance in flattened
            # multi-CDU data. We clamp to avoid astronomically inflated improvement values.
            linear_r2_clamped = max(linear_r2, 0.0)
            best_nonlinear_r2_clamped = max(best_nonlinear_r2, 0.0)
            
            # Absolute R² improvement (percentage points) using clamped values
            r2_improvement = best_nonlinear_r2_clamped - linear_r2_clamped
            r2_improvement_pct = r2_improvement * 100  # percentage points
            
            # Also compute relative improvement safely for cases where linear R² > 0
            if linear_r2_clamped > 0.01:
                r2_improvement_relative_pct = (r2_improvement / linear_r2_clamped) * 100
            else:
                # When linear R² is near zero, relative improvement is not meaningful
                # Use absolute improvement instead
                r2_improvement_relative_pct = r2_improvement_pct
            
            comparison_data.append({
                'input': result['input'],
                'output': result['output'],
                'linear_r2': linear_r2,
                'linear_r2_clamped': linear_r2_clamped,
                'best_nonlinear_r2': best_nonlinear_r2,
                'best_nonlinear_r2_clamped': best_nonlinear_r2_clamped,
                'best_degree': best_nonlinear_degree,
                'r2_improvement': r2_improvement,
                'r2_improvement_pct': r2_improvement_pct,
                'r2_improvement_relative_pct': r2_improvement_relative_pct,
                'linear_r2_raw': linear_r2,  # Keep raw values for debugging
                'best_nonlinear_r2_raw': best_nonlinear_r2,
                'linear_aic': linear_aic,
                'best_nonlinear_aic': best_nonlinear_aic,
                'aic_improvement': linear_aic - best_nonlinear_aic,
                'strongly_nonlinear': r2_improvement_pct > 5,  # >5 percentage points improvement
                'recommended_degree': best_nonlinear_degree if r2_improvement_pct > 5 else 1
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Log warnings for pairs with very negative raw R² values
        if len(comparison_df) > 0:
            negative_r2 = comparison_df[comparison_df['linear_r2_raw'] < -1.0]
            if len(negative_r2) > 0:
                logger.warning(
                    f"{len(negative_r2)} pairs have very negative linear R² (< -1.0). "
                    f"This typically indicates the single input variable explains very little "
                    f"variance in the flattened multi-CDU output data. R² values have been "
                    f"clamped to 0 for non-linearity strength computation."
                )
                for _, row in negative_r2.iterrows():
                    logger.warning(
                        f"  {row['input']} → {row['output']}: "
                        f"linear R²={row['linear_r2_raw']:.4f}, "
                        f"best nonlinear R²={row['best_nonlinear_r2_raw']:.4f}"
                    )
        
        logger.info(f"Computed comparison for {len(comparison_df)} pairs")
        
        return comparison_df
    

    def get_prediction_data(
        self,
        prepared_data: Dict[str, np.ndarray],
        input_name: str,
        output_name: str,
        degrees: List[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get prediction data for visualization.
        
        Returns:
            Dictionary with x values and predictions for each model degree
        """
        if degrees is None:
            degrees = [1, 2, 3]
        
        if input_name in prepared_data['inputs']:
            input_data = prepared_data['inputs'][input_name]
        else:
            return {}
        
        if output_name in prepared_data['outputs']:
            output_data = prepared_data['outputs'][output_name]
        elif output_name in prepared_data.get('datacenter_outputs', {}):
            output_data = prepared_data['datacenter_outputs'][output_name]
        else:
            return {}
        
        # Handle different input/output dimensionalities
        if output_data.ndim == 1:
            output_flat = output_data.flatten()
            n_samples = len(output_flat)
            
            if input_data.ndim == 1:
                input_flat = input_data.flatten()
            else:
                input_flat = np.mean(input_data, axis=1) if input_data.shape[0] == n_samples else input_data.flatten()
        else:
            n_timesteps, n_cdus = output_data.shape
            output_flat = output_data.flatten()
            
            if input_data.ndim == 1:
                input_flat = np.tile(input_data, n_cdus)
            elif input_data.ndim == 2:
                input_flat = input_data.flatten()
            else:
                input_flat = input_data.flatten()
        
        # Ensure shapes match
        if len(input_flat) != len(output_flat):
            if len(input_flat) < len(output_flat):
                repeat_factor = len(output_flat) // len(input_flat)
                if repeat_factor * len(input_flat) == len(output_flat):
                    input_flat = np.tile(input_flat, repeat_factor)
                else:
                    return {}
            else:
                return {}
        
        # Clean data
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        input_clean = input_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input_clean) < 50:
            return {}
        
        # Create prediction grid
        x_min, x_max = input_clean.min(), input_clean.max()
        x_pred = np.linspace(x_min, x_max, 200)
        
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_clean.reshape(-1, 1))
        x_pred_scaled = scaler.transform(x_pred.reshape(-1, 1))
        
        predictions = {
            'x': x_pred,
            'x_data': input_clean,
            'y_data': output_clean
        }
        
        for degree in degrees:
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                X_poly = poly.fit_transform(input_scaled)
                X_pred_poly = poly.transform(x_pred_scaled)
                
                model = Ridge(alpha=1.0)
                model.fit(X_poly, output_clean)
                
                predictions[f'y_degree_{degree}'] = model.predict(X_pred_poly)
                
                # Compute residuals
                y_fitted = model.predict(X_poly)
                predictions[f'residuals_degree_{degree}'] = output_clean - y_fitted
                predictions[f'r2_degree_{degree}'] = r2_score(output_clean, y_fitted)
                
            except Exception as e:
                logger.warning(f"Failed to compute predictions for degree {degree}: {e}")
        
        return predictions