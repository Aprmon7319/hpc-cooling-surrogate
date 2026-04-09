"""
Threshold and saturation detection for FMU input-output relationships.

Provides segmented regression and breakpoint detection to identify
operating regimes with different behaviors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class ThresholdDetector:
    """
    Detects thresholds and saturation points in input-output relationships
    using segmented regression and breakpoint analysis.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        n_workers: int = 8, 
        threads_per_worker: int = 1, 
        memory_limit: str = '5GB',
        max_breakpoints: int = 3,
        **config_overrides
    ):
        """
        Initialize the threshold detector.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            max_breakpoints: Maximum number of breakpoints to detect
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
        self.max_breakpoints = max_breakpoints
        
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        self.client = None
        
        logger.info(f"ThresholdDetector initialized for system: {system_name}")
        logger.info(f"Max breakpoints: {max_breakpoints}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client."""
        if self.client is None:
            logger.info("Initializing Dask cluster for threshold detection...")
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
    
    @staticmethod
    def _piecewise_linear(x: np.ndarray, breakpoints: np.ndarray, slopes: np.ndarray, intercept: float) -> np.ndarray:
        """
        Compute piecewise linear function.
        
        Args:
            x: Input values
            breakpoints: Array of breakpoint locations
            slopes: Array of slopes for each segment
            intercept: Y-intercept at x=0
        
        Returns:
            Predicted y values
        """
        y = np.zeros_like(x, dtype=float)
        
        # First segment
        mask = x <= breakpoints[0] if len(breakpoints) > 0 else np.ones(len(x), dtype=bool)
        y[mask] = intercept + slopes[0] * x[mask]
        
        # Middle segments
        for i in range(len(breakpoints) - 1):
            mask = (x > breakpoints[i]) & (x <= breakpoints[i + 1])
            # Ensure continuity at breakpoint
            y_at_bp = intercept + slopes[0] * breakpoints[0]
            for j in range(i):
                y_at_bp += slopes[j + 1] * (breakpoints[j + 1] - breakpoints[j])
            y_at_bp += slopes[i + 1] * (breakpoints[i] - breakpoints[max(0, i - 1)] if i > 0 else 0)
            
            # Actually compute with continuity
            prev_y = intercept
            prev_x = 0
            for j in range(i + 1):
                segment_end = breakpoints[j]
                prev_y += slopes[j] * (segment_end - prev_x)
                prev_x = segment_end
            
            y[mask] = prev_y + slopes[i + 1] * (x[mask] - breakpoints[i])
        
        # Last segment
        if len(breakpoints) > 0:
            mask = x > breakpoints[-1]
            prev_y = intercept
            prev_x = 0
            for j in range(len(breakpoints)):
                segment_end = breakpoints[j]
                prev_y += slopes[j] * (segment_end - prev_x)
                prev_x = segment_end
            y[mask] = prev_y + slopes[-1] * (x[mask] - breakpoints[-1])
        
        return y
    

    @staticmethod
    def _fit_segmented_regression_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str,
        max_breakpoints: int = 3
    ) -> Dict:
        """
        Static method for fitting segmented regression with optimal breakpoints.
        Uses iterative algorithm to find best breakpoint locations.
        """
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
        
        if len(input_clean) < 100:
            return {}
        
        # Sort by input
        sort_idx = np.argsort(input_clean)
        x = input_clean[sort_idx]
        y = output_clean[sort_idx]
        
        results = {
            'input': input_name,
            'output': output_name,
            'n_samples': len(x),
            'breakpoints': [],
            'segments': [],
            'regime_changes': False
        }
        
        # Fit simple linear regression first
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_pred_linear = lr.predict(x.reshape(-1, 1))
        r2_linear = r2_score(y, y_pred_linear)
        rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
        
        results['linear'] = {
            'r2': r2_linear,
            'rmse': rmse_linear,
            'slope': float(lr.coef_[0]),
            'intercept': float(lr.intercept_)
        }
        
        best_breakpoints = []
        best_r2 = r2_linear
        best_segments = []
        
        # Try different numbers of breakpoints
        for n_bp in range(1, max_breakpoints + 1):
            # Use quantiles as initial breakpoint locations
            bp_percentiles = np.linspace(20, 80, n_bp)
            initial_bp = np.percentile(x, bp_percentiles)
            
            def objective(bp_locs):
                """Objective function: negative R² for segmented fit."""
                bp_sorted = np.sort(bp_locs)
                
                try:
                    # Fit separate linear models for each segment
                    segments = []
                    y_pred = np.zeros_like(y)
                    
                    boundaries = [x.min()] + list(bp_sorted) + [x.max()]
                    
                    for i in range(len(boundaries) - 1):
                        mask_seg = (x >= boundaries[i]) & (x <= boundaries[i + 1])
                        if np.sum(mask_seg) < 10:
                            return 1e10  # Penalty for too small segments
                        
                        x_seg = x[mask_seg].reshape(-1, 1)
                        y_seg = y[mask_seg]
                        
                        lr_seg = LinearRegression()
                        lr_seg.fit(x_seg, y_seg)
                        y_pred[mask_seg] = lr_seg.predict(x_seg)
                        
                        segments.append({
                            'x_start': float(boundaries[i]),
                            'x_end': float(boundaries[i + 1]),
                            'slope': float(lr_seg.coef_[0]),
                            'intercept': float(lr_seg.intercept_),
                            'n_points': int(np.sum(mask_seg))
                        })
                    
                    r2 = r2_score(y, y_pred)
                    return -r2
                    
                except Exception:
                    return 1e10
            
            # Optimize breakpoint locations
            bounds = [(x.min() + 0.1 * (x.max() - x.min()), 
                      x.max() - 0.1 * (x.max() - x.min()))] * n_bp
            
            try:
                result = minimize(
                    objective,
                    initial_bp,
                    method='L-BFGS-B',
                    bounds=bounds
                )
                
                if result.success or result.fun < 0:
                    bp_optimized = np.sort(result.x)
                    
                    # Compute final fit with optimized breakpoints
                    segments = []
                    y_pred = np.zeros_like(y)
                    
                    boundaries = [x.min()] + list(bp_optimized) + [x.max()]
                    
                    for i in range(len(boundaries) - 1):
                        mask_seg = (x >= boundaries[i]) & (x <= boundaries[i + 1])
                        
                        x_seg = x[mask_seg].reshape(-1, 1)
                        y_seg = y[mask_seg]
                        
                        lr_seg = LinearRegression()
                        lr_seg.fit(x_seg, y_seg)
                        y_pred[mask_seg] = lr_seg.predict(x_seg)
                        
                        segments.append({
                            'x_start': float(boundaries[i]),
                            'x_end': float(boundaries[i + 1]),
                            'slope': float(lr_seg.coef_[0]),
                            'intercept': float(lr_seg.intercept_),
                            'n_points': int(np.sum(mask_seg))
                        })
                    
                    r2 = r2_score(y, y_pred)
                    
                    # Use AIC/BIC for model selection
                    n = len(y)
                    k = 2 * (n_bp + 1)  # 2 params per segment
                    residuals = y - y_pred
                    mse = np.mean(residuals ** 2)
                    log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    # Check if this is better than previous best
                    # Use BIC improvement threshold
                    if r2 > best_r2 + 0.01:  # Require meaningful improvement
                        best_r2 = r2
                        best_breakpoints = bp_optimized.tolist()
                        best_segments = segments
                        
                        results[f'{n_bp}_breakpoints'] = {
                            'breakpoints': bp_optimized.tolist(),
                            'segments': segments,
                            'r2': r2,
                            'rmse': np.sqrt(mse),
                            'aic': aic,
                            'bic': bic
                        }
            except Exception as e:
                continue
        
        # Store best results
        results['breakpoints'] = best_breakpoints
        results['segments'] = best_segments
        results['best_r2'] = best_r2
        results['r2_improvement'] = best_r2 - r2_linear
        
        # Detect if there are significant regime changes
        if best_segments and len(best_segments) > 1:
            slopes = [s['slope'] for s in best_segments]
            max_slope_diff = max(slopes) - min(slopes)
            avg_slope = np.mean(np.abs(slopes))
            
            # Significant regime change if slope varies by more than 50%
            results['regime_changes'] = max_slope_diff > 0.5 * avg_slope if avg_slope > 0 else False
            
            # Check for saturation (slope near zero in a segment)
            results['has_saturation'] = any(abs(s['slope']) < 0.1 * avg_slope for s in best_segments) if avg_slope > 0 else False
            
            # Check for threshold behavior (abrupt slope change)
            if len(slopes) > 1:
                slope_changes = [abs(slopes[i+1] - slopes[i]) for i in range(len(slopes)-1)]
                results['has_threshold'] = any(sc > 0.3 * avg_slope for sc in slope_changes) if avg_slope > 0 else False
            else:
                results['has_threshold'] = False
        else:
            results['has_saturation'] = False
            results['has_threshold'] = False
        
        return results
    

    @staticmethod
    def _classify_operating_regimes_static(
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_name: str,
        output_name: str,
        n_regimes: int = 3
    ) -> Dict:
        """
        Classify data points into operating regimes using clustering.
        """
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
        
        # Remove NaN
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        input_clean = input_flat[mask]
        output_clean = output_flat[mask]
        
        if len(input_clean) < 50:
            return {}
        
        # Standardize for clustering
        scaler = StandardScaler()
        X = np.column_stack([input_clean, output_clean])
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouettes = []
        
        from sklearn.metrics import silhouette_score
        
        for k in range(2, min(n_regimes + 2, len(X_scaled) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Use silhouette score to pick optimal k
        if silhouettes:
            optimal_k = 2 + np.argmax(silhouettes)
        else:
            optimal_k = 2
        
        # Fit with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Characterize each regime
        regimes = []
        for regime in range(optimal_k):
            mask_regime = labels == regime
            
            x_regime = input_clean[mask_regime]
            y_regime = output_clean[mask_regime]
            
            # Fit linear model within regime
            if len(x_regime) > 10:
                lr = LinearRegression()
                lr.fit(x_regime.reshape(-1, 1), y_regime)
                r2 = r2_score(y_regime, lr.predict(x_regime.reshape(-1, 1)))
                
                regimes.append({
                    'regime_id': regime,
                    'n_points': int(np.sum(mask_regime)),
                    'x_range': [float(x_regime.min()), float(x_regime.max())],
                    'y_range': [float(y_regime.min()), float(y_regime.max())],
                    'x_mean': float(x_regime.mean()),
                    'y_mean': float(y_regime.mean()),
                    'slope': float(lr.coef_[0]),
                    'intercept': float(lr.intercept_),
                    'r2': r2
                })
        
        return {
            'input': input_name,
            'output': output_name,
            'n_regimes': optimal_k,
            'regimes': regimes,
            'labels': labels,
            'silhouette_score': silhouettes[optimal_k - 2] if optimal_k - 2 < len(silhouettes) else 0,
            'x_data': input_clean,
            'y_data': output_clean
        }


    def detect_thresholds_all_pairs(
        self,
        prepared_data: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Detect thresholds for all input-output pairs.
        
        Returns:
            Tuple of (summary DataFrame, full results list)
        """
        logger.info("Detecting thresholds for all input-output pairs...")
        
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
                
                task = delayed(self._fit_segmented_regression_static)(
                    input_data, output_data, input_name, output_name,
                    self.max_breakpoints
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        logger.info(f"Executing {len(delayed_tasks)} threshold detection tasks...")
        results_list = dask.compute(*delayed_tasks)
        
        # Process results
        summary_data = []
        full_results = []
        
        for (input_name, output_name), result in zip(task_info, results_list):
            if result:
                full_results.append(result)
                
                summary_data.append({
                    'input': input_name,
                    'output': output_name,
                    'n_breakpoints': len(result.get('breakpoints', [])),
                    'linear_r2': result.get('linear', {}).get('r2', 0),
                    'segmented_r2': result.get('best_r2', 0),
                    'r2_improvement': result.get('r2_improvement', 0),
                    'has_threshold': result.get('has_threshold', False),
                    'has_saturation': result.get('has_saturation', False),
                    'regime_changes': result.get('regime_changes', False)
                })
        
        logger.info(f"Detected thresholds for {len(summary_data)} pairs")
        
        return pd.DataFrame(summary_data), full_results
    
    def classify_regimes_all_pairs(
        self,
        prepared_data: Dict[str, np.ndarray],
        n_regimes: int = 3
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Classify operating regimes for all input-output pairs.
        
        Returns:
            Tuple of (summary DataFrame, full results with labels)
        """
        logger.info("Classifying operating regimes for all pairs...")
        
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
                
                task = delayed(self._classify_operating_regimes_static)(
                    input_data, output_data, input_name, output_name, n_regimes
                )
                delayed_tasks.append(task)
                task_info.append((input_name, output_name))
        
        logger.info(f"Executing {len(delayed_tasks)} regime classification tasks...")
        results_list = dask.compute(*delayed_tasks)
        
        summary_data = []
        full_results = []
        
        for (input_name, output_name), result in zip(task_info, results_list):
            if result:
                full_results.append(result)
                
                summary_data.append({
                    'input': input_name,
                    'output': output_name,
                    'n_regimes': result.get('n_regimes', 0),
                    'silhouette_score': result.get('silhouette_score', 0)
                })
        
        logger.info(f"Classified regimes for {len(summary_data)} pairs")
        
        return pd.DataFrame(summary_data), full_results
    

    def get_segmented_fit_data(
        self,
        prepared_data: Dict[str, np.ndarray],
        input_name: str,
        output_name: str
    ) -> Dict:
        """
        Get segmented regression fit data for visualization.
        """
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
        
        result = self._fit_segmented_regression_static(
            input_data, output_data, input_name, output_name,
            self.max_breakpoints
        )
        
        if not result:
            return {}
        
        # Prepare visualization data - handle different dimensionalities
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
        
        mask = ~(np.isnan(input_flat) | np.isnan(output_flat))
        x = input_flat[mask]
        y = output_flat[mask]
        
        # Create prediction lines
        x_pred = np.linspace(x.min(), x.max(), 500)
        
        # Linear prediction
        y_linear = result['linear']['intercept'] + result['linear']['slope'] * x_pred
        
        # Segmented prediction
        y_segmented = np.zeros_like(x_pred)
        segments = result.get('segments', [])
        
        if segments:
            for seg in segments:
                mask_seg = (x_pred >= seg['x_start']) & (x_pred <= seg['x_end'])
                y_segmented[mask_seg] = seg['intercept'] + seg['slope'] * x_pred[mask_seg]
        else:
            y_segmented = y_linear.copy()
        
        return {
            'x_data': x,
            'y_data': y,
            'x_pred': x_pred,
            'y_linear': y_linear,
            'y_segmented': y_segmented,
            'breakpoints': result.get('breakpoints', []),
            'segments': segments,
            'linear_r2': result['linear']['r2'],
            'segmented_r2': result.get('best_r2', result['linear']['r2'])
        }