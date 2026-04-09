"""
Impulse Response Analyzer

Analyzes system dynamics by:
1. Identifying step changes (impulse events) in inputs
2. Measuring output response characteristics
3. Characterizing time constants and gains
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class ImpulseResponseAnalyzer:
    """
    Analyzes impulse/step responses to characterize system dynamics.
    Measures rise time, settling time, overshoot, and gain.
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
        Initialize the impulse response analyzer.
        
        Args:
            system_name: System configuration name
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
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
        
        self.input_vars = ['Q_flow', 'T_Air', 'T_ext']
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        self.client = None
        self._impulse_events = None
        self._response_curves = None
        
        logger.info(f"ImpulseResponseAnalyzer initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
    
    def _init_dask_client(self):
        """Initialize Dask distributed client."""
        if self.client is None:
            logger.info("Initializing Dask cluster for impulse response analysis...")
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
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare data for impulse response analysis.
        
        Args:
            data: Raw simulation data
            
        Returns:
            Prepared data dictionary
        """
        logger.info("Preparing data for impulse response analysis...")
        
        prepared_data = {
            'inputs': {},
            'outputs': {},
            'time': None
        }
        
        # Extract time
        if 'time' in data.columns:
            prepared_data['time'] = data['time'].values
        else:
            prepared_data['time'] = np.arange(len(data))
        
        # Aggregate inputs (average across CDUs for detection)
        q_flow_cols = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total' 
                       for i in range(1, self.num_cdus + 1)]
        t_air_cols = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air' 
                      for i in range(1, self.num_cdus + 1)]
        
        available_q_cols = [c for c in q_flow_cols if c in data.columns]
        available_t_cols = [c for c in t_air_cols if c in data.columns]
        
        if available_q_cols:
            prepared_data['inputs']['Q_flow'] = data[available_q_cols].mean(axis=1).values
        if available_t_cols:
            prepared_data['inputs']['T_Air'] = data[available_t_cols].mean(axis=1).values
        
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        if t_ext_col in data.columns:
            prepared_data['inputs']['T_ext'] = data[t_ext_col].values
        
        # Aggregate outputs (average across CDUs)
        for output_var in self.output_vars:
            output_cols = [f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{output_var}' 
                          for i in range(1, self.num_cdus + 1)]
            available_cols = [c for c in output_cols if c in data.columns]
            
            if available_cols:
                prepared_data['outputs'][output_var] = data[available_cols].mean(axis=1).values
        
        # Datacenter-level outputs
        dc_v_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if dc_v_flow_col in data.columns:
            prepared_data['outputs']['V_flow_prim_GPM_dc'] = data[dc_v_flow_col].values
        
        logger.info(f"Prepared {len(prepared_data['inputs'])} inputs, {len(prepared_data['outputs'])} outputs")
        return prepared_data
    
    def detect_step_changes(
        self,
        prepared_data: Dict[str, np.ndarray],
        threshold_percentile: float = 90,
        min_change_fraction: float = 0.1,
        min_duration: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Detect step changes (impulse events) in input signals.
        
        Args:
            prepared_data: Prepared data dictionary
            threshold_percentile: Percentile threshold for detecting changes
            min_change_fraction: Minimum change as fraction of signal range
            min_duration: Minimum duration between detected steps
            
        Returns:
            Dictionary mapping input names to list of step events
        """
        logger.info("Detecting step changes in inputs...")
        
        step_events = {}
        
        for input_name, input_data in prepared_data['inputs'].items():
            events = []
            
            # Compute derivative
            derivative = np.gradient(input_data)
            
            # Compute threshold
            signal_range = np.nanmax(input_data) - np.nanmin(input_data)
            min_change = min_change_fraction * signal_range
            
            # Find large changes
            abs_derivative = np.abs(derivative)
            threshold = np.nanpercentile(abs_derivative, threshold_percentile)
            threshold = max(threshold, min_change / 2)  # Ensure meaningful threshold
            
            # Find step locations
            step_mask = abs_derivative > threshold
            step_indices = np.where(step_mask)[0]
            
            if len(step_indices) == 0:
                logger.warning(f"No step changes detected in {input_name}")
                step_events[input_name] = []
                continue
            
            # Cluster nearby steps
            clustered_steps = []
            current_cluster = [step_indices[0]]
            
            for idx in step_indices[1:]:
                if idx - current_cluster[-1] <= min_duration:
                    current_cluster.append(idx)
                else:
                    # Store cluster center
                    center = int(np.mean(current_cluster))
                    clustered_steps.append(center)
                    current_cluster = [idx]
            
            # Don't forget last cluster
            if current_cluster:
                center = int(np.mean(current_cluster))
                clustered_steps.append(center)
            
            # Create event records
            for step_idx in clustered_steps:
                # Get values before and after
                pre_start = max(0, step_idx - min_duration)
                post_end = min(len(input_data), step_idx + min_duration)
                
                pre_value = np.nanmean(input_data[pre_start:step_idx])
                post_value = np.nanmean(input_data[step_idx:post_end])
                
                step_magnitude = post_value - pre_value
                
                if abs(step_magnitude) >= min_change:
                    events.append({
                        'step_index': step_idx,
                        'time': prepared_data['time'][step_idx] if prepared_data['time'] is not None else step_idx,
                        'pre_value': pre_value,
                        'post_value': post_value,
                        'step_magnitude': step_magnitude,
                        'step_direction': 'up' if step_magnitude > 0 else 'down'
                    })
            
            step_events[input_name] = events
            logger.info(f"  {input_name}: {len(events)} step changes detected")
        
        self._impulse_events = step_events
        return step_events
    
    @staticmethod
    def _analyze_single_response_static(
        input_name: str,
        output_name: str,
        step_event: Dict,
        output_data: np.ndarray,
        time_data: np.ndarray,
        pre_window: int,
        post_window: int
    ) -> Dict:
        """
        Analyze output response to a single step event.
        Static method for parallel execution.
        """
        try:
            step_idx = step_event['step_index']
            step_magnitude = step_event['step_magnitude']
            
            # Extract windows
            pre_start = max(0, step_idx - pre_window)
            post_end = min(len(output_data), step_idx + post_window)
            
            if post_end - step_idx < 5:  # Not enough post-step data
                return {}
            
            pre_output = output_data[pre_start:step_idx]
            post_output = output_data[step_idx:post_end]
            
            if len(pre_output) < 3 or len(post_output) < 5:
                return {}
            
            # Compute response characteristics
            initial_value = np.nanmean(pre_output[-min(5, len(pre_output)):])
            final_value = np.nanmean(post_output[-min(10, len(post_output)):])
            output_change = final_value - initial_value
            
            # Gain: Δoutput / Δinput
            gain = output_change / step_magnitude if abs(step_magnitude) > 1e-10 else np.nan
            
            # Normalize response for timing analysis
            if abs(output_change) > 1e-10:
                normalized_response = (post_output - initial_value) / output_change
            else:
                normalized_response = post_output - initial_value
            
            # Rise time: time to reach 90% of final value
            if abs(output_change) > 1e-10:
                target_90 = 0.9
                target_10 = 0.1
                
                indices_above_90 = np.where(normalized_response >= target_90)[0]
                indices_above_10 = np.where(normalized_response >= target_10)[0]
                
                if len(indices_above_90) > 0:
                    rise_time_90 = indices_above_90[0]
                else:
                    rise_time_90 = len(normalized_response)
                
                if len(indices_above_10) > 0:
                    rise_time_10 = indices_above_10[0]
                else:
                    rise_time_10 = 0
                
                rise_time = rise_time_90 - rise_time_10
            else:
                rise_time = np.nan
                rise_time_90 = np.nan
            
            # Overshoot
            peak_value = np.nanmax(post_output) if output_change > 0 else np.nanmin(post_output)
            if abs(output_change) > 1e-10:
                overshoot = (peak_value - final_value) / abs(output_change) * 100
            else:
                overshoot = 0
            
            # Settling time: time to stay within 5% of final value
            tolerance = 0.05
            if abs(output_change) > 1e-10:
                within_tolerance = np.abs(normalized_response - 1.0) <= tolerance
                
                # Find last index outside tolerance
                outside_tolerance = np.where(~within_tolerance)[0]
                if len(outside_tolerance) > 0:
                    settling_time = outside_tolerance[-1] + 1
                else:
                    settling_time = 0
            else:
                settling_time = 0
            
            # Try to fit first-order response: y(t) = y_final * (1 - exp(-t/tau))
            time_constant = np.nan
            fit_r2 = np.nan
            
            if abs(output_change) > 1e-10 and len(post_output) >= 5:
                try:
                    t = np.arange(len(post_output))
                    y = normalized_response
                    
                    # Simple fit using linear regression on log transform
                    valid_mask = (y > 0.01) & (y < 0.99) & np.isfinite(y)
                    if np.sum(valid_mask) >= 3:
                        y_valid = y[valid_mask]
                        t_valid = t[valid_mask]
                        
                        # For first-order: ln(1-y) = -t/tau
                        log_term = np.log(1 - y_valid + 1e-10)
                        
                        lr = LinearRegression()
                        lr.fit(t_valid.reshape(-1, 1), log_term)
                        
                        if lr.coef_[0] < 0:
                            time_constant = -1 / lr.coef_[0]
                            
                            # Compute R² of first-order fit
                            y_pred = 1 - np.exp(-t / time_constant)
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - np.mean(y)) ** 2)
                            fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                except Exception:
                    pass
            
            # Determine response order (first-order vs higher)
            if not np.isnan(fit_r2):
                if fit_r2 > 0.9:
                    response_order = 'first_order'
                elif fit_r2 > 0.7:
                    response_order = 'approx_first_order'
                else:
                    response_order = 'higher_order'
            else:
                response_order = 'unknown'
            
            return {
                'input': input_name,
                'output': output_name,
                'step_index': step_idx,
                'step_magnitude': step_magnitude,
                'step_direction': step_event['step_direction'],
                'initial_value': initial_value,
                'final_value': final_value,
                'output_change': output_change,
                'gain': gain,
                'rise_time': rise_time,
                'rise_time_90': rise_time_90,
                'settling_time': settling_time,
                'overshoot_pct': overshoot,
                'time_constant': time_constant,
                'fit_r2': fit_r2,
                'response_order': response_order
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            return {}
    
    def analyze_all_responses(
        self,
        prepared_data: Dict[str, np.ndarray],
        step_events: Dict[str, List[Dict]],
        pre_window: int = 30,
        post_window: int = 60
    ) -> pd.DataFrame:
        """
        Analyze output responses to all detected step events.
        
        Args:
            prepared_data: Prepared data dictionary
            step_events: Detected step events
            pre_window: Time steps before step to analyze
            post_window: Time steps after step to analyze
            
        Returns:
            DataFrame with response characteristics
        """
        logger.info("Analyzing impulse responses for all input-output pairs...")
        
        self._init_dask_client()
        
        delayed_tasks = []
        response_curves = []
        
        for input_name, events in step_events.items():
            if not events:
                continue
            
            for output_name, output_data in prepared_data['outputs'].items():
                for event in events:
                    task = delayed(self._analyze_single_response_static)(
                        input_name, output_name, event, output_data,
                        prepared_data['time'], pre_window, post_window
                    )
                    delayed_tasks.append(task)
                    
                    # Store curve data for visualization
                    step_idx = event['step_index']
                    pre_start = max(0, step_idx - pre_window)
                    post_end = min(len(output_data), step_idx + post_window)
                    
                    response_curves.append({
                        'input': input_name,
                        'output': output_name,
                        'step_index': step_idx,
                        'time': prepared_data['time'][pre_start:post_end] if prepared_data['time'] is not None 
                               else np.arange(pre_start, post_end),
                        'output_values': output_data[pre_start:post_end],
                        'step_position': step_idx - pre_start
                    })
        
        logger.info(f"Executing {len(delayed_tasks)} response analysis tasks...")
        results_list = dask.compute(*delayed_tasks)
        
        results = [r for r in results_list if r]
        results_df = pd.DataFrame(results)
        
        self._response_curves = response_curves
        
        logger.info(f"Analyzed {len(results_df)} response events")
        return results_df
    
    def get_response_curves(self) -> List[Dict]:
        """Return response curve data for visualization."""
        return self._response_curves if self._response_curves else []
    
    def get_impulse_events(self) -> Dict[str, List[Dict]]:
        """Return detected impulse events."""
        return self._impulse_events if self._impulse_events else {}
    
    def summarize_response_characteristics(
        self,
        response_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create summary statistics of response characteristics per input-output pair.
        
        Args:
            response_df: Individual response analysis results
            
        Returns:
            Summary DataFrame with aggregated statistics
        """
        logger.info("Creating response characteristics summary...")
        
        if response_df.empty:
            return pd.DataFrame()
        
        summary = response_df.groupby(['input', 'output']).agg({
            'gain': ['mean', 'std', 'min', 'max'],
            'rise_time': ['mean', 'std', 'median'],
            'settling_time': ['mean', 'std', 'median'],
            'overshoot_pct': ['mean', 'std', 'max'],
            'time_constant': ['mean', 'std', 'median'],
            'fit_r2': ['mean', 'min'],
            'step_index': 'count'
        })
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        summary = summary.rename(columns={'step_index_count': 'n_events'})
        
        # Add interpretation
        summary['avg_response_time'] = summary['rise_time_mean']
        summary['response_consistency'] = 1 - (summary['rise_time_std'] / (summary['rise_time_mean'] + 1e-10))
        summary['is_stable'] = summary['overshoot_pct_max'] < 20
        
        # Categorize response speed
        summary['speed_category'] = pd.cut(
            summary['rise_time_mean'],
            bins=[0, 5, 15, 30, float('inf')],
            labels=['very_fast', 'fast', 'medium', 'slow']
        )
        
        return summary
    
    def identify_critical_dynamics(
        self,
        summary_df: pd.DataFrame,
        gain_threshold: float = 0.1,
        response_time_threshold: int = 20
    ) -> pd.DataFrame:
        """
        Identify input-output pairs with critical dynamic characteristics.
        
        Args:
            summary_df: Summary of response characteristics
            gain_threshold: Minimum gain for significant effect
            response_time_threshold: Maximum response time for fast dynamics
            
        Returns:
            DataFrame with critical dynamics identified
        """
        logger.info("Identifying critical dynamics...")
        
        critical = summary_df.copy()
        
        # Flag significant gains
        critical['significant_gain'] = np.abs(critical['gain_mean']) > gain_threshold
        
        # Flag fast responses
        critical['fast_response'] = critical['rise_time_mean'] < response_time_threshold
        
        # Flag stable responses
        critical['stable'] = critical['is_stable']
        
        # Prioritize: significant gain + fast response + stable
        critical['priority_score'] = (
            critical['significant_gain'].astype(int) * 3 +
            critical['fast_response'].astype(int) * 2 +
            critical['stable'].astype(int) * 1
        )
        
        critical = critical.sort_values('priority_score', ascending=False)
        
        return critical
