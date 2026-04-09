import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, List


class StabilityDetector:
    """Detects when FMU simulation has reached steady state"""
    
    __slots__ = ['window_size', 'threshold', 'min_steps', 'max_steps', 
                 'metric_buffers', 'step_count', 'stable_count', 
                 'required_stable_steps', '_metric_keys_cache']
    
    def __init__(
        self,
        window_size: int = 300,
        threshold: float = 0.1,
        min_steps: int = 3600,
        max_steps: int = 14400
    ):
        """Initialize stability detector"""
        self.window_size = window_size
        self.threshold = threshold
        self.min_steps = min_steps
        self.max_steps = max_steps
        
        # Buffers for key metrics - using numpy arrays for efficiency
        self.metric_buffers = {
            'temperatures': deque(maxlen=window_size),
            'pressures': deque(maxlen=window_size),
            'flow_rates': deque(maxlen=window_size),
            'power': deque(maxlen=window_size)
        }
        
        self.step_count = 0
        self.stable_count = 0
        self.required_stable_steps = 100
        
        # Cache for metric keys to avoid repeated string operations
        self._metric_keys_cache = {
            'temp': [],
            'pressure': [],
            'flow': [],
            'power': [],
            'dc_flow': None,
            'initialized': False
        }

    def _initialize_key_cache(self, fmu_output: dict):
        """Initialize key cache for faster lookup"""
        if self._metric_keys_cache['initialized']:
            return
        
        keys = fmu_output.keys()
        
        # Pre-compute key lists
        self._metric_keys_cache['temp'] = [
            k for k in keys if 'T_prim_s_C' in k or 'T_sec_s_C' in k
        ]
        self._metric_keys_cache['pressure'] = [
            k for k in keys if 'psig' in k
        ]
        self._metric_keys_cache['flow'] = [
            k for k in keys if 'V_flow_prim_GPM' in k
        ]
        self._metric_keys_cache['power'] = [
            k for k in keys if 'W_flow_CDUP_kW' in k
        ]
        
        # Find datacenter flow key
        dc_flow_keys = [
            k for k in keys if 'datacenter[1].summary.V_flow_prim_GPM' in k
        ]
        if dc_flow_keys:
            self._metric_keys_cache['dc_flow'] = dc_flow_keys[0]
        
        self._metric_keys_cache['initialized'] = True

    def extract_key_metrics(self, fmu_output: dict) -> Dict[str, float]:
        """Extract key metrics from FMU output for stability checking"""
        # Initialize cache on first call
        self._initialize_key_cache(fmu_output)
        
        metrics = {}
        cache = self._metric_keys_cache
        
        # Temperature metrics - use numpy for efficiency
        if cache['temp']:
            temp_values = np.array([fmu_output[k] for k in cache['temp']])
            metrics['temperature_mean'] = np.mean(temp_values)
            metrics['temperature_std'] = np.std(temp_values)
        
        # Pressure metrics
        if cache['pressure']:
            pressure_values = np.array([fmu_output[k] for k in cache['pressure']])
            metrics['pressure_mean'] = np.mean(pressure_values)
            metrics['pressure_std'] = np.std(pressure_values)
        
        # Flow rate metrics
        if cache['flow']:
            flow_values = np.array([fmu_output[k] for k in cache['flow']])
            metrics['flow_mean'] = np.mean(flow_values)
            metrics['flow_std'] = np.std(flow_values)
        
        # Power metrics
        if cache['power']:
            power_values = np.array([fmu_output[k] for k in cache['power']])
            metrics['power_mean'] = np.mean(power_values)
        
        # Datacenter total flow
        if cache['dc_flow']:
            metrics['dc_total_flow'] = fmu_output[cache['dc_flow']]
        
        # PUE
        metrics['pue'] = fmu_output.get('pue', 1.0)
        
        return metrics

    def add_step(self, fmu_output: dict) -> None:
        """Add a simulation step to the stability detector"""
        self.step_count += 1
        
        metrics = self.extract_key_metrics(fmu_output)
        
        # Store aggregated metrics as tuples for memory efficiency
        self.metric_buffers['temperatures'].append(
            (metrics.get('temperature_mean', 0), metrics.get('temperature_std', 0))
        )
        self.metric_buffers['pressures'].append(
            (metrics.get('pressure_mean', 0), metrics.get('pressure_std', 0))
        )
        self.metric_buffers['flow_rates'].append(
            (metrics.get('flow_mean', 0), metrics.get('dc_total_flow', 0))
        )
        self.metric_buffers['power'].append(
            (metrics.get('power_mean', 0), metrics.get('pue', 1.0))
        )

    @staticmethod
    def _calculate_stability(values: np.ndarray, threshold: float) -> Tuple[bool, float]:
        """Stability calculation"""
        if len(values) == 0:
            return False, np.inf
        
        # Handle zero or near-zero values
        mean_val = np.mean(values)
        if abs(mean_val) < 1e-10:
            return True, 0.0
        
        # Calculate relative standard deviation
        std_val = np.std(values)
        rel_std = std_val / abs(mean_val)
        
        # Calculate max relative change between consecutive windows
        window_half = len(values) // 2
        first_half_mean = np.mean(values[:window_half])
        second_half_mean = np.mean(values[window_half:])
        
        if abs(first_half_mean) > 1e-10:
            rel_change = abs(second_half_mean - first_half_mean) / abs(first_half_mean)
        else:
            rel_change = 0.0
        
        max_change = max(rel_std, rel_change)
        is_stable = max_change < threshold
        
        return is_stable, max_change

    def calculate_metric_stability(
        self,
        buffer: deque
    ) -> Tuple[bool, float]:
        """Calculate if a metric buffer is stable"""
        if len(buffer) < self.window_size:
            return False, float('inf')
        
        # Convert to numpy array efficiently
        values = np.array([item[0] for item in buffer], dtype=np.float64)
        
        return self._calculate_stability(values, self.threshold)

    def is_stable(self) -> Tuple[bool, Dict]:
        """Check if simulation has reached steady state"""
        # Must meet minimum steps
        if self.step_count < self.min_steps:
            return False, {'reason': 'minimum_steps_not_met'}
        
        # Force completion at max steps
        if self.step_count >= self.max_steps:
            return True, {'reason': 'max_steps_reached', 'forced': True}
        
        # Check if buffers are full
        if any(len(buf) < self.window_size for buf in self.metric_buffers.values()):
            return False, {'reason': 'buffers_not_full'}
        
        # Check stability of each metric
        stability_results = {}
        all_stable = True
        
        for metric_name, buffer in self.metric_buffers.items():
            is_stable, max_change = self.calculate_metric_stability(buffer)
            stability_results[metric_name] = {
                'stable': is_stable,
                'max_change': max_change
            }
            all_stable = all_stable and is_stable
        
        # Track consecutive stable steps
        if all_stable:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        # Need consecutive stable steps
        is_fully_stable = self.stable_count >= self.required_stable_steps
        
        return is_fully_stable, {
            'stable_count': self.stable_count,
            'required': self.required_stable_steps,
            'metrics': stability_results
        }

    def get_summary(self) -> Dict:
        """Get summary of stability detection"""
        return {
            'total_steps': self.step_count,
            'final_stable_count': self.stable_count,
            'required_stable_steps': self.required_stable_steps,
            'window_size': self.window_size,
            'threshold': self.threshold,
            'is_stable': self.stable_count >= self.required_stable_steps
        }