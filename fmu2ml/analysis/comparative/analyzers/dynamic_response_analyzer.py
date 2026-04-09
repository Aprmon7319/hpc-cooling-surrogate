"""
Dynamic Response Analyzer for CDU-Level Comparative Analysis.

Analyzes time-domain responses including step response, impulse response,
time constants, settling times, and dynamic behavior characteristics.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class DynamicConfig:
    """Configuration for dynamic response analysis."""
    
    # Time series parameters
    dt: float = 1.0  # Time step in seconds
    resample: bool = True  # Resample to uniform time
    
    # Step response analysis
    settling_threshold: float = 0.02  # 2% settling criterion
    rise_time_thresholds: Tuple[float, float] = (0.1, 0.9)  # 10-90% rise time
    
    # Time constant estimation
    fit_method: str = "exponential"  # exponential, first_order, second_order
    max_time_constants: int = 3  # For multi-exponential fits
    
    # Impulse response
    impulse_duration: int = 5  # Samples
    
    # Frequency analysis
    compute_frequency_response: bool = True
    freq_range: Tuple[float, float] = (0.001, 1.0)  # Hz
    n_freq_points: int = 50
    
    # Delay estimation
    estimate_delay: bool = True
    max_delay_samples: int = 100


@dataclass
class DynamicResult:
    """Results from dynamic response analysis."""
    
    model_name: str
    cdu_id: int
    input_var: str
    output_var: str
    
    # Step response characteristics
    step_response: Optional[np.ndarray] = None
    steady_state_gain: float = 0.0
    rise_time: float = 0.0  # seconds
    settling_time: float = 0.0  # seconds
    overshoot: float = 0.0  # percentage
    undershoot: float = 0.0  # percentage
    
    # Time constants
    time_constants: List[float] = field(default_factory=list)
    dominant_time_constant: float = 0.0
    
    # Impulse response
    impulse_response: Optional[np.ndarray] = None
    
    # Delay
    delay: float = 0.0  # seconds
    
    # Frequency response
    frequencies: Optional[np.ndarray] = None
    magnitude_response: Optional[np.ndarray] = None
    phase_response: Optional[np.ndarray] = None
    bandwidth: float = 0.0  # Hz
    
    # Model fit
    fitted_parameters: Dict[str, float] = field(default_factory=dict)
    fit_quality: float = 0.0  # R-squared


class DynamicResponseAnalyzer:
    """
    Analyzes dynamic response characteristics of CDU I/O relationships.
    
    Key capabilities:
    1. Step response analysis (rise time, settling time, overshoot)
    2. Time constant estimation via exponential fitting
    3. Impulse response computation
    4. Delay estimation
    5. Frequency response analysis
    6. Cross-model dynamic comparison
    """
    
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    OUTPUT_VARS = [
        "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW",
        "T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C",
        "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig", "p_sec_r_psig"
    ]
    
    def __init__(self, config: Optional[DynamicConfig] = None):
        """Initialize dynamic response analyzer."""
        self.config = config or DynamicConfig()
        self.results: Dict[str, Dict[int, Dict[str, DynamicResult]]] = {}
        
    def analyze_step_response(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
        model_name: str = "model",
        cdu_id: int = 0,
        input_var: str = "input",
        output_var: str = "output"
    ) -> DynamicResult:
        """
        Analyze step response characteristics.
        
        Args:
            time: Time array
            input_signal: Input signal (should contain step change)
            output_signal: Output signal response
            model_name: Model identifier
            cdu_id: CDU identifier
            input_var: Input variable name
            output_var: Output variable name
            
        Returns:
            DynamicResult with step response characteristics
        """
        result = DynamicResult(
            model_name=model_name,
            cdu_id=cdu_id,
            input_var=input_var,
            output_var=output_var
        )
        
        # Resample to uniform time if needed
        if self.config.resample:
            time, input_signal, output_signal = self._resample_uniform(
                time, input_signal, output_signal
            )
            
        # Detect step in input
        step_idx = self._detect_step(input_signal)
        if step_idx is None:
            logger.warning(f"No step detected in input signal for {output_var}")
            return result
            
        # Extract response after step
        t_response = time[step_idx:] - time[step_idx]
        y_response = output_signal[step_idx:]
        
        if len(y_response) < 10:
            logger.warning(f"Insufficient response data for {output_var}")
            return result
            
        # Initial and final values
        y_initial = output_signal[max(0, step_idx-5):step_idx].mean() if step_idx > 0 else y_response[0]
        y_final = y_response[-10:].mean()  # Average of last 10 samples
        
        # Step magnitude
        u_step = input_signal[step_idx:step_idx+10].mean() - input_signal[max(0, step_idx-5):step_idx].mean()
        
        # Steady-state gain
        if abs(u_step) > 1e-10:
            result.steady_state_gain = (y_final - y_initial) / u_step
        
        # Normalize response for characteristic analysis
        if abs(y_final - y_initial) > 1e-10:
            y_norm = (y_response - y_initial) / (y_final - y_initial)
        else:
            y_norm = np.zeros_like(y_response)
            
        result.step_response = y_norm
        
        # Rise time (10% to 90%)
        low_thresh, high_thresh = self.config.rise_time_thresholds
        try:
            t_low = t_response[np.where(y_norm >= low_thresh)[0][0]]
            t_high = t_response[np.where(y_norm >= high_thresh)[0][0]]
            result.rise_time = t_high - t_low
        except (IndexError, ValueError):
            result.rise_time = np.nan
            
        # Settling time (2% criterion)
        settling_band = self.config.settling_threshold
        try:
            # Find last time outside settling band
            outside_band = np.where(np.abs(y_norm - 1.0) > settling_band)[0]
            if len(outside_band) > 0:
                result.settling_time = t_response[outside_band[-1]]
            else:
                result.settling_time = 0.0
        except (IndexError, ValueError):
            result.settling_time = np.nan
            
        # Overshoot and undershoot
        if len(y_norm) > 0:
            max_val = np.max(y_norm)
            min_val = np.min(y_norm)
            
            result.overshoot = max(0, (max_val - 1.0) * 100)  # percentage
            result.undershoot = max(0, -min_val * 100) if min_val < 0 else 0.0
            
        # Estimate time constants
        self._estimate_time_constants(t_response, y_norm, result)
        
        return result
    
    def _estimate_time_constants(
        self,
        time: np.ndarray,
        response: np.ndarray,
        result: DynamicResult
    ) -> None:
        """Estimate time constants from step response."""
        if len(time) < 10 or len(response) < 10:
            return
            
        try:
            if self.config.fit_method == "exponential":
                # Single exponential: y = 1 - exp(-t/tau)
                def exp_model(t, tau):
                    return 1.0 - np.exp(-t / tau)
                    
                popt, _ = curve_fit(
                    exp_model, time, response,
                    p0=[time[-1] / 4],
                    bounds=(1e-6, time[-1] * 10),
                    maxfev=1000
                )
                result.time_constants = [float(popt[0])]
                result.dominant_time_constant = float(popt[0])
                
                # Compute fit quality
                y_fit = exp_model(time, *popt)
                ss_res = np.sum((response - y_fit) ** 2)
                ss_tot = np.sum((response - np.mean(response)) ** 2)
                result.fit_quality = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                result.fitted_parameters = {"tau": float(popt[0])}
                
            elif self.config.fit_method == "first_order":
                # First-order with delay: y = K * (1 - exp(-(t-L)/tau))
                def first_order(t, K, tau, L):
                    t_delayed = np.maximum(t - L, 0)
                    return K * (1.0 - np.exp(-t_delayed / tau))
                    
                popt, _ = curve_fit(
                    first_order, time, response,
                    p0=[1.0, time[-1] / 4, 0],
                    bounds=([0.5, 1e-6, 0], [1.5, time[-1] * 10, time[-1] / 2]),
                    maxfev=2000
                )
                result.time_constants = [float(popt[1])]
                result.dominant_time_constant = float(popt[1])
                result.delay = float(popt[2])
                result.fitted_parameters = {
                    "K": float(popt[0]),
                    "tau": float(popt[1]),
                    "L": float(popt[2])
                }
                
            elif self.config.fit_method == "second_order":
                # Second-order: underdamped/overdamped based on response
                if result.overshoot > 0:
                    # Underdamped second-order
                    def second_order_under(t, wn, zeta):
                        wd = wn * np.sqrt(1 - zeta**2)
                        return 1.0 - np.exp(-zeta * wn * t) * (
                            np.cos(wd * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * t)
                        )
                        
                    popt, _ = curve_fit(
                        second_order_under, time, response,
                        p0=[1.0 / (time[-1] / 4), 0.5],
                        bounds=([1e-6, 0.01], [10 / time[1], 0.99]),
                        maxfev=2000
                    )
                    wn, zeta = popt
                    tau1 = 1 / (zeta * wn)
                    result.time_constants = [tau1]
                    result.dominant_time_constant = tau1
                    result.fitted_parameters = {"wn": float(wn), "zeta": float(zeta)}
                else:
                    # Overdamped - use two exponentials
                    def two_exp(t, a1, tau1, a2, tau2):
                        return 1.0 - a1 * np.exp(-t / tau1) - a2 * np.exp(-t / tau2)
                        
                    try:
                        popt, _ = curve_fit(
                            two_exp, time, response,
                            p0=[0.5, time[-1] / 4, 0.5, time[-1] / 8],
                            bounds=([0, 1e-6, 0, 1e-6], [2, time[-1] * 10, 2, time[-1] * 10]),
                            maxfev=3000
                        )
                        result.time_constants = sorted([float(popt[1]), float(popt[3])], reverse=True)
                        result.dominant_time_constant = result.time_constants[0]
                        result.fitted_parameters = {
                            "a1": float(popt[0]), "tau1": float(popt[1]),
                            "a2": float(popt[2]), "tau2": float(popt[3])
                        }
                    except Exception:
                        # Fall back to single exponential
                        def exp_model(t, tau):
                            return 1.0 - np.exp(-t / tau)
                        popt, _ = curve_fit(exp_model, time, response, p0=[time[-1] / 4])
                        result.time_constants = [float(popt[0])]
                        result.dominant_time_constant = float(popt[0])
                        
        except Exception as e:
            logger.warning(f"Time constant estimation failed: {e}")
            # Estimate from 63.2% point (time constant definition)
            try:
                idx_63 = np.where(response >= 0.632)[0][0]
                result.time_constants = [float(time[idx_63])]
                result.dominant_time_constant = float(time[idx_63])
            except (IndexError, ValueError):
                pass
                
    def compute_impulse_response(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute impulse response via deconvolution.
        
        Args:
            time: Time array
            input_signal: Input signal
            output_signal: Output signal
            
        Returns:
            Tuple of (time, impulse_response)
        """
        # Resample to uniform time
        if self.config.resample:
            time, input_signal, output_signal = self._resample_uniform(
                time, input_signal, output_signal
            )
            
        # Use FFT-based deconvolution
        n = len(output_signal)
        
        # Compute FFTs
        U = np.fft.fft(input_signal)
        Y = np.fft.fft(output_signal)
        
        # Regularized deconvolution
        epsilon = 1e-6 * np.max(np.abs(U))
        H = Y / (U + epsilon)
        
        # Inverse FFT
        h = np.real(np.fft.ifft(H))
        
        # Return only causal part
        return time[:n//2], h[:n//2]
    
    def estimate_delay(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
        dt: float
    ) -> float:
        """
        Estimate input-output delay via cross-correlation.
        
        Args:
            input_signal: Input signal
            output_signal: Output signal
            dt: Time step
            
        Returns:
            Estimated delay in seconds
        """
        # Normalize signals
        u = (input_signal - np.mean(input_signal)) / (np.std(input_signal) + 1e-10)
        y = (output_signal - np.mean(output_signal)) / (np.std(output_signal) + 1e-10)
        
        # Cross-correlation
        correlation = np.correlate(y, u, mode='full')
        lags = np.arange(-len(u) + 1, len(u))
        
        # Find peak in positive lag region (output lags input)
        positive_lags = lags >= 0
        positive_corr = correlation[positive_lags]
        positive_lag_values = lags[positive_lags]
        
        max_idx = np.argmax(positive_corr[:self.config.max_delay_samples])
        delay_samples = positive_lag_values[max_idx]
        
        return delay_samples * dt
    
    def compute_frequency_response(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute frequency response (Bode plot data).
        
        Args:
            time: Time array
            input_signal: Input signal
            output_signal: Output signal
            
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        # Resample to uniform time
        if self.config.resample:
            time, input_signal, output_signal = self._resample_uniform(
                time, input_signal, output_signal
            )
            
        dt = time[1] - time[0] if len(time) > 1 else self.config.dt
        
        # FFT
        n = len(output_signal)
        U = np.fft.fft(input_signal)
        Y = np.fft.fft(output_signal)
        freqs = np.fft.fftfreq(n, dt)
        
        # Transfer function estimate
        epsilon = 1e-6 * np.max(np.abs(U))
        H = Y / (U + epsilon)
        
        # Only positive frequencies
        pos_freq_idx = freqs > 0
        freqs = freqs[pos_freq_idx]
        H = H[pos_freq_idx]
        
        # Filter to frequency range of interest
        freq_mask = (freqs >= self.config.freq_range[0]) & (freqs <= self.config.freq_range[1])
        freqs = freqs[freq_mask]
        H = H[freq_mask]
        
        # Magnitude and phase
        magnitude = 20 * np.log10(np.abs(H) + 1e-10)
        phase = np.angle(H, deg=True)
        
        return freqs, magnitude, phase
    
    def analyze_from_data(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model",
        time_col: str = "time"
    ) -> Dict[str, Dict[str, DynamicResult]]:
        """
        Analyze dynamic response for all I/O pairs from DataFrame.
        
        Args:
            data: DataFrame with time series data
            cdu_id: CDU identifier
            model_name: Model identifier
            time_col: Time column name
            
        Returns:
            Dict mapping input->output to DynamicResult
        """
        results = {}
        
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        # Get time array
        if time_col in data.columns:
            time = data[time_col].values
        else:
            time = np.arange(len(data)) * self.config.dt
            
        for input_var, input_col in input_cols.items():
            if input_col not in data.columns:
                continue
                
            input_signal = data[input_col].values
            results[input_var] = {}
            
            for output_var, output_col in output_cols.items():
                if output_col not in data.columns:
                    continue
                    
                output_signal = data[output_col].values
                
                # Analyze step response
                result = self.analyze_step_response(
                    time, input_signal, output_signal,
                    model_name, cdu_id, input_var, output_var
                )
                
                # Estimate delay
                if self.config.estimate_delay:
                    result.delay = self.estimate_delay(
                        input_signal, output_signal, self.config.dt
                    )
                    
                # Compute frequency response
                if self.config.compute_frequency_response:
                    try:
                        freqs, mag, phase = self.compute_frequency_response(
                            time, input_signal, output_signal
                        )
                        result.frequencies = freqs
                        result.magnitude_response = mag
                        result.phase_response = phase
                        
                        # Estimate bandwidth (-3dB point)
                        if len(mag) > 0:
                            dc_gain = mag[0]
                            bw_idx = np.where(mag < dc_gain - 3)[0]
                            if len(bw_idx) > 0:
                                result.bandwidth = freqs[bw_idx[0]]
                    except Exception as e:
                        logger.warning(f"Frequency response failed for {output_var}: {e}")
                        
                results[input_var][output_var] = result
                
        # Cache results
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][cdu_id] = results
        
        return results
    
    def analyze_step_test_data(
        self,
        step_data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model",
        step_input: str = "Q_flow",
        time_col: str = "time"
    ) -> Dict[str, DynamicResult]:
        """
        Analyze step response from step test data.
        
        Args:
            step_data: DataFrame from step test
            cdu_id: CDU identifier
            model_name: Model identifier
            step_input: Input that was stepped
            time_col: Time column name
            
        Returns:
            Dict mapping output to DynamicResult
        """
        results = {}
        
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        if step_input not in input_cols:
            logger.error(f"Unknown input: {step_input}")
            return results
            
        input_col = input_cols[step_input]
        if input_col not in step_data.columns:
            logger.error(f"Input column {input_col} not in data")
            return results
            
        # Get time and input
        time = step_data[time_col].values if time_col in step_data.columns else np.arange(len(step_data)) * self.config.dt
        input_signal = step_data[input_col].values
        
        for output_var, output_col in output_cols.items():
            if output_col not in step_data.columns:
                continue
                
            output_signal = step_data[output_col].values
            
            result = self.analyze_step_response(
                time, input_signal, output_signal,
                model_name, cdu_id, step_input, output_var
            )
            
            results[output_var] = result
            
        return results
    
    def compare_dynamics_across_models(
        self,
        model_results: Dict[str, Dict[str, DynamicResult]]
    ) -> Dict[str, Any]:
        """
        Compare dynamic characteristics across models.
        
        Args:
            model_results: Dict mapping model name to output->DynamicResult
            
        Returns:
            Comparison metrics
        """
        comparison = {
            "models": list(model_results.keys()),
            "time_constant_comparison": {},
            "settling_time_comparison": {},
            "gain_comparison": {},
            "delay_comparison": {}
        }
        
        models = list(model_results.keys())
        outputs = set()
        for results in model_results.values():
            outputs.update(results.keys())
            
        for output in outputs:
            # Collect metrics for this output
            time_constants = {}
            settling_times = {}
            gains = {}
            delays = {}
            
            for model in models:
                if output in model_results.get(model, {}):
                    result = model_results[model][output]
                    time_constants[model] = result.dominant_time_constant
                    settling_times[model] = result.settling_time
                    gains[model] = result.steady_state_gain
                    delays[model] = result.delay
                    
            if len(time_constants) >= 2:
                # Compute relative differences
                tc_vals = list(time_constants.values())
                st_vals = list(settling_times.values())
                gain_vals = list(gains.values())
                delay_vals = list(delays.values())
                
                comparison["time_constant_comparison"][output] = {
                    "values": time_constants,
                    "range": max(tc_vals) - min(tc_vals) if tc_vals else 0,
                    "cv": np.std(tc_vals) / np.mean(tc_vals) if np.mean(tc_vals) > 0 else 0
                }
                
                comparison["settling_time_comparison"][output] = {
                    "values": settling_times,
                    "range": max(st_vals) - min(st_vals) if st_vals else 0
                }
                
                comparison["gain_comparison"][output] = {
                    "values": gains,
                    "range": max(gain_vals) - min(gain_vals) if gain_vals else 0
                }
                
                comparison["delay_comparison"][output] = {
                    "values": delays,
                    "range": max(delay_vals) - min(delay_vals) if delay_vals else 0
                }
                
        return comparison
    
    def _resample_uniform(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample signals to uniform time steps."""
        # Check if already uniform
        dt = np.diff(time)
        if len(dt) > 0 and np.std(dt) / np.mean(dt) < 0.01:
            return time, input_signal, output_signal
            
        # Create uniform time vector
        t_uniform = np.arange(time[0], time[-1], self.config.dt)
        
        # Interpolate signals
        f_input = interp1d(time, input_signal, kind='linear', fill_value='extrapolate')
        f_output = interp1d(time, output_signal, kind='linear', fill_value='extrapolate')
        
        return t_uniform, f_input(t_uniform), f_output(t_uniform)
    
    def _detect_step(self, signal: np.ndarray, threshold: float = 0.1) -> Optional[int]:
        """Detect step change in signal."""
        # Compute derivative
        d_signal = np.diff(signal)
        
        # Normalize
        d_max = np.max(np.abs(d_signal))
        if d_max < 1e-10:
            return None
            
        d_norm = np.abs(d_signal) / d_max
        
        # Find step (large derivative)
        step_candidates = np.where(d_norm > threshold)[0]
        
        if len(step_candidates) == 0:
            return None
            
        return step_candidates[0]
    
    def _get_input_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get input column names for a CDU."""
        return {
            "Q_flow": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total",
            "T_Air": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_T_Air",
            "T_ext": "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext"
        }
        
    def _get_output_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get output column names for a CDU."""
        base = f"simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary"
        return {var: f"{base}.{var}" for var in self.OUTPUT_VARS}
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame."""
        rows = []
        
        for model_name, cdu_results in self.results.items():
            for cdu_id, io_results in cdu_results.items():
                for input_var, output_results in io_results.items():
                    for output_var, result in output_results.items():
                        rows.append({
                            "model": model_name,
                            "cdu_id": cdu_id,
                            "input": input_var,
                            "output": output_var,
                            "steady_state_gain": result.steady_state_gain,
                            "rise_time": result.rise_time,
                            "settling_time": result.settling_time,
                            "overshoot": result.overshoot,
                            "dominant_time_constant": result.dominant_time_constant,
                            "delay": result.delay,
                            "bandwidth": result.bandwidth,
                            "fit_quality": result.fit_quality
                        })
                        
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary DataFrame
        df = self.to_dataframe()
        df.to_csv(output_dir / "dynamic_response_summary.csv", index=False)
        
        # Save detailed results as JSON
        import json
        
        for model_name, cdu_results in self.results.items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            for cdu_id, io_results in cdu_results.items():
                cdu_data = {}
                for input_var, output_results in io_results.items():
                    cdu_data[input_var] = {}
                    for output_var, result in output_results.items():
                        cdu_data[input_var][output_var] = {
                            "steady_state_gain": result.steady_state_gain,
                            "rise_time": result.rise_time,
                            "settling_time": result.settling_time,
                            "overshoot": result.overshoot,
                            "undershoot": result.undershoot,
                            "time_constants": result.time_constants,
                            "dominant_time_constant": result.dominant_time_constant,
                            "delay": result.delay,
                            "bandwidth": result.bandwidth,
                            "fit_quality": result.fit_quality,
                            "fitted_parameters": result.fitted_parameters
                        }
                        
                with open(model_dir / f"cdu_{cdu_id}_dynamics.json", "w") as f:
                    json.dump(cdu_data, f, indent=2)
