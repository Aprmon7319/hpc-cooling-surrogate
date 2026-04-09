"""
Dynamic Data Generator

Generates simulation data specifically designed for dynamic analysis:
1. Step changes for impulse response
2. Ramp inputs for rate analysis
3. Varying rate inputs to test derivative effects
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional, Callable
import logging
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


def _run_single_scenario(
    scenario_id: int,
    input_data: pd.DataFrame,
    system_name: str,
    config_overrides: Dict,
    stabilization_hours: int,
    step_size: int
) -> Tuple[int, pd.DataFrame]:
    """
    Run a single complete scenario simulation.
    Each scenario runs sequentially to maintain state causality.
    """
    from fmu2ml.simulation.fmu_simulator import FMUSimulator
    
    logger.info(f"Scenario {scenario_id}: Starting simulation with {len(input_data)} steps...")
    
    try:
        simulator = FMUSimulator(
            system_name=system_name,
            **config_overrides
        )
        
        results = simulator.run_simulation(
            input_data=input_data,
            stabilization_hours=stabilization_hours,
            step_size=step_size,
            save_history=False
        )
        
        # Add scenario ID for tracking
        results['scenario_id'] = scenario_id
        
        logger.info(f"Scenario {scenario_id}: Completed with {len(results)} samples")
        return scenario_id, results
        
    except Exception as e:
        logger.error(f"Scenario {scenario_id}: Simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return scenario_id, pd.DataFrame()
    finally:
        try:
            if 'simulator' in locals():
                simulator.cleanup()
                del simulator
        except:
            pass


class DynamicDataGenerator:
    """
    Generates simulation data with controlled dynamic patterns
    for rate of change and impulse response analysis.
    
    Key features:
    - CDU-wise input variation for increased data diversity
    - Proper sequential FMU simulation (no time-series splitting)
    - Parallel scenario generation for multiple independent runs
    """
    
    def __init__(
        self,
        system_name: str = 'marconi100',
        n_workers: int = 8,
        vary_cdus: bool = True,
        **config_overrides
    ):
        """
        Initialize dynamic data generator.
        
        Args:
            system_name: System configuration name
            n_workers: Number of parallel workers for scenario-level parallelism
            vary_cdus: Whether to vary inputs across CDUs
            **config_overrides: Additional configuration overrides
        """
        self.system_name = system_name
        self.vary_cdus = vary_cdus
        
        # Load system configuration
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        self.num_cdus = self.config.get('NUM_CDUS', self.config.get('num_cdus', 49))
        self.config_overrides = {k: v for k, v in self.config.items() if k != 'system_name'}
        self.n_workers = n_workers
        
        # Minimum steps to ensure at least 1 hour of simulation
        self.min_steps = 3600
        
        logger.info(f"DynamicDataGenerator initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
        logger.info(f"CDU variation enabled: {vary_cdus}")
        logger.info(f"Parallel workers (scenario-level): {n_workers}")
    
    def _generate_cdu_phase_offsets(self, n_steps: int) -> np.ndarray:
        """
        Generate phase offsets for each CDU to create temporal diversity.
        
        Args:
            n_steps: Total number of time steps
            
        Returns:
            Array of phase offsets for each CDU (in time steps)
        """
        # Use golden ratio for well-distributed phases
        golden_ratio = (1 + np.sqrt(5)) / 2
        offsets = np.array([(i * golden_ratio) % 1.0 for i in range(self.num_cdus)])
        
        # Convert to step offsets (max 20% of total steps)
        max_offset = int(n_steps * 0.2)
        return (offsets * max_offset).astype(int)
    
    def _generate_cdu_amplitude_factors(self) -> np.ndarray:
        """
        Generate amplitude scaling factors for each CDU.
        Uses Latin Hypercube Sampling for good coverage.
        
        Returns:
            Array of amplitude factors for each CDU (0.7 to 1.3)
        """
        sampler = qmc.LatinHypercube(d=1, seed=42)
        samples = sampler.random(n=self.num_cdus)
        # Scale to 0.7-1.3 range
        return 0.7 + 0.6 * samples.flatten()
    
    def _apply_cdu_variation(
        self,
        base_signal: np.ndarray,
        cdu_idx: int,
        phase_offsets: np.ndarray,
        amplitude_factors: np.ndarray,
        signal_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Apply CDU-specific variation to a base signal.
        
        Args:
            base_signal: Base time-series signal
            cdu_idx: CDU index (0-based)
            phase_offsets: Phase offset array
            amplitude_factors: Amplitude factor array
            signal_range: (min, max) valid range for the signal
            
        Returns:
            Modified signal for this CDU
        """
        if not self.vary_cdus:
            return base_signal
        
        n_steps = len(base_signal)
        offset = phase_offsets[cdu_idx]
        amp_factor = amplitude_factors[cdu_idx]
        
        # Apply phase shift by rolling the signal
        shifted_signal = np.roll(base_signal, offset)
        
        # Apply amplitude scaling around the mean
        mean_val = np.mean(base_signal)
        scaled_signal = mean_val + amp_factor * (shifted_signal - mean_val)
        
        # Clip to valid range
        scaled_signal = np.clip(scaled_signal, signal_range[0], signal_range[1])
        
        return scaled_signal
    
    def _create_input_dataframe(
        self,
        q_flow_base: np.ndarray,
        t_air_base: np.ndarray,
        t_ext_series: np.ndarray,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Create input DataFrame with CDU-wise variations.
        
        Args:
            q_flow_base: Base Q_flow time-series (in kW, will be converted to W)
            t_air_base: Base T_Air time-series
            t_ext_series: T_ext time-series (common for all CDUs)
            input_ranges: Input value ranges
            
        Returns:
            DataFrame with all CDU inputs
        """
        n_steps = len(q_flow_base)
        
        # Generate CDU variation parameters
        phase_offsets = self._generate_cdu_phase_offsets(n_steps)
        amplitude_factors = self._generate_cdu_amplitude_factors()
        
        # Q_flow range in W (input is in kW)
        q_flow_range_w = (input_ranges['Q_flow'][0] * 1000, input_ranges['Q_flow'][1] * 1000)
        q_flow_base_w = q_flow_base * 1000  # Convert to W
        
        input_data_rows = []
        for i in range(n_steps):
            row = {
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext': t_ext_series[i]
            }
            
            for cdu_idx in range(self.num_cdus):
                # Apply CDU-specific variation
                q_flow_cdu = self._apply_cdu_variation(
                    q_flow_base_w, cdu_idx, phase_offsets, amplitude_factors, q_flow_range_w
                )[i]
                
                t_air_cdu = self._apply_cdu_variation(
                    t_air_base, cdu_idx, phase_offsets, amplitude_factors,
                    (input_ranges['T_Air'][0], input_ranges['T_Air'][1])
                )[i]
                
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx + 1}_cabinet_1_sources_Q_flow_total'] = q_flow_cdu
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx + 1}_cabinet_1_sources_T_Air'] = t_air_cdu
            
            input_data_rows.append(row)
        
        return pd.DataFrame(input_data_rows)
    
    def generate_step_response_data(
        self,
        n_steps: int = 10,
        step_duration: int = 180,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Generate data with step changes in inputs for impulse response analysis.
        
        With CDU variation, we get n_steps * num_cdus effective step responses.
        
        Args:
            n_steps: Number of step changes per input type
            step_duration: Duration of each step in time steps
            input_ranges: Input value ranges
            stabilization_hours: Hours for initial stabilization
            step_size: Simulation step size
            
        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Generating step response data with {n_steps} steps per input...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),  # kW
                'T_Air': (288.15, 308.15),
                'T_ext': (283.15, 313.15)
            }
        
        # Calculate total steps (ensure minimum of 3600)
        # 3 phases: Q_flow steps, T_Air steps, T_ext steps
        total_steps = max(n_steps * step_duration * 3, self.min_steps)
        steps_per_phase = total_steps // 3
        n_steps_adjusted = steps_per_phase // step_duration
        
        logger.info(f"Adjusted to {n_steps_adjusted} steps per input, {total_steps} total steps")
        
        # Initialize with mid values
        q_flow_mid = (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_air_mid = (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_mid = (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        q_flow_series = np.full(total_steps, q_flow_mid)
        t_air_series = np.full(total_steps, t_air_mid)
        t_ext_series = np.full(total_steps, t_ext_mid)
        
        # Phase 1: Q_flow step sequence
        for i in range(n_steps_adjusted):
            start = i * step_duration
            end = min(start + step_duration, steps_per_phase)
            q_flow_series[start:end] = input_ranges['Q_flow'][0] if i % 2 == 0 else input_ranges['Q_flow'][1]
        
        # Phase 2: T_Air step sequence
        offset = steps_per_phase
        for i in range(n_steps_adjusted):
            start = offset + i * step_duration
            end = min(start + step_duration, offset + steps_per_phase)
            t_air_series[start:end] = input_ranges['T_Air'][0] if i % 2 == 0 else input_ranges['T_Air'][1]
        
        # Phase 3: T_ext step sequence
        offset = 2 * steps_per_phase
        for i in range(n_steps_adjusted):
            start = offset + i * step_duration
            end = min(start + step_duration, total_steps)
            t_ext_series[start:end] = input_ranges['T_ext'][0] if i % 2 == 0 else input_ranges['T_ext'][1]
        
        # Create input DataFrame with CDU variations
        input_df = self._create_input_dataframe(
            q_flow_series, t_air_series, t_ext_series, input_ranges
        )
        
        logger.info(f"Created step response time-series with {len(input_df)} steps")
        logger.info(f"Effective step responses with CDU variation: {n_steps_adjusted * 3 * self.num_cdus}")
        
        return self._run_sequential_simulation(input_df, stabilization_hours, step_size)
    
    def generate_ramp_data(
        self,
        n_ramps: int = 6,
        ramp_duration: int = 300,
        hold_duration: int = 300,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Generate data with ramp (gradual) changes for rate analysis.
        
        Args:
            n_ramps: Number of ramp events per input
            ramp_duration: Duration of each ramp
            hold_duration: Duration to hold after ramp
            input_ranges: Input value ranges
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Generating ramp data with {n_ramps} ramps per input...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),
                'T_Air': (288.15, 308.15),
                'T_ext': (283.15, 313.15)
            }
        
        cycle_duration = ramp_duration + hold_duration
        total_steps = max(n_ramps * cycle_duration * 3, self.min_steps)
        steps_per_phase = total_steps // 3
        n_ramps_adjusted = steps_per_phase // cycle_duration
        
        logger.info(f"Adjusted to {n_ramps_adjusted} ramps per input, {total_steps} total steps")
        
        # Initialize with mid values
        q_flow_mid = (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_air_mid = (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_mid = (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        q_flow_series = np.full(total_steps, q_flow_mid)
        t_air_series = np.full(total_steps, t_air_mid)
        t_ext_series = np.full(total_steps, t_ext_mid)
        
        # Phase 1: Q_flow ramps
        for i in range(n_ramps_adjusted):
            cycle_start = i * cycle_duration
            ramp_end = cycle_start + ramp_duration
            hold_end = min(ramp_end + hold_duration, steps_per_phase)
            
            if i % 2 == 0:
                start_val, end_val = input_ranges['Q_flow'][0], input_ranges['Q_flow'][1]
            else:
                start_val, end_val = input_ranges['Q_flow'][1], input_ranges['Q_flow'][0]
            
            q_flow_series[cycle_start:ramp_end] = np.linspace(start_val, end_val, ramp_duration)
            q_flow_series[ramp_end:hold_end] = end_val
        
        # Phase 2: T_Air ramps
        offset = steps_per_phase
        for i in range(n_ramps_adjusted):
            cycle_start = offset + i * cycle_duration
            ramp_end = cycle_start + ramp_duration
            hold_end = min(ramp_end + hold_duration, offset + steps_per_phase)
            
            if i % 2 == 0:
                start_val, end_val = input_ranges['T_Air'][0], input_ranges['T_Air'][1]
            else:
                start_val, end_val = input_ranges['T_Air'][1], input_ranges['T_Air'][0]
            
            t_air_series[cycle_start:ramp_end] = np.linspace(start_val, end_val, ramp_duration)
            t_air_series[ramp_end:hold_end] = end_val
        
        # Phase 3: T_ext ramps
        offset = 2 * steps_per_phase
        for i in range(n_ramps_adjusted):
            cycle_start = offset + i * cycle_duration
            ramp_end = cycle_start + ramp_duration
            hold_end = min(ramp_end + hold_duration, total_steps)
            
            if i % 2 == 0:
                start_val, end_val = input_ranges['T_ext'][0], input_ranges['T_ext'][1]
            else:
                start_val, end_val = input_ranges['T_ext'][1], input_ranges['T_ext'][0]
            
            t_ext_series[cycle_start:ramp_end] = np.linspace(start_val, end_val, ramp_duration)
            t_ext_series[ramp_end:hold_end] = end_val
        
        # Create input DataFrame with CDU variations
        input_df = self._create_input_dataframe(
            q_flow_series, t_air_series, t_ext_series, input_ranges
        )
        
        logger.info(f"Created ramp time-series with {len(input_df)} steps")
        
        return self._run_sequential_simulation(input_df, stabilization_hours, step_size)
    
    def generate_varying_rate_data(
        self,
        n_samples: int = 3600,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        rate_range: Tuple[float, float] = (0.005, 0.05),
        stabilization_hours: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Generate data with varying rates of change to test derivative effects.
        
        Args:
            n_samples: Number of samples (minimum 3600)
            input_ranges: Input value ranges
            rate_range: Range of rates (fraction of range per step)
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            DataFrame with simulation results
        """
        n_samples = max(n_samples, self.min_steps)
        logger.info(f"Generating varying rate data with {n_samples} samples...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),
                'T_Air': (288.15, 308.15),
                'T_ext': (283.15, 313.15)
            }
        
        # Initialize at mid values
        q_flow = [(input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2]
        t_air = [(input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2]
        t_ext = [(input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2]
        
        # Generate correlated random walks with varying rates
        np.random.seed(42)
        
        for i in range(n_samples - 1):
            # Random rate for each variable
            rate_q = np.random.uniform(rate_range[0], rate_range[1])
            rate_t_air = np.random.uniform(rate_range[0], rate_range[1])
            rate_t_ext = np.random.uniform(rate_range[0], rate_range[1])
            
            # Random direction with momentum (tends to continue same direction)
            dir_q = np.random.choice([-1, 1])
            dir_t_air = np.random.choice([-1, 1])
            dir_t_ext = np.random.choice([-1, 1])
            
            # Compute ranges
            q_range = input_ranges['Q_flow'][1] - input_ranges['Q_flow'][0]
            t_air_range = input_ranges['T_Air'][1] - input_ranges['T_Air'][0]
            t_ext_range = input_ranges['T_ext'][1] - input_ranges['T_ext'][0]
            
            new_q = q_flow[-1] + dir_q * rate_q * q_range
            new_t_air = t_air[-1] + dir_t_air * rate_t_air * t_air_range
            new_t_ext = t_ext[-1] + dir_t_ext * rate_t_ext * t_ext_range
            
            # Clip and reflect at boundaries
            new_q = np.clip(new_q, input_ranges['Q_flow'][0], input_ranges['Q_flow'][1])
            new_t_air = np.clip(new_t_air, input_ranges['T_Air'][0], input_ranges['T_Air'][1])
            new_t_ext = np.clip(new_t_ext, input_ranges['T_ext'][0], input_ranges['T_ext'][1])
            
            q_flow.append(new_q)
            t_air.append(new_t_air)
            t_ext.append(new_t_ext)
        
        q_flow_series = np.array(q_flow)
        t_air_series = np.array(t_air)
        t_ext_series = np.array(t_ext)
        
        # Create input DataFrame with CDU variations
        input_df = self._create_input_dataframe(
            q_flow_series, t_air_series, t_ext_series, input_ranges
        )
        
        logger.info(f"Created varying rate time-series with {len(input_df)} steps")
        logger.info(f"Q_flow derivative range: [{np.min(np.diff(q_flow_series)):.3f}, {np.max(np.diff(q_flow_series)):.3f}] kW/step")
        
        return self._run_sequential_simulation(input_df, stabilization_hours, step_size)
    
    def generate_combined_dynamic_data(
        self,
        n_samples: int = 3600,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Generate comprehensive dynamic data combining steps, ramps, and random variations.
        
        Args:
            n_samples: Total number of samples (minimum 3600)
            input_ranges: Input value ranges
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            DataFrame with simulation results
        """
        n_samples = max(n_samples, self.min_steps)
        logger.info(f"Generating combined dynamic data with {n_samples} total samples...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),
                'T_Air': (288.15, 308.15),
                'T_ext': (283.15, 313.15)
            }
        
        # Divide samples into segments
        step_samples = n_samples // 4
        ramp_samples = n_samples // 4
        varying_samples = n_samples - step_samples - ramp_samples
        
        # Initialize series
        q_flow_series = []
        t_air_series = []
        t_ext_series = []
        
        # Mid values
        q_mid = (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_air_mid = (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_mid = (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        # 1. Step changes section (25%)
        step_duration = max(step_samples // 8, 50)
        n_steps = step_samples // step_duration
        for i in range(n_steps):
            if i % 2 == 0:
                q_val = input_ranges['Q_flow'][0]
                t_val = input_ranges['T_Air'][0]
                t_ext_val = input_ranges['T_ext'][0]
            else:
                q_val = input_ranges['Q_flow'][1]
                t_val = input_ranges['T_Air'][1]
                t_ext_val = input_ranges['T_ext'][1]
            
            q_flow_series.extend([q_val] * step_duration)
            t_air_series.extend([t_val] * step_duration)
            t_ext_series.extend([t_ext_val] * step_duration)
        
        # Pad if needed
        while len(q_flow_series) < step_samples:
            q_flow_series.append(q_mid)
            t_air_series.append(t_air_mid)
            t_ext_series.append(t_ext_mid)
        
        # 2. Ramp section (25%)
        ramp_duration = max(ramp_samples // 4, 100)
        n_ramps = ramp_samples // ramp_duration
        for i in range(n_ramps):
            if i % 2 == 0:
                q_start, q_end = input_ranges['Q_flow'][0], input_ranges['Q_flow'][1]
                t_start, t_end = input_ranges['T_Air'][0], input_ranges['T_Air'][1]
                t_ext_start, t_ext_end = input_ranges['T_ext'][0], input_ranges['T_ext'][1]
            else:
                q_start, q_end = input_ranges['Q_flow'][1], input_ranges['Q_flow'][0]
                t_start, t_end = input_ranges['T_Air'][1], input_ranges['T_Air'][0]
                t_ext_start, t_ext_end = input_ranges['T_ext'][1], input_ranges['T_ext'][0]
            
            q_flow_series.extend(np.linspace(q_start, q_end, ramp_duration))
            t_air_series.extend(np.linspace(t_start, t_end, ramp_duration))
            t_ext_series.extend(np.linspace(t_ext_start, t_ext_end, ramp_duration))
        
        # 3. Varying rate section (50%) - random walk
        np.random.seed(42)
        q_current = q_flow_series[-1] if q_flow_series else q_mid
        t_current = t_air_series[-1] if t_air_series else t_air_mid
        t_ext_current = t_ext_series[-1] if t_ext_series else t_ext_mid
        
        for _ in range(varying_samples):
            rate = np.random.uniform(0.005, 0.03)
            
            q_range = input_ranges['Q_flow'][1] - input_ranges['Q_flow'][0]
            t_range = input_ranges['T_Air'][1] - input_ranges['T_Air'][0]
            t_ext_range = input_ranges['T_ext'][1] - input_ranges['T_ext'][0]
            
            q_current += np.random.choice([-1, 1]) * rate * q_range
            t_current += np.random.choice([-1, 1]) * rate * t_range
            t_ext_current += np.random.choice([-1, 1]) * rate * t_ext_range
            
            q_current = np.clip(q_current, input_ranges['Q_flow'][0], input_ranges['Q_flow'][1])
            t_current = np.clip(t_current, input_ranges['T_Air'][0], input_ranges['T_Air'][1])
            t_ext_current = np.clip(t_ext_current, input_ranges['T_ext'][0], input_ranges['T_ext'][1])
            
            q_flow_series.append(q_current)
            t_air_series.append(t_current)
            t_ext_series.append(t_ext_current)
        
        q_flow_series = np.array(q_flow_series)
        t_air_series = np.array(t_air_series)
        t_ext_series = np.array(t_ext_series)
        
        # Create input DataFrame with CDU variations
        input_df = self._create_input_dataframe(
            q_flow_series, t_air_series, t_ext_series, input_ranges
        )
        
        logger.info(f"Created combined dynamic time-series with {len(input_df)} steps")
        
        return self._run_sequential_simulation(input_df, stabilization_hours, step_size)
    
    def generate_multi_scenario_data(
        self,
        n_scenarios: int = 4,
        steps_per_scenario: int = 1000,
        scenario_type: str = 'combined',
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """
        Generate multiple independent scenarios in parallel.
        Each scenario runs sequentially but different scenarios run in parallel.
        
        Args:
            n_scenarios: Number of independent scenarios
            steps_per_scenario: Steps per scenario (minimum 900 to ensure 3600 total)
            scenario_type: Type of scenario ('step', 'ramp', 'varying', 'combined')
            input_ranges: Input value ranges
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            Combined DataFrame from all scenarios
        """
        # Ensure minimum total steps
        steps_per_scenario = max(steps_per_scenario, self.min_steps // n_scenarios + 1)
        
        logger.info(f"Generating {n_scenarios} parallel scenarios, {steps_per_scenario} steps each...")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),
                'T_Air': (288.15, 308.15),
                'T_ext': (283.15, 313.15)
            }
        
        # Generate input data for each scenario with different random seeds
        scenario_inputs = []
        for scenario_id in range(n_scenarios):
            np.random.seed(42 + scenario_id * 100)
            
            if scenario_type == 'step':
                input_df = self._generate_step_inputs(steps_per_scenario, input_ranges)
            elif scenario_type == 'ramp':
                input_df = self._generate_ramp_inputs(steps_per_scenario, input_ranges)
            elif scenario_type == 'varying':
                input_df = self._generate_varying_inputs(steps_per_scenario, input_ranges)
            else:  # combined
                input_df = self._generate_combined_inputs(steps_per_scenario, input_ranges)
            
            scenario_inputs.append(input_df)
        
        # Run scenarios in parallel
        logger.info(f"Launching {n_scenarios} parallel simulations...")
        
        with ProcessPoolExecutor(max_workers=min(self.n_workers, n_scenarios)) as executor:
            futures = {}
            for scenario_id, input_df in enumerate(scenario_inputs):
                future = executor.submit(
                    _run_single_scenario,
                    scenario_id,
                    input_df,
                    self.system_name,
                    self.config_overrides,
                    stabilization_hours,
                    step_size
                )
                futures[future] = scenario_id
            
            results = []
            for future in as_completed(futures):
                scenario_id, result_df = future.result()
                if not result_df.empty:
                    results.append(result_df)
                    logger.info(f"Scenario {scenario_id} completed with {len(result_df)} samples")
                else:
                    logger.warning(f"Scenario {scenario_id} returned empty results")
        
        if not results:
            logger.error("All scenarios failed!")
            return pd.DataFrame()
        
        combined_df = pd.concat(results, ignore_index=True)
        logger.info(f"Combined results: {len(combined_df)} total samples from {len(results)} scenarios")
        
        return combined_df
    
    def _generate_step_inputs(
        self,
        n_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """Generate step input pattern."""
        step_duration = max(n_steps // 10, 50)
        
        q_mid = (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_air_mid = (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_mid = (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        q_flow_series = np.full(n_steps, q_mid)
        t_air_series = np.full(n_steps, t_air_mid)
        t_ext_series = np.full(n_steps, t_ext_mid)
        
        for i in range(n_steps // step_duration):
            start = i * step_duration
            end = min(start + step_duration, n_steps)
            
            if i % 2 == 0:
                q_flow_series[start:end] = input_ranges['Q_flow'][0]
                t_air_series[start:end] = input_ranges['T_Air'][0]
            else:
                q_flow_series[start:end] = input_ranges['Q_flow'][1]
                t_air_series[start:end] = input_ranges['T_Air'][1]
        
        return self._create_input_dataframe(q_flow_series, t_air_series, t_ext_series, input_ranges)
    
    def _generate_ramp_inputs(
        self,
        n_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """Generate ramp input pattern."""
        ramp_duration = max(n_steps // 6, 100)
        
        q_flow_series = []
        t_air_series = []
        t_ext_series = []
        
        for i in range(n_steps // ramp_duration + 1):
            if i % 2 == 0:
                q_flow_series.extend(np.linspace(input_ranges['Q_flow'][0], input_ranges['Q_flow'][1], ramp_duration))
                t_air_series.extend(np.linspace(input_ranges['T_Air'][0], input_ranges['T_Air'][1], ramp_duration))
                t_ext_series.extend(np.linspace(input_ranges['T_ext'][0], input_ranges['T_ext'][1], ramp_duration))
            else:
                q_flow_series.extend(np.linspace(input_ranges['Q_flow'][1], input_ranges['Q_flow'][0], ramp_duration))
                t_air_series.extend(np.linspace(input_ranges['T_Air'][1], input_ranges['T_Air'][0], ramp_duration))
                t_ext_series.extend(np.linspace(input_ranges['T_ext'][1], input_ranges['T_ext'][0], ramp_duration))
        
        q_flow_series = np.array(q_flow_series[:n_steps])
        t_air_series = np.array(t_air_series[:n_steps])
        t_ext_series = np.array(t_ext_series[:n_steps])
        
        return self._create_input_dataframe(q_flow_series, t_air_series, t_ext_series, input_ranges)
    
    def _generate_varying_inputs(
        self,
        n_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """Generate varying rate input pattern."""
        q_mid = (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_air_mid = (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_mid = (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        q_flow = [q_mid]
        t_air = [t_air_mid]
        t_ext = [t_ext_mid]
        
        for _ in range(n_steps - 1):
            rate = np.random.uniform(0.005, 0.05)
            
            q_range = input_ranges['Q_flow'][1] - input_ranges['Q_flow'][0]
            t_range = input_ranges['T_Air'][1] - input_ranges['T_Air'][0]
            t_ext_range = input_ranges['T_ext'][1] - input_ranges['T_ext'][0]
            
            q_flow.append(np.clip(q_flow[-1] + np.random.choice([-1, 1]) * rate * q_range,
                                  input_ranges['Q_flow'][0], input_ranges['Q_flow'][1]))
            t_air.append(np.clip(t_air[-1] + np.random.choice([-1, 1]) * rate * t_range,
                                 input_ranges['T_Air'][0], input_ranges['T_Air'][1]))
            t_ext.append(np.clip(t_ext[-1] + np.random.choice([-1, 1]) * rate * t_ext_range,
                                 input_ranges['T_ext'][0], input_ranges['T_ext'][1]))
        
        return self._create_input_dataframe(np.array(q_flow), np.array(t_air), np.array(t_ext), input_ranges)
    
    def _generate_combined_inputs(
        self,
        n_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """Generate combined input pattern."""
        # Mix of step, ramp, and varying
        step_portion = n_steps // 3
        ramp_portion = n_steps // 3
        vary_portion = n_steps - step_portion - ramp_portion
        
        q_flow_series = []
        t_air_series = []
        t_ext_series = []
        
        # Steps
        step_dur = max(step_portion // 5, 30)
        for i in range(step_portion // step_dur):
            val = input_ranges['Q_flow'][0] if i % 2 == 0 else input_ranges['Q_flow'][1]
            t_val = input_ranges['T_Air'][0] if i % 2 == 0 else input_ranges['T_Air'][1]
            t_ext_val = input_ranges['T_ext'][0] if i % 2 == 0 else input_ranges['T_ext'][1]
            q_flow_series.extend([val] * step_dur)
            t_air_series.extend([t_val] * step_dur)
            t_ext_series.extend([t_ext_val] * step_dur)
        
        # Ramps
        ramp_dur = max(ramp_portion // 3, 50)
        for i in range(3):
            if i % 2 == 0:
                q_flow_series.extend(np.linspace(input_ranges['Q_flow'][0], input_ranges['Q_flow'][1], ramp_dur))
                t_air_series.extend(np.linspace(input_ranges['T_Air'][0], input_ranges['T_Air'][1], ramp_dur))
                t_ext_series.extend(np.linspace(input_ranges['T_ext'][0], input_ranges['T_ext'][1], ramp_dur))
            else:
                q_flow_series.extend(np.linspace(input_ranges['Q_flow'][1], input_ranges['Q_flow'][0], ramp_dur))
                t_air_series.extend(np.linspace(input_ranges['T_Air'][1], input_ranges['T_Air'][0], ramp_dur))
                t_ext_series.extend(np.linspace(input_ranges['T_ext'][1], input_ranges['T_ext'][0], ramp_dur))
        
        # Varying
        q_current = q_flow_series[-1] if q_flow_series else (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2
        t_current = t_air_series[-1] if t_air_series else (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2
        t_ext_current = t_ext_series[-1] if t_ext_series else (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        
        for _ in range(vary_portion):
            rate = np.random.uniform(0.005, 0.03)
            q_current = np.clip(q_current + np.random.choice([-1, 1]) * rate * (input_ranges['Q_flow'][1] - input_ranges['Q_flow'][0]),
                               input_ranges['Q_flow'][0], input_ranges['Q_flow'][1])
            t_current = np.clip(t_current + np.random.choice([-1, 1]) * rate * (input_ranges['T_Air'][1] - input_ranges['T_Air'][0]),
                               input_ranges['T_Air'][0], input_ranges['T_Air'][1])
            t_ext_current = np.clip(t_ext_current + np.random.choice([-1, 1]) * rate * (input_ranges['T_ext'][1] - input_ranges['T_ext'][0]),
                                   input_ranges['T_ext'][0], input_ranges['T_ext'][1])
            q_flow_series.append(q_current)
            t_air_series.append(t_current)
            t_ext_series.append(t_ext_current)
        
        # Ensure correct length
        q_flow_series = np.array(q_flow_series[:n_steps])
        t_air_series = np.array(t_air_series[:n_steps])
        t_ext_series = np.array(t_ext_series[:n_steps])
        
        # Pad if needed
        if len(q_flow_series) < n_steps:
            pad_len = n_steps - len(q_flow_series)
            q_flow_series = np.concatenate([q_flow_series, np.full(pad_len, q_flow_series[-1])])
            t_air_series = np.concatenate([t_air_series, np.full(pad_len, t_air_series[-1])])
            t_ext_series = np.concatenate([t_ext_series, np.full(pad_len, t_ext_series[-1])])
        
        return self._create_input_dataframe(q_flow_series, t_air_series, t_ext_series, input_ranges)
    
    def _run_sequential_simulation(
        self,
        input_df: pd.DataFrame,
        stabilization_hours: int,
        step_size: int
    ) -> pd.DataFrame:
        """
        Run sequential FMU simulation to maintain state causality.
        
        Args:
            input_df: Input data DataFrame
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            Simulation results DataFrame
        """
        from fmu2ml.simulation.fmu_simulator import FMUSimulator
        
        logger.info(f"Running sequential FMU simulation with {len(input_df)} steps...")
        logger.info("Note: Sequential execution maintains state causality across time steps")
        
        try:
            simulator = FMUSimulator(
                system_name=self.system_name,
                **self.config_overrides
            )
            
            results = simulator.run_simulation(
                input_data=input_df,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                save_history=False
            )
            
            logger.info(f"Simulation completed: {len(results)} samples generated")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        finally:
            try:
                if 'simulator' in locals():
                    simulator.cleanup()
                    del simulator
            except:
                pass
