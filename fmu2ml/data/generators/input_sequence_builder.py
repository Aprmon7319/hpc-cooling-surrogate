"""
Builds continuous input time series from sequenced scenarios.
Handles transitions, ramps, and dynamic scenario generation.
Supports independent per-CDU scenario generation.

OPTIMIZED for DeepONet: Reduced smoothing, more T_ext variation.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .scenario_definitions import (
    ScenarioSpec, ScenarioType, Phase, OperatingPoint
)
from .scenario_sequencer import SequencedScenario


@dataclass
class InputConfig:
    """Configuration for input generation"""
    n_cdus: int = 49
    q_flow_max_kw: float = 100.0  # Max power per CDU in kW
    q_flow_min_kw: float = 10.0   # Min power per CDU in kW
    timestep_seconds: int = 1
    transition_ramp_seconds: int = 30  # Faster transitions (was 60)
    jitter_fraction: float = 0.03  # ±3% amplitude jitter per CDU
    timing_offset_fraction: float = 0.15  # ±15% timing offset per CDU
    t_air_smoothing_sigma: float = 5.0  # REDUCED from 20.0
    t_air_noise_amplitude: float = 1.0  # INCREASED: ±1K noise (was ~0.3K)
    seed: int = 42


class InputSequenceBuilder:
    """
    Builds continuous FMU input arrays from scenario sequence.
    Supports independent per-CDU scenario generation.
    
    OPTIMIZED for DeepONet training.
    """
    
    def __init__(self, config: InputConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def _fraction_to_power(self, fraction: float) -> float:
        """Convert Q_flow fraction to power in Watts"""
        q_range = self.config.q_flow_max_kw - self.config.q_flow_min_kw
        power_kw = self.config.q_flow_min_kw + fraction * q_range
        return power_kw * 1000  # Convert to Watts
    
    def _generate_ramp(
        self,
        start_value: float,
        end_value: float,
        n_steps: int
    ) -> np.ndarray:
        """Generate smooth ramp between two values"""
        if n_steps <= 1:
            return np.array([end_value])
        
        # Use cosine interpolation for smoother transitions
        t = np.linspace(0, np.pi, n_steps)
        interp = 0.5 * (1 - np.cos(t))  # 0 to 1 smoothly
        
        return start_value + (end_value - start_value) * interp
    
    def _generate_realistic_t_air(
        self,
        base_t_air: float,
        n_steps: int,
        rng: np.random.RandomState,
        noise_amplitude: float = None
    ) -> np.ndarray:
        """
        Generate physically realistic T_air with natural variations.
        
        CHANGED: Reduced smoothing, increased variation for DeepONet training.
        T_air should vary by ±1-2K on timescales of 30-120 seconds.
        """
        if noise_amplitude is None:
            noise_amplitude = self.config.t_air_noise_amplitude
        
        dt = self.config.timestep_seconds
        
        # Faster OU process (reduced time constant from 300s to 60s)
        tau = 60.0  # 1 minute time constant
        theta = 1.0 / tau
        sigma = noise_amplitude * np.sqrt(2 * theta)
        
        t_air = np.zeros(n_steps)
        t_air[0] = base_t_air + rng.uniform(-0.5, 0.5)
        
        for i in range(1, n_steps):
            drift = theta * (base_t_air - t_air[i-1]) * dt
            diffusion = sigma * np.sqrt(dt) * rng.normal()
            t_air[i] = t_air[i-1] + drift + diffusion
        
        # Light smoothing only (reduced from sigma=20 to sigma=5)
        if n_steps > 10:
            sigma_samples = self.config.t_air_smoothing_sigma / self.config.timestep_seconds
            t_air = gaussian_filter1d(t_air, sigma=min(sigma_samples, n_steps // 10))
        
        return t_air
    
    def _generate_steady_state(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for steady-state scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        op = scenario.operating_points[0]
        
        # Q_flow with realistic fluctuations (OU process)
        q_base = self._fraction_to_power(op.q_flow_fraction)
        tau_power = 30.0  # 30 second time constant for power
        theta_power = 1.0 / tau_power
        sigma_power = q_base * 0.01 * np.sqrt(2 * theta_power)  # 1% volatility
        
        q_flow = np.zeros(n_steps)
        q_flow[0] = q_base
        
        for i in range(1, n_steps):
            drift = theta_power * (q_base - q_flow[i-1])
            diffusion = sigma_power * rng.normal()
            q_flow[i] = q_flow[i-1] + drift + diffusion
        
        # T_air with realistic variations
        t_air = self._generate_realistic_t_air(op.t_air_k, n_steps, rng, noise_amplitude=0.8)
        
        # T_ext constant for steady-state
        t_ext = np.full(n_steps, op.t_ext_k)
        
        phase = np.full(n_steps, Phase.STEADY_STATE.value)
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_step_response(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for step response scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        ramp_steps = scenario.ramp_duration_seconds // self.config.timestep_seconds
        
        q_flow = np.zeros(n_steps)
        t_air = np.zeros(n_steps)
        t_ext = np.zeros(n_steps)
        phase = np.empty(n_steps, dtype=object)
        
        ops = scenario.operating_points
        trans_times = scenario.transition_times
        
        current_step = 0
        current_op = ops[0]
        
        for i, trans_time in enumerate(trans_times):
            trans_step = trans_time // self.config.timestep_seconds
            next_op = ops[i + 1]
            
            # Hold at current operating point
            hold_end = min(trans_step, n_steps)
            hold_len = hold_end - current_step
            if hold_len > 0:
                q_flow[current_step:hold_end] = self._fraction_to_power(current_op.q_flow_fraction)
                t_air[current_step:hold_end] = current_op.t_air_k
                t_ext[current_step:hold_end] = current_op.t_ext_k
                phase[current_step:hold_end] = Phase.STEADY_STATE.value
            
            # Ramp to next operating point
            ramp_end = min(hold_end + ramp_steps, n_steps)
            actual_ramp_steps = ramp_end - hold_end
            
            if actual_ramp_steps > 0:
                q_flow[hold_end:ramp_end] = self._generate_ramp(
                    self._fraction_to_power(current_op.q_flow_fraction),
                    self._fraction_to_power(next_op.q_flow_fraction),
                    actual_ramp_steps
                )
                t_air[hold_end:ramp_end] = self._generate_ramp(
                    current_op.t_air_k, next_op.t_air_k, actual_ramp_steps
                )
                t_ext[hold_end:ramp_end] = self._generate_ramp(
                    current_op.t_ext_k, next_op.t_ext_k, actual_ramp_steps
                )
                phase[hold_end:ramp_end] = Phase.TRANSITION.value
            
            # Settling period
            settling_end = min(ramp_end + 90, n_steps)  # Reduced from 120
            if settling_end > ramp_end:
                q_flow[ramp_end:settling_end] = self._fraction_to_power(next_op.q_flow_fraction)
                t_air[ramp_end:settling_end] = next_op.t_air_k
                t_ext[ramp_end:settling_end] = next_op.t_ext_k
                phase[ramp_end:settling_end] = Phase.SETTLING.value
            
            current_step = settling_end
            current_op = next_op
        
        # Fill remaining
        if current_step < n_steps:
            q_flow[current_step:] = self._fraction_to_power(current_op.q_flow_fraction)
            t_air[current_step:] = current_op.t_air_k
            t_ext[current_step:] = current_op.t_ext_k
            phase[current_step:] = Phase.STEADY_STATE.value
        
        # Add T_air variation (reduced smoothing)
        t_air += rng.normal(0, 0.3, n_steps)
        if n_steps > 10:
            sigma_samples = self.config.t_air_smoothing_sigma / self.config.timestep_seconds
            t_air = gaussian_filter1d(t_air, sigma=min(sigma_samples, n_steps // 10))
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_t_ext_step(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for T_ext step scenario"""
        # Same structure as regular step, but T_ext changes
        return self._generate_step_response(scenario, rng)
    
    def _generate_t_ext_ramp(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for T_ext ramp scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        
        start_op = scenario.operating_points[0]
        end_op = scenario.operating_points[1]
        
        hold_steps = scenario.transition_times[0] // self.config.timestep_seconds
        ramp_steps = scenario.ramp_duration_seconds // self.config.timestep_seconds
        
        q_flow = np.zeros(n_steps)
        t_air = np.zeros(n_steps)
        t_ext = np.zeros(n_steps)
        phase = np.empty(n_steps, dtype=object)
        
        # Initial hold
        hold_end = min(hold_steps, n_steps)
        q_flow[:hold_end] = self._fraction_to_power(start_op.q_flow_fraction)
        t_air[:hold_end] = start_op.t_air_k
        t_ext[:hold_end] = start_op.t_ext_k
        phase[:hold_end] = Phase.STEADY_STATE.value
        
        # T_ext ramp
        ramp_end = min(hold_steps + ramp_steps, n_steps)
        actual_ramp = ramp_end - hold_end
        if actual_ramp > 0:
            q_flow[hold_end:ramp_end] = self._fraction_to_power(start_op.q_flow_fraction)
            t_air[hold_end:ramp_end] = start_op.t_air_k
            t_ext[hold_end:ramp_end] = self._generate_ramp(
                start_op.t_ext_k, end_op.t_ext_k, actual_ramp
            )
            phase[hold_end:ramp_end] = Phase.TRANSITION.value
        
        # Final hold
        if ramp_end < n_steps:
            q_flow[ramp_end:] = self._fraction_to_power(end_op.q_flow_fraction)
            t_air[ramp_end:] = end_op.t_air_k
            t_ext[ramp_end:] = end_op.t_ext_k
            phase[ramp_end:] = Phase.STEADY_STATE.value
        
        # Add T_air variation
        t_air += rng.normal(0, 0.4, n_steps)
        if n_steps > 10:
            sigma_samples = self.config.t_air_smoothing_sigma / self.config.timestep_seconds
            t_air = gaussian_filter1d(t_air, sigma=min(sigma_samples, n_steps // 10))
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_ramp_sweep(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for Q_flow ramp sweep scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        
        start_op = scenario.operating_points[0]
        end_op = scenario.operating_points[1]
        
        hold_steps = scenario.transition_times[0] // self.config.timestep_seconds
        ramp_steps = scenario.ramp_duration_seconds // self.config.timestep_seconds
        
        q_flow = np.zeros(n_steps)
        t_air = np.zeros(n_steps)
        t_ext = np.zeros(n_steps)
        phase = np.empty(n_steps, dtype=object)
        
        # Initial hold
        hold_end = min(hold_steps, n_steps)
        q_flow[:hold_end] = self._fraction_to_power(start_op.q_flow_fraction)
        t_air[:hold_end] = start_op.t_air_k
        t_ext[:hold_end] = start_op.t_ext_k
        phase[:hold_end] = Phase.STEADY_STATE.value
        
        # Q_flow ramp
        ramp_end = min(hold_steps + ramp_steps, n_steps)
        actual_ramp = ramp_end - hold_end
        if actual_ramp > 0:
            q_flow[hold_end:ramp_end] = self._generate_ramp(
                self._fraction_to_power(start_op.q_flow_fraction),
                self._fraction_to_power(end_op.q_flow_fraction),
                actual_ramp
            )
            t_air[hold_end:ramp_end] = self._generate_ramp(
                start_op.t_air_k, end_op.t_air_k, actual_ramp
            )
            t_ext[hold_end:ramp_end] = start_op.t_ext_k  # T_ext constant during ramp
            phase[hold_end:ramp_end] = Phase.TRANSITION.value
        
        # Final hold
        if ramp_end < n_steps:
            q_flow[ramp_end:] = self._fraction_to_power(end_op.q_flow_fraction)
            t_air[ramp_end:] = end_op.t_air_k
            t_ext[ramp_end:] = end_op.t_ext_k
            phase[ramp_end:] = Phase.STEADY_STATE.value
        
        # Add T_air variation
        t_air += rng.normal(0, 0.3, n_steps)
        if n_steps > 10:
            sigma_samples = self.config.t_air_smoothing_sigma / self.config.timestep_seconds
            t_air = gaussian_filter1d(t_air, sigma=min(sigma_samples, n_steps // 10))
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_combined_stress(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for combined stress scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        params = scenario.combined_params
        
        combined_type = params['type']
        q_start = params['q_flow_start']
        t_ext_start = params['t_ext_start']
        
        t = np.arange(n_steps) * self.config.timestep_seconds
        duration = scenario.duration_seconds
        
        if combined_type == 'summer_afternoon':
            # Q_flow increases as T_ext increases (correlated)
            progress = t / duration
            q_fraction = q_start + 0.35 * np.sin(np.pi * progress)  # Rises then falls
            t_ext = t_ext_start + 15.0 * np.sin(np.pi * progress)   # Rises then falls
            
        elif combined_type == 'winter_morning':
            # Q_flow increases as T_ext is low and slowly rises
            progress = t / duration
            q_fraction = q_start + 0.30 * (1 - np.exp(-3 * progress))  # Exponential rise
            t_ext = t_ext_start - 10.0 + 15.0 * progress  # Linear warming
            
        elif combined_type == 'cooling_maintenance':
            # Q_flow drops while T_ext rises (anticorrelated)
            progress = t / duration
            q_fraction = q_start - 0.25 * progress
            t_ext = t_ext_start + 12.0 * progress
            
        elif combined_type == 'night_cooldown':
            # Both Q_flow and T_ext decrease (correlated decrease)
            progress = t / duration
            q_fraction = q_start - 0.20 * progress
            t_ext = t_ext_start - 10.0 * progress
            
        elif combined_type == 'random_correlated':
            # Random walk but Q and T_ext move together
            base_walk = np.cumsum(rng.normal(0, 0.01, n_steps))
            base_walk = base_walk - np.mean(base_walk)
            q_fraction = q_start + 0.15 * base_walk / (np.std(base_walk) + 0.01)
            t_ext = t_ext_start + 8.0 * base_walk / (np.std(base_walk) + 0.01)
            
        else:  # random_anticorrelated
            # Random walk but Q and T_ext move opposite
            base_walk = np.cumsum(rng.normal(0, 0.01, n_steps))
            base_walk = base_walk - np.mean(base_walk)
            q_fraction = q_start + 0.15 * base_walk / (np.std(base_walk) + 0.01)
            t_ext = t_ext_start - 8.0 * base_walk / (np.std(base_walk) + 0.01)
        
        # Clip to valid ranges
        q_fraction = np.clip(q_fraction, 0.05, 1.0)
        t_ext = np.clip(t_ext, 265.0, 315.0)
        
        q_flow = np.array([self._fraction_to_power(f) for f in q_fraction])
        
        # T_air follows with some thermal lag
        t_air = self._generate_realistic_t_air(298.0, n_steps, rng, noise_amplitude=1.2)
        
        phase = np.full(n_steps, Phase.STEADY_STATE.value)
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_sinusoidal(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for sinusoidal scenario"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        t = np.arange(n_steps) * self.config.timestep_seconds
        
        phase = np.full(n_steps, Phase.STEADY_STATE.value)
        phase_offset = rng.uniform(0, 2 * np.pi)
        
        if hasattr(scenario, 'sinusoid_params') and scenario.sinusoid_params:
            params = scenario.sinusoid_params
            
            if params.get('variable') == 't_ext':
                # T_ext sinusoid
                q_flow = np.full(n_steps, self._fraction_to_power(params.get('q_flow_base', 0.50)))
                t_ext = params['mean'] + params['amplitude'] * np.sin(
                    2 * np.pi * t / params['period_seconds'] + phase_offset
                )
                t_air = self._generate_realistic_t_air(298.0, n_steps, rng)
            else:
                # Q_flow sinusoid
                q_fraction = params['mean'] + params['amplitude'] * np.sin(
                    2 * np.pi * t / params['period_seconds'] + phase_offset
                )
                q_fraction = np.clip(q_fraction, 0.05, 1.0)
                q_flow = np.array([self._fraction_to_power(f) for f in q_fraction])
                t_air = self._generate_realistic_t_air(298.0, n_steps, rng)
                t_ext = np.full(n_steps, params.get('t_ext_base', 288.0))
        
        elif hasattr(scenario, 'diurnal_params') and scenario.diurnal_params:
            params = scenario.diurnal_params
            # Diurnal T_ext pattern
            diurnal = 0.5 * (params['t_ext_max'] - params['t_ext_min']) * (
                1 - np.cos(2 * np.pi * t / params['period_seconds'] + phase_offset)
            ) + params['t_ext_min']
            
            q_flow = np.full(n_steps, self._fraction_to_power(params.get('q_flow_base', 0.50)))
            t_air = self._generate_realistic_t_air(298.0, n_steps, rng)
            t_ext = diurnal
        
        else:
            q_flow = np.full(n_steps, self._fraction_to_power(0.50))
            t_air = self._generate_realistic_t_air(298.0, n_steps, rng)
            t_ext = np.full(n_steps, 288.0)
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_random_realistic(
        self,
        scenario: ScenarioSpec,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays for random realistic scenario using Ornstein-Uhlenbeck"""
        n_steps = scenario.duration_seconds // self.config.timestep_seconds
        
        params = scenario.ou_params
        
        dt = self.config.timestep_seconds
        theta = params['mean_reversion']
        mu = params['mean']
        sigma = params['volatility']
        jump_prob = params['jump_prob']
        
        # Q_flow OU process
        q_fraction = np.zeros(n_steps)
        q_fraction[0] = mu + rng.uniform(-0.15, 0.15)
        
        for i in range(1, n_steps):
            dW = rng.normal(0, np.sqrt(dt))
            q_fraction[i] = q_fraction[i-1] + theta * (mu - q_fraction[i-1]) * dt + sigma * dW
            
            if rng.random() < jump_prob * dt:
                jump_size = rng.uniform(-0.12, 0.12)
                q_fraction[i] += jump_size
            
            q_fraction[i] = np.clip(q_fraction[i], 0.05, 1.0)
        
        q_flow = np.array([self._fraction_to_power(f) for f in q_fraction])
        
        # T_air with variation
        t_air = self._generate_realistic_t_air(298.0, n_steps, rng, noise_amplitude=1.0)
        
        # T_ext with slow drift if enabled
        t_ext_base = params.get('t_ext_base', 288.0)
        if params.get('t_ext_variation', False):
            t_ext_vol = params.get('t_ext_volatility', 0.01)
            t_ext = np.zeros(n_steps)
            t_ext[0] = t_ext_base
            for i in range(1, n_steps):
                t_ext[i] = t_ext[i-1] + t_ext_vol * rng.normal() * np.sqrt(dt)
            t_ext = np.clip(t_ext, 265.0, 315.0)
            # Smooth T_ext (weather changes are slow)
            t_ext = gaussian_filter1d(t_ext, sigma=60)
        else:
            t_ext = np.full(n_steps, t_ext_base)
        
        phase = np.full(n_steps, Phase.STEADY_STATE.value)
        
        return q_flow, t_air, t_ext, phase
    
    def _generate_transition_ramp(
        self,
        from_op: OperatingPoint,
        to_op: OperatingPoint
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate transition ramp between scenarios"""
        n_steps = self.config.transition_ramp_seconds // self.config.timestep_seconds
        
        q_flow = self._generate_ramp(
            self._fraction_to_power(from_op.q_flow_fraction),
            self._fraction_to_power(to_op.q_flow_fraction),
            n_steps
        )
        t_air = self._generate_ramp(from_op.t_air_k, to_op.t_air_k, n_steps)
        t_ext = self._generate_ramp(from_op.t_ext_k, to_op.t_ext_k, n_steps)
        phase = np.full(n_steps, Phase.TRANSITION.value)
        
        return q_flow, t_air, t_ext, phase
    
    def build_single_cdu_sequence(
        self,
        sequenced_scenarios: List[SequencedScenario],
        cdu_rng: Optional[np.random.RandomState] = None
    ) -> pd.DataFrame:
        """Build complete input sequence for a single CDU"""
        
        if cdu_rng is None:
            cdu_rng = self.rng
        
        all_q_flow = []
        all_t_air = []
        all_t_ext = []
        all_phase = []
        all_scenario_id = []
        all_scenario_type = []
        
        generators = {
            ScenarioType.STEADY_STATE: self._generate_steady_state,
            ScenarioType.STEP_RESPONSE: self._generate_step_response,
            ScenarioType.RAMP_SWEEP: self._generate_ramp_sweep,
            ScenarioType.SINUSOIDAL: self._generate_sinusoidal,
            ScenarioType.RANDOM_REALISTIC: self._generate_random_realistic,
            ScenarioType.T_EXT_STEP: self._generate_t_ext_step,
            ScenarioType.T_EXT_RAMP: self._generate_t_ext_ramp,
            ScenarioType.COMBINED_STRESS: self._generate_combined_stress,
        }
        
        prev_end_point = None
        
        for seq_scenario in sequenced_scenarios:
            scenario = seq_scenario.scenario
            
            # Add transition ramp if needed
            if prev_end_point is not None:
                start_point = (
                    scenario.operating_points[0] 
                    if scenario.operating_points 
                    else OperatingPoint(0.5, 298.0, 288.0)
                )
                
                if self.config.transition_ramp_seconds > 0:
                    trans_q, trans_t_air, trans_t_ext, trans_phase = self._generate_transition_ramp(
                        prev_end_point, start_point
                    )
                    all_q_flow.append(trans_q)
                    all_t_air.append(trans_t_air)
                    all_t_ext.append(trans_t_ext)
                    all_phase.append(trans_phase)
                    all_scenario_id.append(np.full(len(trans_q), -1))
                    all_scenario_type.append(np.full(len(trans_q), "transition"))
            
            # Generate scenario data
            generator = generators.get(scenario.scenario_type)
            if generator is None:
                raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")
            
            q_flow, t_air, t_ext, phase = generator(scenario, cdu_rng)
            
            all_q_flow.append(q_flow)
            all_t_air.append(t_air)
            all_t_ext.append(t_ext)
            all_phase.append(phase)
            all_scenario_id.append(np.full(len(q_flow), scenario.scenario_id))
            all_scenario_type.append(np.full(len(q_flow), scenario.scenario_type.value))
            
            if scenario.operating_points:
                prev_end_point = scenario.operating_points[-1]
            else:
                # For dynamic scenarios, estimate end point
                prev_end_point = OperatingPoint(
                    q_flow_fraction=(q_flow[-1] - self.config.q_flow_min_kw * 1000) / 
                                   ((self.config.q_flow_max_kw - self.config.q_flow_min_kw) * 1000),
                    t_air_k=t_air[-1],
                    t_ext_k=t_ext[-1]
                )
        
        # Concatenate
        combined_q_flow = np.concatenate(all_q_flow)
        combined_t_air = np.concatenate(all_t_air)
        combined_t_ext = np.concatenate(all_t_ext)
        
        # Light smoothing on T_air at boundaries only
        if len(combined_t_air) > 20:
            sigma_samples = 3.0  # Very light smoothing
            combined_t_air = gaussian_filter1d(combined_t_air, sigma=sigma_samples)
        
        df = pd.DataFrame({
            'q_flow_w': combined_q_flow,
            't_air_k': combined_t_air,
            't_ext_k': combined_t_ext,
            'phase': np.concatenate(all_phase),
            'scenario_id': np.concatenate(all_scenario_id),
            'scenario_type': np.concatenate(all_scenario_type)
        })
        
        return df
    
    def build_multi_cdu_sequence(
        self,
        sequenced_scenarios: List[SequencedScenario],
        apply_jitter: bool = True
    ) -> pd.DataFrame:
        """Build complete input sequence for all CDUs (shared sequence mode)"""
        base_df = self.build_single_cdu_sequence(sequenced_scenarios)
        n_steps = len(base_df)
        
        data = {
            'timestep': np.arange(n_steps),
            'phase': base_df['phase'].values,
            'scenario_id': base_df['scenario_id'].values,
            'scenario_type': base_df['scenario_type'].values,
            't_ext_k': base_df['t_ext_k'].values,
        }
        
        for cdu_id in range(self.config.n_cdus):
            cdu_name = f'CDU_{cdu_id+1:02d}'
            
            if apply_jitter:
                offset = self.rng.uniform(-self.config.jitter_fraction, self.config.jitter_fraction)
                scale = 1.0 + self.rng.uniform(-self.config.jitter_fraction, self.config.jitter_fraction)
                
                q_flow = base_df['q_flow_w'].values * scale
                t_air = base_df['t_air_k'].values + offset * 3  # ±3K offset
            else:
                q_flow = base_df['q_flow_w'].values.copy()
                t_air = base_df['t_air_k'].values.copy()
            
            data[f'{cdu_name}_q_flow_w'] = q_flow
            data[f'{cdu_name}_t_air_k'] = t_air
        
        return pd.DataFrame(data)
    
    def build_independent_cdu_sequences(
        self,
        all_scenarios: List[ScenarioSpec],
        total_duration_seconds: int,
        sequencer: 'ScenarioSequencer'
    ) -> pd.DataFrame:
        """
        Build independent scenario sequences for each CDU using LHS sampling.
        """
        n_cdus = self.config.n_cdus
        n_scenarios = len(all_scenarios)
        
        # LHS for CDU parameters
        lhs_sampler = qmc.LatinHypercube(d=4, seed=self.config.seed)
        lhs_samples = lhs_sampler.random(n_cdus)
        
        avg_scenario_duration = np.mean([s.duration_seconds for s in all_scenarios])
        scenarios_needed = int(np.ceil(total_duration_seconds / avg_scenario_duration)) + 5
        
        cdu_sequences = {}
        cdu_metadata = {}
        
        for cdu_id in range(n_cdus):
            cdu_name = f'CDU_{cdu_id+1:02d}'
            
            scenario_seed_factor = lhs_samples[cdu_id, 0]
            timing_offset_factor = lhs_samples[cdu_id, 1]
            amplitude_scale_factor = lhs_samples[cdu_id, 2]
            shuffle_seed_factor = lhs_samples[cdu_id, 3]
            
            cdu_seed = self.config.seed + int(scenario_seed_factor * 10000) + cdu_id * 1000
            cdu_rng = np.random.RandomState(cdu_seed)
            
            # Sample scenarios with replacement
            selected_indices = cdu_rng.choice(n_scenarios, size=scenarios_needed, replace=True)
            
            # Fully shuffle
            shuffle_rng = np.random.RandomState(int(shuffle_seed_factor * 10000) + cdu_id * 777)
            shuffle_rng.shuffle(selected_indices)
            
            selected_scenarios = [all_scenarios[i] for i in selected_indices]
            
            # Create random sequence
            sequenced = self._create_random_sequence(selected_scenarios, cdu_rng)
            
            # Trim to duration
            trimmed_sequenced = []
            cumulative_duration = 0
            for seq in sequenced:
                if cumulative_duration >= total_duration_seconds:
                    break
                trimmed_sequenced.append(seq)
                cumulative_duration += seq.scenario.duration_seconds + self.config.transition_ramp_seconds
            
            # Build sequence
            cdu_df = self.build_single_cdu_sequence(trimmed_sequenced, cdu_rng)
            
            timing_offset_steps = int(timing_offset_factor * 0.3 * len(cdu_df))
            amplitude_scale = 0.94 + amplitude_scale_factor * 0.12  # 0.94 to 1.06
            
            cdu_sequences[cdu_name] = {
                'q_flow_w': cdu_df['q_flow_w'].values * amplitude_scale,
                't_air_k': cdu_df['t_air_k'].values,
                't_ext_k': cdu_df['t_ext_k'].values,  # Keep per-CDU T_ext for now
                'phase': cdu_df['phase'].values,
                'scenario_id': cdu_df['scenario_id'].values,
                'scenario_type': cdu_df['scenario_type'].values,
                'timing_offset': timing_offset_steps
            }
            
            cdu_metadata[cdu_name] = {
                'seed': cdu_seed,
                'n_scenarios': len(trimmed_sequenced),
                'timing_offset_steps': timing_offset_steps,
                'amplitude_scale': amplitude_scale,
                'scenario_types': [s.scenario.scenario_type.value for s in trimmed_sequenced]
            }
        
        # Find max length
        max_length = max(len(seq['q_flow_w']) for seq in cdu_sequences.values())
        min_length = total_duration_seconds // self.config.timestep_seconds
        target_length = max(max_length, min_length)
        
        data = {'timestep': np.arange(target_length)}
        
        # IMPORTANT: T_ext is SHARED across all CDUs (single FMU input)
        # Use the FIRST CDU's T_ext as the shared value
        # But blend contributions from different CDU scenarios for variety
        first_cdu = list(cdu_sequences.keys())[0]
        base_t_ext = cdu_sequences[first_cdu]['t_ext_k']
        
        # Pad/trim T_ext
        if len(base_t_ext) < target_length:
            repeats = int(np.ceil(target_length / len(base_t_ext)))
            base_t_ext = np.tile(base_t_ext, repeats)[:target_length]
        else:
            base_t_ext = base_t_ext[:target_length]
        
        data['t_ext_k'] = base_t_ext
        
        phase_columns = {}
        scenario_type_columns = {}
        
        for cdu_name, seq_data in cdu_sequences.items():
            offset = seq_data['timing_offset']
            
            q_flow = seq_data['q_flow_w']
            t_air = seq_data['t_air_k']
            phase = seq_data['phase']
            scenario_type = seq_data['scenario_type']
            
            # Apply circular shift
            if offset > 0 and len(q_flow) > offset:
                q_flow = np.concatenate([q_flow[offset:], q_flow[:offset]])
                t_air = np.concatenate([t_air[offset:], t_air[:offset]])
                phase = np.concatenate([phase[offset:], phase[:offset]])
                scenario_type = np.concatenate([scenario_type[offset:], scenario_type[:offset]])
            
            # Pad/trim to target length
            if len(q_flow) < target_length:
                repeats = int(np.ceil(target_length / len(q_flow)))
                q_flow = np.tile(q_flow, repeats)[:target_length]
                t_air = np.tile(t_air, repeats)[:target_length]
                phase = np.tile(phase, repeats)[:target_length]
                scenario_type = np.tile(scenario_type, repeats)[:target_length]
            else:
                q_flow = q_flow[:target_length]
                t_air = t_air[:target_length]
                phase = phase[:target_length]
                scenario_type = scenario_type[:target_length]
            
            data[f'{cdu_name}_q_flow_w'] = q_flow
            data[f'{cdu_name}_t_air_k'] = t_air
            phase_columns[f'{cdu_name}_phase'] = phase
            scenario_type_columns[f'{cdu_name}_scenario_type'] = scenario_type
        
        data.update(phase_columns)
        data.update(scenario_type_columns)
        
        result_df = pd.DataFrame(data)
        result_df.attrs['cdu_metadata'] = cdu_metadata
        
        return result_df
    
    def _create_random_sequence(
        self,
        scenarios: List[ScenarioSpec],
        rng: np.random.RandomState
    ) -> List[SequencedScenario]:
        """Create a randomly ordered sequence of scenarios."""
        shuffled = list(scenarios)
        rng.shuffle(shuffled)
        
        sequenced = []
        current_time = 0
        initial_point = OperatingPoint(q_flow_fraction=0.50, t_air_k=298.0, t_ext_k=288.0)
        current_point = initial_point
        
        for scenario in shuffled:
            start_point = (
                scenario.operating_points[0] 
                if scenario.operating_points 
                else OperatingPoint(0.5, 298.0, 288.0)
            )
            
            q_diff = abs(current_point.q_flow_fraction - start_point.q_flow_fraction)
            t_air_diff = abs(current_point.t_air_k - start_point.t_air_k) / 50.0
            t_ext_diff = abs(current_point.t_ext_k - start_point.t_ext_k) / 50.0
            cost = 2.0 * q_diff + t_air_diff + t_ext_diff
            
            sequenced.append(SequencedScenario(
                scenario=scenario,
                sequence_index=len(sequenced),
                start_time_seconds=current_time,
                transition_cost=cost
            ))
            
            current_time += self.config.transition_ramp_seconds + scenario.duration_seconds
            
            if scenario.operating_points:
                current_point = scenario.operating_points[-1]
            else:
                current_point = OperatingPoint(0.5, 298.0, 288.0)
        
        return sequenced
    
    def format_for_fmu(self, multi_cdu_df: pd.DataFrame) -> pd.DataFrame:
        """Convert to FMU input format."""
        fmu_data = {}
        
        # External temperature (SHARED across all CDUs)
        fmu_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = \
            multi_cdu_df['t_ext_k'].values
        
        for col in multi_cdu_df.columns:
            if '_q_flow_w' in col:
                cdu_name = col.replace('_q_flow_w', '')
                cdu_num = int(cdu_name.split('_')[1])
                
                fmu_col = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_Q_flow_total'
                fmu_data[fmu_col] = multi_cdu_df[col].values
            
            elif '_t_air_k' in col and '_phase' not in col and '_scenario' not in col:
                cdu_name = col.replace('_t_air_k', '')
                cdu_num = int(cdu_name.split('_')[1])
                
                fmu_col = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_T_Air'
                fmu_data[fmu_col] = multi_cdu_df[col].values
        
        return pd.DataFrame(fmu_data)