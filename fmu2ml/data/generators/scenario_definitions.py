"""
Scenario definitions for systematic FMU input generation.
Defines the state space coverage requirements.

Optimized for DeepONet training with emphasis on dynamic scenarios.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import itertools


class ScenarioType(Enum):
    STEADY_STATE = "steady_state"
    STEP_RESPONSE = "step_response"
    RAMP_SWEEP = "ramp_sweep"
    SINUSOIDAL = "sinusoidal"
    RANDOM_REALISTIC = "random_realistic"
    T_EXT_STEP = "t_ext_step"          # NEW: External temperature steps
    T_EXT_RAMP = "t_ext_ramp"          # NEW: External temperature ramps
    COMBINED_STRESS = "combined_stress" # NEW: Simultaneous Q_flow + T_ext changes


class Phase(Enum):
    """Labels for data segments within a scenario"""
    TRANSITION = "transition"      # Ramping to new setpoint
    SETTLING = "settling"          # Controller responding
    STEADY_STATE = "steady_state"  # Equilibrium reached


@dataclass
class OperatingPoint:
    """Single operating point definition"""
    q_flow_fraction: float      # 0.0 to 1.0 (fraction of max)
    t_air_k: float              # Kelvin
    t_ext_k: float              # Kelvin
    
    def to_dict(self) -> Dict:
        return {
            'q_flow_fraction': self.q_flow_fraction,
            't_air_k': self.t_air_k,
            't_ext_k': self.t_ext_k
        }


@dataclass
class ScenarioSpec:
    """Specification for a single scenario"""
    scenario_type: ScenarioType
    scenario_id: int
    duration_seconds: int
    operating_points: List[OperatingPoint]
    transition_times: List[int] = field(default_factory=list)  # When transitions occur
    ramp_duration_seconds: int = 60  # Duration of ramps between points
    description: str = ""
    
    # Dynamic scenario parameters (set by generators)
    sinusoid_params: Optional[Dict] = field(default=None, repr=False)
    diurnal_params: Optional[Dict] = field(default=None, repr=False)
    ou_params: Optional[Dict] = field(default=None, repr=False)
    combined_params: Optional[Dict] = field(default=None, repr=False)
    
    @property
    def total_duration(self) -> int:
        return self.duration_seconds


class SteadyStateGridGenerator:
    """
    Generate steady-state grid scenarios.
    
    REDUCED for DeepONet: Sample grid instead of full factorial.
    Full factorial = 125 points, we use ~50 LHS-sampled points.
    """
    
    # Default grid values
    Q_FLOW_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]  # Fraction of max
    T_AIR_LEVELS = [293.0, 298.0, 303.0, 308.0, 313.0]  # Kelvin (20-40°C)
    T_EXT_LEVELS = [268.0, 278.0, 288.0, 298.0, 308.0]  # Kelvin (-5 to 35°C)
    
    DURATION_PER_POINT = 600  # 10 minutes per steady-state point
    
    def __init__(
        self,
        q_flow_levels: Optional[List[float]] = None,
        t_air_levels: Optional[List[float]] = None,
        t_ext_levels: Optional[List[float]] = None,
        duration_per_point: int = 600,
        use_lhs_sampling: bool = True,  # NEW: Use LHS instead of full factorial
        n_samples: int = 50,            # NEW: Number of LHS samples
        seed: int = 42
    ):
        self.q_flow_levels = q_flow_levels or self.Q_FLOW_LEVELS
        self.t_air_levels = t_air_levels or self.T_AIR_LEVELS
        self.t_ext_levels = t_ext_levels or self.T_EXT_LEVELS
        self.duration_per_point = duration_per_point
        self.use_lhs_sampling = use_lhs_sampling
        self.n_samples = n_samples
        self.seed = seed
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate steady-state scenarios"""
        scenarios = []
        
        if self.use_lhs_sampling:
            # LHS sampling for better coverage with fewer points
            from scipy.stats import qmc
            
            sampler = qmc.LatinHypercube(d=3, seed=self.seed)
            samples = sampler.random(self.n_samples)
            
            # Scale to ranges
            q_min, q_max = min(self.q_flow_levels), max(self.q_flow_levels)
            t_air_min, t_air_max = min(self.t_air_levels), max(self.t_air_levels)
            t_ext_min, t_ext_max = min(self.t_ext_levels), max(self.t_ext_levels)
            
            for idx, sample in enumerate(samples):
                q = q_min + sample[0] * (q_max - q_min)
                t_air = t_air_min + sample[1] * (t_air_max - t_air_min)
                t_ext = t_ext_min + sample[2] * (t_ext_max - t_ext_min)
                
                op_point = OperatingPoint(
                    q_flow_fraction=q,
                    t_air_k=t_air,
                    t_ext_k=t_ext
                )
                
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.STEADY_STATE,
                    scenario_id=idx,
                    duration_seconds=self.duration_per_point,
                    operating_points=[op_point],
                    description=f"SS: Q={q:.0%}, T_air={t_air:.0f}K, T_ext={t_ext:.0f}K"
                )
                scenarios.append(scenario)
        else:
            # Full factorial (original behavior)
            combinations = list(itertools.product(
                self.q_flow_levels,
                self.t_air_levels,
                self.t_ext_levels
            ))
            
            for idx, (q, t_air, t_ext) in enumerate(combinations):
                op_point = OperatingPoint(
                    q_flow_fraction=q,
                    t_air_k=t_air,
                    t_ext_k=t_ext
                )
                
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.STEADY_STATE,
                    scenario_id=idx,
                    duration_seconds=self.duration_per_point,
                    operating_points=[op_point],
                    description=f"SS: Q={q:.0%}, T_air={t_air:.0f}K, T_ext={t_ext:.0f}K"
                )
                scenarios.append(scenario)
        
        return scenarios


class StepResponseGenerator:
    """Generate step response scenarios for Q_flow"""
    
    # More step sizes for better coverage
    STEP_SIZES = [0.10, 0.15, 0.25, 0.30, 0.40, 0.50, 
                  -0.10, -0.15, -0.25, -0.30, -0.40, -0.50]
    BASELINE_LEVELS = [0.20, 0.35, 0.50, 0.65, 0.80]
    
    # Timing
    HOLD_BEFORE = 180    # 3 min at baseline (reduced from 5)
    STEP_OBSERVE = 300   # 5 min after step
    RECOVERY = 180       # 3 min recovery
    
    def __init__(
        self,
        step_sizes: Optional[List[float]] = None,
        baseline_levels: Optional[List[float]] = None,
        ramp_duration: int = 15,  # Faster ramps (15s instead of 30s)
        t_ext_values: Optional[List[float]] = None,  # NEW: Vary T_ext in steps
        seed: int = 42
    ):
        self.step_sizes = step_sizes or self.STEP_SIZES
        self.baseline_levels = baseline_levels or self.BASELINE_LEVELS
        self.ramp_duration = ramp_duration
        self.t_ext_values = t_ext_values or [273.0, 288.0, 303.0]  # Cold, mild, hot
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate step response scenarios"""
        scenarios = []
        scenario_id = 0
        
        # Q_flow steps at different T_ext conditions
        for t_ext in self.t_ext_values:
            for baseline in self.baseline_levels:
                for step_size in self.step_sizes:
                    target = baseline + step_size
                    
                    # Skip invalid targets
                    if target < 0.05 or target > 1.0:
                        continue
                    
                    # Randomize T_air slightly for each scenario
                    t_air = 298.0 + self.rng.uniform(-3, 3)
                    
                    op_baseline = OperatingPoint(
                        q_flow_fraction=baseline,
                        t_air_k=t_air,
                        t_ext_k=t_ext
                    )
                    
                    op_target = OperatingPoint(
                        q_flow_fraction=target,
                        t_air_k=t_air + self.rng.uniform(-1, 1),  # Slight T_air drift
                        t_ext_k=t_ext
                    )
                    
                    scenario = ScenarioSpec(
                        scenario_type=ScenarioType.STEP_RESPONSE,
                        scenario_id=scenario_id,
                        duration_seconds=self.HOLD_BEFORE + self.STEP_OBSERVE + self.RECOVERY,
                        operating_points=[op_baseline, op_target, op_baseline],
                        transition_times=[self.HOLD_BEFORE, self.HOLD_BEFORE + self.STEP_OBSERVE],
                        ramp_duration_seconds=self.ramp_duration,
                        description=f"Step Q: {baseline:.0%}→{target:.0%} @ T_ext={t_ext:.0f}K"
                    )
                    scenarios.append(scenario)
                    scenario_id += 1
        
        return scenarios


class TExtStepGenerator:
    """NEW: Generate external temperature step scenarios"""
    
    # T_ext step configurations
    T_EXT_STEPS = [
        (288.0, 268.0),  # Mild to cold (-20K)
        (288.0, 278.0),  # Mild to cool (-10K)
        (288.0, 298.0),  # Mild to warm (+10K)
        (288.0, 308.0),  # Mild to hot (+20K)
        (273.0, 293.0),  # Cold to mild (+20K)
        (303.0, 283.0),  # Hot to cool (-20K)
    ]
    
    HOLD_BEFORE = 180    # 3 min at baseline
    STEP_OBSERVE = 420   # 7 min after step (T_ext effects are slower)
    RECOVERY = 300       # 5 min recovery
    
    def __init__(
        self,
        t_ext_steps: Optional[List[Tuple[float, float]]] = None,
        q_flow_levels: Optional[List[float]] = None,
        ramp_duration: int = 30,  # T_ext changes can be faster (weather fronts)
        seed: int = 42
    ):
        self.t_ext_steps = t_ext_steps or self.T_EXT_STEPS
        self.q_flow_levels = q_flow_levels or [0.30, 0.50, 0.70]
        self.ramp_duration = ramp_duration
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate T_ext step scenarios"""
        scenarios = []
        scenario_id = 0
        
        for q_flow in self.q_flow_levels:
            for t_ext_base, t_ext_target in self.t_ext_steps:
                t_air = 298.0 + self.rng.uniform(-2, 2)
                
                op_baseline = OperatingPoint(
                    q_flow_fraction=q_flow,
                    t_air_k=t_air,
                    t_ext_k=t_ext_base
                )
                op_target = OperatingPoint(
                    q_flow_fraction=q_flow,
                    t_air_k=t_air,
                    t_ext_k=t_ext_target
                )
                op_return = OperatingPoint(
                    q_flow_fraction=q_flow,
                    t_air_k=t_air,
                    t_ext_k=t_ext_base
                )
                
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.T_EXT_STEP,
                    scenario_id=scenario_id,
                    duration_seconds=self.HOLD_BEFORE + self.STEP_OBSERVE + self.RECOVERY,
                    operating_points=[op_baseline, op_target, op_return],
                    transition_times=[self.HOLD_BEFORE, self.HOLD_BEFORE + self.STEP_OBSERVE],
                    ramp_duration_seconds=self.ramp_duration,
                    description=f"T_ext Step: {t_ext_base:.0f}K→{t_ext_target:.0f}K @ Q={q_flow:.0%}"
                )
                scenarios.append(scenario)
                scenario_id += 1
        
        return scenarios


class TExtRampGenerator:
    """NEW: Generate external temperature ramp scenarios"""
    
    # Ramp rates in K/min
    RAMP_RATES = [0.5, 1.0, 2.0]  # Slow, medium, fast weather changes
    
    def __init__(
        self,
        ramp_rates: Optional[List[float]] = None,
        t_ext_range: Tuple[float, float] = (273.0, 308.0),
        q_flow_levels: Optional[List[float]] = None,
        seed: int = 42
    ):
        self.ramp_rates = ramp_rates or self.RAMP_RATES
        self.t_ext_range = t_ext_range
        self.q_flow_levels = q_flow_levels or [0.40, 0.60]
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate T_ext ramp scenarios"""
        scenarios = []
        scenario_id = 0
        
        t_ext_low, t_ext_high = self.t_ext_range
        t_ext_span = t_ext_high - t_ext_low
        
        for q_flow in self.q_flow_levels:
            for rate in self.ramp_rates:
                # Calculate ramp duration
                ramp_time_minutes = t_ext_span / rate
                ramp_time_seconds = int(ramp_time_minutes * 60)
                
                hold_time = 120  # 2 minutes hold at each end
                total_duration = hold_time + ramp_time_seconds + hold_time
                
                t_air = 298.0 + self.rng.uniform(-2, 2)
                
                # Warming ramp
                op_start = OperatingPoint(q_flow_fraction=q_flow, t_air_k=t_air, t_ext_k=t_ext_low)
                op_end = OperatingPoint(q_flow_fraction=q_flow, t_air_k=t_air, t_ext_k=t_ext_high)
                
                scenario_up = ScenarioSpec(
                    scenario_type=ScenarioType.T_EXT_RAMP,
                    scenario_id=scenario_id,
                    duration_seconds=total_duration,
                    operating_points=[op_start, op_end],
                    transition_times=[hold_time],
                    ramp_duration_seconds=ramp_time_seconds,
                    description=f"T_ext Ramp Up: {t_ext_low:.0f}K→{t_ext_high:.0f}K @ {rate:.1f}K/min"
                )
                scenarios.append(scenario_up)
                scenario_id += 1
                
                # Cooling ramp
                scenario_down = ScenarioSpec(
                    scenario_type=ScenarioType.T_EXT_RAMP,
                    scenario_id=scenario_id,
                    duration_seconds=total_duration,
                    operating_points=[op_end, op_start],
                    transition_times=[hold_time],
                    ramp_duration_seconds=ramp_time_seconds,
                    description=f"T_ext Ramp Down: {t_ext_high:.0f}K→{t_ext_low:.0f}K @ {rate:.1f}K/min"
                )
                scenarios.append(scenario_down)
                scenario_id += 1
        
        return scenarios


class RampSweepGenerator:
    """Generate Q_flow ramp sweep scenarios"""
    
    # More ramp rates for better coverage
    RAMP_RATES = [0.01, 0.02, 0.05, 0.08, 0.10]  # 1% to 10% per minute
    
    def __init__(
        self,
        ramp_rates: Optional[List[float]] = None,
        q_flow_range: Tuple[float, float] = (0.15, 0.90),
        t_ext_values: Optional[List[float]] = None,
        seed: int = 42
    ):
        self.ramp_rates = ramp_rates or self.RAMP_RATES
        self.q_flow_range = q_flow_range
        self.t_ext_values = t_ext_values or [278.0, 288.0, 298.0]
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate ramp scenarios (up and down for each rate)"""
        scenarios = []
        scenario_id = 0
        
        q_low, q_high = self.q_flow_range
        q_range = q_high - q_low
        
        for t_ext in self.t_ext_values:
            for rate in self.ramp_rates:
                # Calculate ramp duration
                ramp_time_minutes = q_range / rate
                ramp_time_seconds = int(ramp_time_minutes * 60)
                
                # Add hold periods at each end
                hold_time = 90  # 1.5 minutes (reduced)
                total_duration = hold_time + ramp_time_seconds + hold_time
                
                t_air = 298.0 + self.rng.uniform(-2, 2)
                
                # Up ramp
                op_start = OperatingPoint(q_flow_fraction=q_low, t_air_k=t_air, t_ext_k=t_ext)
                op_end = OperatingPoint(q_flow_fraction=q_high, t_air_k=t_air, t_ext_k=t_ext)
                
                scenario_up = ScenarioSpec(
                    scenario_type=ScenarioType.RAMP_SWEEP,
                    scenario_id=scenario_id,
                    duration_seconds=total_duration,
                    operating_points=[op_start, op_end],
                    transition_times=[hold_time],
                    ramp_duration_seconds=ramp_time_seconds,
                    description=f"Ramp Up: {q_low:.0%}→{q_high:.0%} @ {rate:.0%}/min, T_ext={t_ext:.0f}K"
                )
                scenarios.append(scenario_up)
                scenario_id += 1
                
                # Down ramp
                scenario_down = ScenarioSpec(
                    scenario_type=ScenarioType.RAMP_SWEEP,
                    scenario_id=scenario_id,
                    duration_seconds=total_duration,
                    operating_points=[op_end, op_start],
                    transition_times=[hold_time],
                    ramp_duration_seconds=ramp_time_seconds,
                    description=f"Ramp Down: {q_high:.0%}→{q_low:.0%} @ {rate:.0%}/min, T_ext={t_ext:.0f}K"
                )
                scenarios.append(scenario_down)
                scenario_id += 1
        
        return scenarios


class CombinedStressGenerator:
    """
    NEW: Generate combined stress scenarios where multiple inputs change simultaneously.
    
    This is crucial for DeepONet to learn interaction effects.
    """
    
    def __init__(
        self,
        n_scenarios: int = 30,
        duration_per_scenario: int = 900,  # 15 minutes each
        seed: int = 42
    ):
        self.n_scenarios = n_scenarios
        self.duration_per_scenario = duration_per_scenario
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate combined stress scenarios"""
        scenarios = []
        
        combined_types = [
            'summer_afternoon',     # Q_flow up + T_ext up (hot day, high load)
            'winter_morning',       # Q_flow up + T_ext down (cold start, warming up)
            'cooling_maintenance',  # Q_flow down + T_ext up (reduce load in heat)
            'night_cooldown',       # Q_flow down + T_ext down (overnight)
            'random_correlated',    # Random but correlated changes
            'random_anticorrelated' # Random anticorrelated changes
        ]
        
        scenarios_per_type = self.n_scenarios // len(combined_types)
        
        for scenario_id in range(self.n_scenarios):
            combined_type = combined_types[scenario_id % len(combined_types)]
            
            scenario = ScenarioSpec(
                scenario_type=ScenarioType.COMBINED_STRESS,
                scenario_id=scenario_id,
                duration_seconds=self.duration_per_scenario,
                operating_points=[],  # Will be generated dynamically
                description=f"Combined: {combined_type} #{scenario_id}"
            )
            
            # Store parameters for later generation
            scenario.combined_params = {
                'type': combined_type,
                'seed': self.seed + scenario_id,
                'q_flow_start': 0.3 + self.rng.uniform(0, 0.4),
                't_ext_start': 278.0 + self.rng.uniform(0, 20),
            }
            scenarios.append(scenario)
        
        return scenarios


class SinusoidalGenerator:
    """Generate sinusoidal/diurnal pattern scenarios"""
    
    PERIODS_MINUTES = [10, 20, 30, 45, 60]  # More variety
    
    def __init__(
        self,
        periods_minutes: Optional[List[int]] = None,
        q_flow_mean: float = 0.50,
        q_flow_amplitude: float = 0.25,
        include_diurnal: bool = True,
        include_t_ext_sinusoid: bool = True,  # NEW
        seed: int = 42
    ):
        self.periods_minutes = periods_minutes or self.PERIODS_MINUTES
        self.q_flow_mean = q_flow_mean
        self.q_flow_amplitude = q_flow_amplitude
        self.include_diurnal = include_diurnal
        self.include_t_ext_sinusoid = include_t_ext_sinusoid
        self.seed = seed
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate sinusoidal scenarios"""
        scenarios = []
        scenario_id = 0
        
        # Q_flow sinusoids at different T_ext
        t_ext_values = [278.0, 288.0, 298.0]
        
        for t_ext in t_ext_values:
            for period_min in self.periods_minutes:
                duration = period_min * 60 * 2  # Two full periods
                
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.SINUSOIDAL,
                    scenario_id=scenario_id,
                    duration_seconds=duration,
                    operating_points=[],
                    description=f"Sinusoid Q: T={period_min}min, T_ext={t_ext:.0f}K"
                )
                scenario.sinusoid_params = {
                    'mean': self.q_flow_mean,
                    'amplitude': self.q_flow_amplitude,
                    'period_seconds': period_min * 60,
                    't_ext_base': t_ext
                }
                scenarios.append(scenario)
                scenario_id += 1
        
        # Diurnal T_ext pattern with varying Q_flow
        if self.include_diurnal:
            for q_flow_base in [0.35, 0.50, 0.65]:
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.SINUSOIDAL,
                    scenario_id=scenario_id,
                    duration_seconds=3600,  # 1 hour compressed diurnal
                    operating_points=[],
                    description=f"Diurnal T_ext @ Q={q_flow_base:.0%}"
                )
                scenario.diurnal_params = {
                    't_ext_min': 278.0,
                    't_ext_max': 303.0,
                    'period_seconds': 3600,
                    'q_flow_base': q_flow_base
                }
                scenarios.append(scenario)
                scenario_id += 1
        
        # NEW: T_ext sinusoidal (weather oscillations)
        if self.include_t_ext_sinusoid:
            for period_min in [30, 60]:
                scenario = ScenarioSpec(
                    scenario_type=ScenarioType.SINUSOIDAL,
                    scenario_id=scenario_id,
                    duration_seconds=period_min * 60 * 2,
                    operating_points=[],
                    description=f"Sinusoid T_ext: T={period_min}min"
                )
                scenario.sinusoid_params = {
                    'variable': 't_ext',
                    'mean': 288.0,
                    'amplitude': 10.0,  # ±10K
                    'period_seconds': period_min * 60,
                    'q_flow_base': 0.50
                }
                scenarios.append(scenario)
                scenario_id += 1
        
        return scenarios


class RandomRealisticGenerator:
    """Generate random realistic scenarios using Ornstein-Uhlenbeck process"""
    
    def __init__(
        self,
        n_scenarios: int = 30,  # Increased from 20
        duration_per_scenario: int = 2400,  # 40 min each (reduced from 1 hour)
        mean_reversion_strength: float = 0.02,  # Faster reversion
        volatility: float = 0.002,  # More volatility
        jump_probability: float = 0.0002,  # More jumps
        include_t_ext_variation: bool = True,  # NEW
        seed: int = 42
    ):
        self.n_scenarios = n_scenarios
        self.duration_per_scenario = duration_per_scenario
        self.mean_reversion_strength = mean_reversion_strength
        self.volatility = volatility
        self.jump_probability = jump_probability
        self.include_t_ext_variation = include_t_ext_variation
        self.seed = seed
    
    def generate(self) -> List[ScenarioSpec]:
        """Generate random realistic scenarios"""
        scenarios = []
        
        for scenario_id in range(self.n_scenarios):
            scenario = ScenarioSpec(
                scenario_type=ScenarioType.RANDOM_REALISTIC,
                scenario_id=scenario_id,
                duration_seconds=self.duration_per_scenario,
                operating_points=[],
                description=f"Random realistic #{scenario_id}"
            )
            
            # Randomize parameters per scenario for diversity
            rng = np.random.RandomState(self.seed + scenario_id)
            
            scenario.ou_params = {
                'mean': 0.35 + rng.uniform(0, 0.30),  # Mean between 35% and 65%
                'mean_reversion': self.mean_reversion_strength * (0.5 + rng.uniform(0, 1)),
                'volatility': self.volatility * (0.5 + rng.uniform(0, 1)),
                'jump_prob': self.jump_probability * (0.5 + rng.uniform(0, 1)),
                'seed': self.seed + scenario_id,
                't_ext_base': 278.0 + rng.uniform(0, 25),  # Random T_ext base
                't_ext_variation': self.include_t_ext_variation,
                't_ext_volatility': 0.01 if self.include_t_ext_variation else 0  # Slow T_ext drift
            }
            scenarios.append(scenario)
        
        return scenarios


def generate_all_scenarios(
    steady_state_duration: int = 600,
    step_ramp_duration: int = 15,  # Faster transitions
    include_random: bool = True,
    use_lhs_steady_state: bool = True,  # NEW: LHS for steady-state
    n_steady_state: int = 50,           # NEW: Fewer steady-state points
    seed: int = 42
) -> List[ScenarioSpec]:
    """
    Generate complete scenario set optimized for DeepONet training.
    
    Scenario distribution:
    - Steady-state: ~50 (LHS sampled, ~8 hours)
    - Q_flow steps: ~80 (various sizes/conditions, ~9 hours)
    - T_ext steps: ~18 (weather changes, ~3 hours)
    - T_ext ramps: ~12 (gradual weather, ~3 hours)
    - Q_flow ramps: ~30 (various rates, ~8 hours)
    - Combined stress: ~30 (multi-input, ~7.5 hours)
    - Sinusoidal: ~20 (periodic, ~6 hours)
    - Random realistic: ~30 (OU process, ~20 hours)
    
    Total: ~270 scenarios, ~65 hours
    """
    all_scenarios = []
    
    # 1. Steady-State Grid (reduced, LHS sampled)
    ss_gen = SteadyStateGridGenerator(
        duration_per_point=steady_state_duration,
        use_lhs_sampling=use_lhs_steady_state,
        n_samples=n_steady_state,
        seed=seed
    )
    all_scenarios.extend(ss_gen.generate())
    
    # 2. Q_flow Step Responses (expanded)
    step_gen = StepResponseGenerator(
        ramp_duration=step_ramp_duration,
        seed=seed
    )
    all_scenarios.extend(step_gen.generate())
    
    # 3. T_ext Step Responses (NEW)
    t_ext_step_gen = TExtStepGenerator(seed=seed)
    all_scenarios.extend(t_ext_step_gen.generate())
    
    # 4. T_ext Ramp Scenarios (NEW)
    t_ext_ramp_gen = TExtRampGenerator(seed=seed)
    all_scenarios.extend(t_ext_ramp_gen.generate())
    
    # 5. Q_flow Ramp Sweeps (expanded)
    ramp_gen = RampSweepGenerator(seed=seed)
    all_scenarios.extend(ramp_gen.generate())
    
    # 6. Combined Stress Scenarios (NEW)
    combined_gen = CombinedStressGenerator(seed=seed)
    all_scenarios.extend(combined_gen.generate())
    
    # 7. Sinusoidal/Diurnal (expanded)
    sin_gen = SinusoidalGenerator(seed=seed)
    all_scenarios.extend(sin_gen.generate())
    
    # 8. Random Realistic (expanded)
    if include_random:
        rand_gen = RandomRealisticGenerator(seed=seed)
        all_scenarios.extend(rand_gen.generate())
    
    return all_scenarios