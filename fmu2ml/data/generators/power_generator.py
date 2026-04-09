import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .scenario_generator import ScenarioGenerator


class PowerGenerator:
    """Generate power consumption data for CDUs"""
    
    def __init__(self, config: Dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.scenario_generator = ScenarioGenerator(seed=seed)
        np.random.seed(seed)
        
        # Get config parameters
        self.min_power = config.get('MIN_POWER', 10.0)
        self.max_power = config.get('MAX_POWER', 100.0)
        self.normal_load_min = config.get('MIN_NORMAL_LOAD', 0.4)
        self.normal_load_max = config.get('MAX_NORMAL_LOAD', 0.6)
    
    def generate_time_series(
        self,
        scenario_type: str,
        scenario_params: np.ndarray,
        num_timesteps: int = 3600,
        base_power: Optional[float] = None
    ) -> np.ndarray:
        """Generate CDU power time series based on scenario type"""
        
        if scenario_type == "normal":
            return self._generate_normal_scenario(
                scenario_params, num_timesteps, base_power
            )
        elif scenario_type == "edge":
            return self._generate_edge_scenario(
                scenario_params, num_timesteps, base_power
            )
        elif scenario_type == "fault":
            return self._generate_fault_scenario(
                scenario_params, num_timesteps, base_power
            )
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def _generate_normal_scenario(
        self,
        params: np.ndarray,
        num_timesteps: int,
        base_power: Optional[float]
    ) -> np.ndarray:
        """Normal operation: 40-60% utilization with small variations"""
        load_factor, variation_level = params[:2]
        
        # Calculate normal operating range
        normal_min = self.min_power + self.normal_load_min * (self.max_power - self.min_power)
        normal_max = self.min_power + self.normal_load_max * (self.max_power - self.min_power)
        
        if base_power is None or base_power < self.min_power:
            base_power = normal_min + load_factor * (normal_max - normal_min)
        
        # Small variations (1-3% of range)
        max_step = (0.01 + 0.02 * variation_level) * (self.max_power - self.min_power)
        
        power = np.zeros(num_timesteps)
        power[0] = base_power
        
        for t in range(1, num_timesteps):
            step = np.random.uniform(-max_step, max_step)
            power[t] = np.clip(power[t-1] + step, normal_min, normal_max)
        
        return power
    
    def _generate_edge_scenario(
        self,
        params: np.ndarray,
        num_timesteps: int,
        base_power: Optional[float]
    ) -> np.ndarray:
        """Edge cases: near limits or sudden spikes"""
        edge_type, spike_probability, spike_magnitude = params
        
        if base_power is None or base_power < self.min_power:
            if edge_type < 0.33:  # Near idle
                base_power = self.min_power + 0.1 * (self.max_power - self.min_power)
            elif edge_type < 0.66:  # Near peak
                base_power = self.max_power - 0.1 * (self.max_power - self.min_power)
            else:  # Traffic spike scenario
                base_power = self.min_power + 0.5 * (self.max_power - self.min_power)
        
        power = np.zeros(num_timesteps)
        power[0] = base_power
        
        for t in range(1, num_timesteps):
            # Check for spike event
            if np.random.random() < spike_probability * 0.001:
                spike_duration = np.random.randint(30, 300)
                spike_power = min(
                    self.max_power,
                    power[t-1] + spike_magnitude * (self.max_power - self.min_power)
                )
                
                for i in range(min(spike_duration, num_timesteps - t)):
                    if i < 10:  # Ramp up
                        power[t+i] = power[t-1] + (spike_power - power[t-1]) * (i/10)
                    elif i > spike_duration - 10:  # Ramp down
                        power[t+i] = spike_power - (spike_power - power[t-1]) * ((i - spike_duration + 10)/10)
                    else:  # Sustained
                        power[t+i] = spike_power + np.random.uniform(-0.5, 0.5)
            else:
                power[t] = np.clip(
                    power[t-1] + np.random.uniform(-0.5, 0.5),
                    self.min_power,
                    self.max_power
                )
        
        return power
    
    def _generate_fault_scenario(
        self,
        params: np.ndarray,
        num_timesteps: int,
        base_power: Optional[float]
    ) -> np.ndarray:
        """Fault/extreme: component failures or beyond design limits"""
        fault_type, severity, recovery_time = params
        
        power = np.zeros(num_timesteps)
        if base_power is None or base_power < self.min_power:
            base_power = self.min_power + 0.5 * (self.max_power - self.min_power)
        power[0] = base_power
        
        # Fault occurs at random time in first half
        fault_time = np.random.randint(num_timesteps//4, num_timesteps//2)
        
        for t in range(1, fault_time):
            power[t] = power[t-1] + np.random.uniform(-0.5, 0.5)
        
        if fault_type < 0.5:  # Component failure
            fault_power = self.min_power + severity * 0.2 * (self.max_power - self.min_power)
            recovery_duration = int(recovery_time * 600)
            
            # Sudden drop
            for i in range(min(10, num_timesteps - fault_time)):
                power[fault_time + i] = power[fault_time-1] - (power[fault_time-1] - fault_power) * (i/10)
            
            # Stay at fault level
            for t in range(fault_time + 10, min(fault_time + recovery_duration, num_timesteps)):
                power[t] = fault_power + np.random.uniform(-1, 1)
            
            # Recovery
            if fault_time + recovery_duration < num_timesteps:
                for t in range(fault_time + recovery_duration, num_timesteps):
                    power[t] = min(power[t-1] + 0.5, base_power)
        
        else:  # Extreme condition
            extreme_power = self.max_power + severity * 0.2 * (self.max_power - self.min_power)
            
            for i in range(min(60, num_timesteps - fault_time)):
                power[fault_time + i] = power[fault_time-1] + (extreme_power - power[fault_time-1]) * (i/60)
            
            for t in range(fault_time + 60, num_timesteps):
                if t > fault_time + 300:
                    power[t] = max(self.min_power, power[t-1] - 0.1)
                else:
                    power[t] = extreme_power + np.random.uniform(-2, 2)
        
        return power
    
    def generate_continuous_power_data(
        self,
        n_cdus,
        duration_hours: int = 24,
        timestep_seconds: int = 1,
        initial_base_power: Optional[float] = None,
        distribution: List[float] = [0.5, 0.4, 0.1]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Generate continuous power data for multiple hours"""
        
        # Initialize storage
        all_cdu_data = {f'CDU_{i+1:02d}': [] for i in range(n_cdus)}
        last_power_values = {f'CDU_{i+1:02d}': initial_base_power for i in range(n_cdus)}
        gen_scenario = {f'CDU_{i+1:02d}': [] for i in range(n_cdus)}
        
        # Generate data hour by hour
        for hour in range(duration_hours):
            hour_seed = self.seed + hour
            
            # Create a new scenario generator with the hour-specific seed
            hour_scenario_generator = ScenarioGenerator(seed=hour_seed)
            
            # Generate scenarios for this hour
            scenarios = hour_scenario_generator.generate_scenario_lhs(
                n_scenarios=30,
                n_cdus=n_cdus,
                distribution=distribution,
                seed=hour_seed
            )
            
            # Process each CDU's data
            for cdu_id in range(n_cdus):
                cdu_name = f'CDU_{cdu_id+1:02d}'
                cdu_scenarios = [s for s in scenarios if s['cdu_id'] == cdu_id]
                
                if cdu_scenarios:
                    scenario = cdu_scenarios[hour % len(cdu_scenarios)]
                    gen_scenario[cdu_name].append(scenario['type'])
                    
                    power_series = self.generate_time_series(
                        scenario_type=scenario['type'],
                        scenario_params=scenario['params'],
                        num_timesteps=3600 // timestep_seconds,
                        base_power=last_power_values[cdu_name]
                    )
                    
                    all_cdu_data[cdu_name].extend(power_series)
                    last_power_values[cdu_name] = power_series[-1]
                else:
                    gen_scenario[cdu_name].append('normal')
                    params = np.random.random(2)
                    power_series = self.generate_time_series(
                        scenario_type='normal',
                        scenario_params=params,
                        num_timesteps=3600 // timestep_seconds,
                        base_power=last_power_values[cdu_name]
                    )
                    all_cdu_data[cdu_name].extend(power_series)
                    last_power_values[cdu_name] = power_series[-1]
        
        # Create DataFrame and convert kW to W
        df = pd.DataFrame(all_cdu_data) * 1000
        
        return df, gen_scenario


def generate_continuous_power_data(
    n_cdus: int = 49,
    duration_hours: int = 24,
    timestep_seconds: int = 1,
    initial_base_power: Optional[float] = None,
    seed: int = 42,
    distribution: List[float] = [0.5, 0.4, 0.1],
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to generate continuous power data
    
    Parameters:
    -----------
    n_cdus : int
        Number of CDUs
    duration_hours : int
        Duration in hours
    timestep_seconds : int
        Time step in seconds
    initial_base_power : float, optional
        Initial base power level
    seed : int
        Random seed
    distribution : List[float]
        Distribution of [normal, edge, fault] scenarios
    config : Dict, optional
        Configuration dictionary
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict] : Power data and scenario information
    """
    if config is None:
        try:
            from raps.config import ConfigManager
            config = ConfigManager(system_name="marconi100").get_config()
            print("Using Marconi100 configuration for power generation.")
        except ImportError:
            print("Marconi100 configuration not found. Using default parameters.")
            config = {
                'MIN_POWER': 12.3652,
                'MAX_POWER': 38.5552,
                'MIN_NORMAL_LOAD': 0.4,
                'MAX_NORMAL_LOAD': 0.6
            }
    
    generator = PowerGenerator(config=config, seed=seed)
    return generator.generate_continuous_power_data(
        n_cdus=n_cdus,
        duration_hours=duration_hours,
        timestep_seconds=timestep_seconds,
        initial_base_power=initial_base_power,
        distribution=distribution
    )