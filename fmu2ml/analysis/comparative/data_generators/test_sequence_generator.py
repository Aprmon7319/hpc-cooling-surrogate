"""
Test Sequence Generators for Dynamic Response Analysis.

Provides specialized input generators for step response, ramp response,
and frequency sweep analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class StepInputGenerator:
    """Generates step input sequences for transient response analysis."""
    
    def __init__(
        self,
        system_name: str,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.system_name = system_name
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', 1)
        
        self.input_ranges = input_ranges or {
            'Q_flow': (12.0, 40.0),
            'T_Air': (288.15, 308.15),
            'T_ext': (283.15, 313.15)
        }
    
    def generate_step_sequence(
        self,
        target_input: str,
        step_from: float,
        step_to: float,
        pre_step_duration: int = 600,
        post_step_duration: int = 1200,
        other_inputs: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate a step input sequence.
        
        Args:
            target_input: Which input to step (Q_flow, T_Air, or T_ext)
            step_from: Initial value
            step_to: Final value after step
            pre_step_duration: Duration before step (seconds)
            post_step_duration: Duration after step (seconds)
            other_inputs: Fixed values for other inputs
            
        Returns:
            DataFrame with step input sequence
        """
        total_duration = pre_step_duration + post_step_duration
        
        # Set defaults for other inputs
        if other_inputs is None:
            other_inputs = {}
        
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var not in other_inputs and var != target_input:
                other_inputs[var] = (self.input_ranges[var][0] + self.input_ranges[var][1]) / 2
        
        # Create step sequence
        values = {}
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var == target_input:
                seq = np.concatenate([
                    np.full(pre_step_duration, step_from),
                    np.full(post_step_duration, step_to)
                ])
            else:
                seq = np.full(total_duration, other_inputs[var])
            values[var] = seq
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            # Q_flow
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            df_data[col] = values['Q_flow']
            
            # T_Air
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
            df_data[col] = values['T_Air']
        
        # T_ext (shared)
        df_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = values['T_ext']
        
        return pd.DataFrame(df_data)
    
    def generate_multi_step_sequence(
        self,
        target_input: str,
        step_values: List[float],
        step_duration: int = 600
    ) -> pd.DataFrame:
        """Generate a sequence with multiple step changes."""
        sequences = []
        
        for i in range(len(step_values) - 1):
            seq = self.generate_step_sequence(
                target_input=target_input,
                step_from=step_values[i],
                step_to=step_values[i + 1],
                pre_step_duration=step_duration if i == 0 else 0,
                post_step_duration=step_duration
            )
            sequences.append(seq)
        
        return pd.concat(sequences, ignore_index=True)


class RampInputGenerator:
    """Generates ramp input sequences for rate sensitivity analysis."""
    
    def __init__(
        self,
        system_name: str,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.system_name = system_name
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', 1)
        
        self.input_ranges = input_ranges or {
            'Q_flow': (12.0, 40.0),
            'T_Air': (288.15, 308.15),
            'T_ext': (283.15, 313.15)
        }
    
    def generate_ramp_sequence(
        self,
        target_input: str,
        start_value: float,
        end_value: float,
        ramp_duration: int = 600,
        hold_duration: int = 300,
        other_inputs: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate a ramp input sequence.
        
        Args:
            target_input: Which input to ramp
            start_value: Starting value
            end_value: Ending value
            ramp_duration: Duration of ramp (seconds)
            hold_duration: Hold duration before and after ramp
            other_inputs: Fixed values for other inputs
            
        Returns:
            DataFrame with ramp input sequence
        """
        total_duration = 2 * hold_duration + ramp_duration
        
        if other_inputs is None:
            other_inputs = {}
        
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var not in other_inputs and var != target_input:
                other_inputs[var] = (self.input_ranges[var][0] + self.input_ranges[var][1]) / 2
        
        # Create ramp sequence
        values = {}
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var == target_input:
                hold_before = np.full(hold_duration, start_value)
                ramp = np.linspace(start_value, end_value, ramp_duration)
                hold_after = np.full(hold_duration, end_value)
                seq = np.concatenate([hold_before, ramp, hold_after])
            else:
                seq = np.full(total_duration, other_inputs[var])
            values[var] = seq
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            df_data[col] = values['Q_flow']
            
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
            df_data[col] = values['T_Air']
        
        df_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = values['T_ext']
        
        return pd.DataFrame(df_data)


class FrequencySweepGenerator:
    """Generates sinusoidal inputs for frequency response analysis."""
    
    def __init__(
        self,
        system_name: str,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.system_name = system_name
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', 1)
        
        self.input_ranges = input_ranges or {
            'Q_flow': (12.0, 40.0),
            'T_Air': (288.15, 308.15),
            'T_ext': (283.15, 313.15)
        }
    
    def generate_single_frequency(
        self,
        target_input: str,
        frequency: float,
        amplitude: float,
        duration: int = 3600,
        other_inputs: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate sinusoidal input at a single frequency.
        
        Args:
            target_input: Which input to oscillate
            frequency: Frequency in Hz
            amplitude: Oscillation amplitude (absolute)
            duration: Duration in seconds
            other_inputs: Fixed values for other inputs
            
        Returns:
            DataFrame with sinusoidal input
        """
        if other_inputs is None:
            other_inputs = {}
        
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var not in other_inputs and var != target_input:
                other_inputs[var] = (self.input_ranges[var][0] + self.input_ranges[var][1]) / 2
        
        t = np.arange(duration)
        center = (self.input_ranges[target_input][0] + self.input_ranges[target_input][1]) / 2
        
        values = {}
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var == target_input:
                signal = center + amplitude * np.sin(2 * np.pi * frequency * t)
                signal = np.clip(signal, self.input_ranges[var][0], self.input_ranges[var][1])
                values[var] = signal
            else:
                values[var] = np.full(duration, other_inputs[var])
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            df_data[col] = values['Q_flow']
            
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
            df_data[col] = values['T_Air']
        
        df_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = values['T_ext']
        
        return pd.DataFrame(df_data)
    
    def generate_frequency_sweep(
        self,
        target_input: str,
        frequencies: List[float],
        amplitude: float,
        cycles_per_frequency: int = 5,
        other_inputs: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int, float]]]:
        """
        Generate a frequency sweep with multiple frequencies.
        
        Returns:
            Tuple of (DataFrame, list of (start_idx, end_idx, frequency))
        """
        sequences = []
        freq_segments = []
        current_idx = 0
        
        for freq in frequencies:
            period = int(1.0 / freq) if freq > 0 else 1000
            duration = cycles_per_frequency * period
            
            seq = self.generate_single_frequency(
                target_input=target_input,
                frequency=freq,
                amplitude=amplitude,
                duration=duration,
                other_inputs=other_inputs
            )
            
            freq_segments.append((current_idx, current_idx + duration, freq))
            current_idx += duration
            sequences.append(seq)
        
        return pd.concat(sequences, ignore_index=True), freq_segments


class GridInputGenerator:
    """Generates grid-based inputs for response surface mapping."""
    
    def __init__(
        self,
        system_name: str,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.system_name = system_name
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', 1)
        
        self.input_ranges = input_ranges or {
            'Q_flow': (12.0, 40.0),
            'T_Air': (288.15, 308.15),
            'T_ext': (283.15, 313.15)
        }
    
    def generate_2d_grid(
        self,
        var1: str,
        var2: str,
        n_points: int = 10,
        fixed_values: Optional[Dict[str, float]] = None,
        transition_steps: int = 60
    ) -> pd.DataFrame:
        """
        Generate 2D grid with smooth transitions between points.
        
        Args:
            var1: First variable to vary
            var2: Second variable to vary
            n_points: Number of points per dimension
            fixed_values: Fixed values for other variables
            transition_steps: Steps for transition between grid points
            
        Returns:
            DataFrame with grid inputs
        """
        if fixed_values is None:
            fixed_values = {}
        
        # Set defaults for fixed variables
        for var in ['Q_flow', 'T_Air', 'T_ext']:
            if var not in [var1, var2] and var not in fixed_values:
                fixed_values[var] = (self.input_ranges[var][0] + self.input_ranges[var][1]) / 2
        
        # Create grid values
        vals1 = np.linspace(self.input_ranges[var1][0], self.input_ranges[var1][1], n_points)
        vals2 = np.linspace(self.input_ranges[var2][0], self.input_ranges[var2][1], n_points)
        
        # Create sequence visiting all grid points with transitions
        all_values = {var: [] for var in ['Q_flow', 'T_Air', 'T_ext']}
        
        prev_v1, prev_v2 = vals1[0], vals2[0]
        
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                # Transition from previous point
                if i > 0 or j > 0:
                    for var in ['Q_flow', 'T_Air', 'T_ext']:
                        if var == var1:
                            trans = np.linspace(prev_v1, v1, transition_steps)
                        elif var == var2:
                            trans = np.linspace(prev_v2, v2, transition_steps)
                        else:
                            trans = np.full(transition_steps, fixed_values[var])
                        all_values[var].extend(trans)
                
                # Hold at current point
                hold_steps = transition_steps
                for var in ['Q_flow', 'T_Air', 'T_ext']:
                    if var == var1:
                        all_values[var].extend(np.full(hold_steps, v1))
                    elif var == var2:
                        all_values[var].extend(np.full(hold_steps, v2))
                    else:
                        all_values[var].extend(np.full(hold_steps, fixed_values[var]))
                
                prev_v1, prev_v2 = v1, v2
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
            df_data[col] = all_values['Q_flow']
            
            col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
            df_data[col] = all_values['T_Air']
        
        df_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = all_values['T_ext']
        
        return pd.DataFrame(df_data)
