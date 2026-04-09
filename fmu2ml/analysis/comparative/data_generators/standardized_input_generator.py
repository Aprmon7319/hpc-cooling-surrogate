"""
Standardized Input Generator for CDU-Level Comparative Analysis.

Generates identical input sequences for all cooling models to enable
fair per-CDU comparison across different systems.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class StandardizedInputGenerator:
    """
    Generates standardized per-CDU input sequences for comparative analysis.
    
    Creates identical input conditions across all cooling models, enabling
    direct comparison of CDU-level responses.
    """
    
    # Standard input variable names (per-CDU)
    INPUT_VARS = ['Q_flow', 'T_Air', 'T_ext']
    
    # Standard output variable names (per-CDU)
    OUTPUT_VARS = [
        'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
        'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
        'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
    ]
    
    # Default input ranges
    DEFAULT_RANGES = {
        'Q_flow': (12.0, 40.0),      # kW - Heat load per CDU
        'T_Air': (288.15, 308.15),   # K (15-35°C) - Inlet air temperature
        'T_ext': (283.15, 313.15)    # K (10-40°C) - External/ambient temperature
    }
    
    def __init__(
        self,
        system_name: str,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the standardized input generator.
        
        Args:
            system_name: System configuration name for column formatting
            input_ranges: Optional custom input ranges
            random_seed: Random seed for reproducibility
        """
        self.system_name = system_name
        self.input_ranges = input_ranges or self.DEFAULT_RANGES.copy()
        self.random_seed = random_seed
        
        # Load system config for CDU count
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', self.config.get('num_cdus', 1))
        
        np.random.seed(random_seed)
        
        logger.info(f"StandardizedInputGenerator initialized for {system_name}")
        logger.info(f"  - CDUs: {self.num_cdus}")
        logger.info(f"  - Input ranges: {self.input_ranges}")
    
    def get_input_column_name(self, var_name: str, cdu_idx: int) -> str:
        """Get the column name for a given input variable and CDU index."""
        if var_name == 'Q_flow':
            return f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
        elif var_name == 'T_Air':
            return f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
        elif var_name == 'T_ext':
            return 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        else:
            raise ValueError(f"Unknown input variable: {var_name}")
    
    def get_output_column_name(self, var_name: str, cdu_idx: int) -> str:
        """Get the column name for a given output variable and CDU index."""
        return f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{var_name}'
    
    def generate_grid_inputs(
        self,
        n_points_per_dim: int = 10,
        fixed_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate grid-based inputs for response surface analysis.
        
        Args:
            n_points_per_dim: Number of points per input dimension
            fixed_values: Optional dict of fixed values for some inputs
            
        Returns:
            DataFrame with grid inputs formatted for all CDUs
        """
        fixed_values = fixed_values or {}
        
        # Create ranges for non-fixed variables
        ranges = {}
        for var in self.INPUT_VARS:
            if var in fixed_values:
                ranges[var] = np.array([fixed_values[var]])
            else:
                ranges[var] = np.linspace(
                    self.input_ranges[var][0],
                    self.input_ranges[var][1],
                    n_points_per_dim
                )
        
        # Create meshgrid
        grids = np.meshgrid(*[ranges[v] for v in self.INPUT_VARS], indexing='ij')
        
        # Flatten to create input combinations
        n_samples = np.prod([len(ranges[v]) for v in self.INPUT_VARS])
        inputs = {var: grids[i].flatten() for i, var in enumerate(self.INPUT_VARS)}
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            for var in self.INPUT_VARS:
                col_name = self.get_input_column_name(var, cdu_idx)
                if var == 'T_ext':
                    if cdu_idx == 1:  # T_ext is shared
                        df_data[col_name] = inputs[var]
                else:
                    df_data[col_name] = inputs[var]
        
        logger.info(f"Generated grid inputs: {n_samples} samples")
        return pd.DataFrame(df_data)
    
    def generate_lhs_inputs(
        self,
        n_samples: int = 500,
        uniform_across_cdus: bool = True
    ) -> pd.DataFrame:
        """
        Generate Latin Hypercube Sampling inputs for sensitivity analysis.
        
        Args:
            n_samples: Number of samples to generate
            uniform_across_cdus: If True, all CDUs get same inputs
            
        Returns:
            DataFrame with LHS inputs
        """
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=len(self.INPUT_VARS), seed=self.random_seed)
        samples = sampler.random(n=n_samples)
        
        # Scale to input ranges
        l_bounds = [self.input_ranges[v][0] for v in self.INPUT_VARS]
        u_bounds = [self.input_ranges[v][1] for v in self.INPUT_VARS]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        
        # Create DataFrame
        df_data = {}
        
        if uniform_across_cdus:
            # All CDUs get the same inputs
            for cdu_idx in range(1, self.num_cdus + 1):
                for i, var in enumerate(self.INPUT_VARS):
                    col_name = self.get_input_column_name(var, cdu_idx)
                    if var == 'T_ext':
                        if cdu_idx == 1:
                            df_data[col_name] = scaled_samples[:, i]
                    else:
                        df_data[col_name] = scaled_samples[:, i]
        else:
            # Each CDU gets different inputs (for spatial analysis)
            for cdu_idx in range(1, self.num_cdus + 1):
                cdu_samples = sampler.random(n=n_samples)
                cdu_scaled = qmc.scale(cdu_samples, l_bounds, u_bounds)
                
                for i, var in enumerate(self.INPUT_VARS):
                    col_name = self.get_input_column_name(var, cdu_idx)
                    if var == 'T_ext':
                        if cdu_idx == 1:
                            df_data[col_name] = scaled_samples[:, i]
                    else:
                        df_data[col_name] = cdu_scaled[:, i]
        
        logger.info(f"Generated LHS inputs: {n_samples} samples")
        return pd.DataFrame(df_data)
    
    def generate_time_series_inputs(
        self,
        n_samples: int = 3600,
        base_values: Optional[Dict[str, float]] = None,
        variation_amplitude: float = 0.1,
        transition_steps: int = 60
    ) -> pd.DataFrame:
        """
        Generate time-series inputs with smooth transitions.
        
        Args:
            n_samples: Number of time steps
            base_values: Optional base values for each input
            variation_amplitude: Relative amplitude of variations (0-1)
            transition_steps: Steps for smooth transitions
            
        Returns:
            DataFrame with time-series inputs
        """
        if base_values is None:
            base_values = {
                var: (self.input_ranges[var][0] + self.input_ranges[var][1]) / 2
                for var in self.INPUT_VARS
            }
        
        # Generate smooth random walks for each input
        inputs = {}
        for var in self.INPUT_VARS:
            base = base_values[var]
            amplitude = (self.input_ranges[var][1] - self.input_ranges[var][0]) * variation_amplitude
            
            # Number of change points
            n_changes = n_samples // transition_steps
            change_points = np.linspace(0, n_samples - 1, n_changes + 1).astype(int)
            target_values = np.random.uniform(
                base - amplitude, base + amplitude, n_changes + 1
            )
            
            # Interpolate smoothly
            values = np.interp(np.arange(n_samples), change_points, target_values)
            
            # Clip to valid range
            values = np.clip(values, self.input_ranges[var][0], self.input_ranges[var][1])
            inputs[var] = values
        
        # Expand to all CDUs
        df_data = {}
        for cdu_idx in range(1, self.num_cdus + 1):
            for var in self.INPUT_VARS:
                col_name = self.get_input_column_name(var, cdu_idx)
                if var == 'T_ext':
                    if cdu_idx == 1:
                        df_data[col_name] = inputs[var]
                else:
                    df_data[col_name] = inputs[var]
        
        logger.info(f"Generated time-series inputs: {n_samples} samples")
        return pd.DataFrame(df_data)
    
    def generate_standardized_test_suite(
        self,
        include_grid: bool = True,
        include_lhs: bool = True,
        include_time_series: bool = True,
        grid_points: int = 10,
        lhs_samples: int = 500,
        time_series_length: int = 3600
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete standardized test suite.
        
        Args:
            include_grid: Include grid-based inputs
            include_lhs: Include LHS inputs
            include_time_series: Include time-series inputs
            grid_points: Points per dimension for grid
            lhs_samples: Number of LHS samples
            time_series_length: Length of time series
            
        Returns:
            Dictionary of test input DataFrames
        """
        test_suite = {}
        
        if include_grid:
            test_suite['grid'] = self.generate_grid_inputs(n_points_per_dim=grid_points)
        
        if include_lhs:
            test_suite['lhs'] = self.generate_lhs_inputs(n_samples=lhs_samples)
        
        if include_time_series:
            test_suite['time_series'] = self.generate_time_series_inputs(
                n_samples=time_series_length
            )
        
        logger.info(f"Generated standardized test suite with {len(test_suite)} test types")
        return test_suite
    
    @staticmethod
    def extract_per_cdu_data(
        data: pd.DataFrame,
        cdu_idx: int,
        num_cdus: int
    ) -> pd.DataFrame:
        """
        Extract per-CDU input/output data from full simulation DataFrame.
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index to extract (1-based)
            num_cdus: Total number of CDUs in system
            
        Returns:
            DataFrame with normalized column names for single CDU
        """
        # Input column patterns
        q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
        t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        
        # Output column patterns
        output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
        extracted = {}
        
        # Extract inputs
        if q_flow_col in data.columns:
            extracted['Q_flow'] = data[q_flow_col].values
        if t_air_col in data.columns:
            extracted['T_Air'] = data[t_air_col].values
        if t_ext_col in data.columns:
            extracted['T_ext'] = data[t_ext_col].values
        
        # Extract outputs
        for var in output_vars:
            col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{var}'
            if col in data.columns:
                extracted[var] = data[col].values
        
        return pd.DataFrame(extracted)
