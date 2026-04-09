import numpy as np
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime, timedelta


class InputFormatter:
    """
    Formats data for FMU simulation input
    
    Handles:
    - Column naming conventions
    - Unit conversions
    - Time series generation
    - Input validation
    - Data structure formatting
    """
    
    def __init__(self, num_cdus: int = 49, timestep: float = 60.0):
        """
        Initialize input formatter
        
        Args:
            num_cdus: Number of CDUs in the system
            timestep: Timestep in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.num_cdus = num_cdus
        self.timestep = timestep
        
        self.logger.info(f"Input formatter initialized: {num_cdus} CDUs, {timestep}s timestep")
    
    def format_power_data(
        self,
        power_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        start_time: float = 0.0
    ) -> pd.DataFrame:
        """
        Format power data into FMU input structure
        
        Args:
            power_data: Array of shape (n_timesteps, n_cdus) with power values in Watts
            timestamps: Optional timestamps array
            start_time: Start time if timestamps not provided
            
        Returns:
            DataFrame with proper column names
        """
        self.logger.debug(f"Formatting power data: shape={power_data.shape}")
        
        # Validate shape
        if power_data.ndim == 1:
            power_data = power_data.reshape(-1, 1)
        
        n_timesteps, n_cdus = power_data.shape
        
        if n_cdus != self.num_cdus:
            raise ValueError(f"Expected {self.num_cdus} CDUs, got {n_cdus}")
        
        # Create column names using proper simulator format
        power_cols = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total' 
                     for i in range(1, self.num_cdus + 1)]
        
        # Create DataFrame
        df = pd.DataFrame(power_data, columns=power_cols)
        
        # Add time column
        if timestamps is not None:
            df['time'] = timestamps
        else:
            df['time'] = start_time + np.arange(n_timesteps) * self.timestep
        
        # Reorder columns (time first)
        cols = ['time'] + power_cols
        df = df[cols]
        
        return df
    
    def format_temperature_data(
        self,
        air_temperature_data: np.ndarray,
        external_temperature: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        start_time: float = 0.0
    ) -> pd.DataFrame:
        """
        Format temperature data (air temperatures per CDU + external temperature)
        
        Args:
            air_temperature_data: Array of shape (n_timesteps, n_cdus) with air temperatures in Kelvin
            external_temperature: Array of shape (n_timesteps,) with external temperature in Kelvin
            timestamps: Optional timestamps array
            start_time: Start time if timestamps not provided
            
        Returns:
            DataFrame with 'time', air temperature columns, and 'T_ext' column
        """
        self.logger.debug(f"Formatting temperature data: shape={air_temperature_data.shape}")
        
        # Validate shape
        if air_temperature_data.ndim == 1:
            air_temperature_data = air_temperature_data.reshape(-1, 1)
        
        n_timesteps, n_cdus = air_temperature_data.shape
        
        if n_cdus != self.num_cdus:
            raise ValueError(f"Expected {self.num_cdus} CDUs, got {n_cdus}")
        
        if len(external_temperature) != n_timesteps:
            raise ValueError(f"External temperature length {len(external_temperature)} "
                           f"doesn't match timesteps {n_timesteps}")
        
        # Create column names for air temperatures
        air_temp_cols = [f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air' 
                        for i in range(1, self.num_cdus + 1)]
        
        # Create DataFrame
        df = pd.DataFrame(air_temperature_data, columns=air_temp_cols)
        
        # Add external temperature
        df['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = external_temperature
        
        # Add time column
        if timestamps is not None:
            df['time'] = timestamps
        else:
            df['time'] = start_time + np.arange(n_timesteps) * self.timestep
        
        # Reorder columns (time first)
        cols = ['time'] + air_temp_cols + ['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext']
        df = df[cols]
        
        return df
    
    def merge_inputs(
        self,
        power_df: pd.DataFrame,
        temperature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge power and temperature data
        
        Args:
            power_df: DataFrame with power data
            temperature_df: DataFrame with temperature data
            
        Returns:
            Merged DataFrame with all inputs
        """
        self.logger.debug("Merging power and temperature data")
        
        # Merge on time
        merged = pd.merge(power_df, temperature_df, on='time', how='inner')
        
        # Validate
        if len(merged) == 0:
            raise ValueError("No matching timestamps in power and temperature data")
        
        self.logger.debug(f"Merged data: {len(merged)} timesteps")
        
        return merged
    
    def convert_units(
        self,
        df: pd.DataFrame,
        power_unit: str = 'W',
        temperature_unit: str = 'K'
    ) -> pd.DataFrame:
        """
        Convert units if needed
        
        Args:
            df: Input DataFrame
            power_unit: Current power unit ('W', 'kW', 'MW')
            temperature_unit: Current temperature unit ('K', 'C', 'F')
            
        Returns:
            DataFrame with converted units (to Watts and Kelvin)
        """
        df = df.copy()
        
        # Convert power to Watts
        power_cols = [col for col in df.columns if 'Q_flow_total' in col]
        
        if power_unit == 'kW':
            df[power_cols] = df[power_cols] * 1000.0
        elif power_unit == 'MW':
            df[power_cols] = df[power_cols] * 1000000.0
        elif power_unit == 'W':
            pass  # Already in Watts
        else:
            raise ValueError(f"Unsupported power unit: {power_unit}")
        
        # Convert temperature to Kelvin
        temp_cols = [col for col in df.columns if 'T_Air' in col or 'T_ext' in col]
        
        if temperature_unit == 'C':
            df[temp_cols] = df[temp_cols] + 273.15
        elif temperature_unit == 'F':
            df[temp_cols] = (df[temp_cols] - 32) * 5/9 + 273.15
        elif temperature_unit == 'K':
            pass  # Already in Kelvin
        else:
            raise ValueError(f"Unsupported temperature unit: {temperature_unit}")
        
        return df
    