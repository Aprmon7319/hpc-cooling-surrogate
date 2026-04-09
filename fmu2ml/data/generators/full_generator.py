"""
generate_fmu_dataset.py
Generate complete FMU-compatible input dataset with power and temperature data
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List

# Import your modules
from ..generators import (
    generate_continuous_power_data,
    generate_temperature_dataset
)

def generate_complete_fmu_dataset(
    n_cdus: int = 49,
    duration_hours: int = 4,
    timestep_seconds: int = 1,
    start_date: Optional[str] = None,
    scenario_distribution: List[float] = [0.5, 0.4, 0.1],  # [normal, edge, fault]
    seed: int = 42,
    output_dir: str = "data",
    save_output: bool = False,
    config: Optional[Dict] = None,
    initial_base_power: Optional[float] = None
) -> Dict:
    """
    Generate complete FMU-compatible dataset with power and temperature data.
    
    Parameters:
    -----------
    n_cdus : int
        Number of CDUs to simulate
    duration_hours : int
        Simulation duration in hours
    timestep_seconds : int
        Time step in seconds (default 60 for FMU)
    start_date : str, optional
        Start date in ISO format (e.g., "2024-01-01T00:00:00Z")
    scenario_distribution : List[float]
        Distribution of [normal, edge, fault] scenarios
    seed : int
        Random seed for reproducibility
    output_dir : str
        Directory to save output files
    save_output : bool
        Whether to save the output to file
    config : Dict, optional
        System configuration (uses Marconi100 defaults if None)
    
    Returns:
    --------
    Dict containing:
        - 'fmu_input': Complete FMU input DataFrame
        - 'power_data': Raw power data DataFrame
        - 'temperature_data': Raw temperature data DataFrame
        - 'scenarios': Scenario information for each CDU
        - 'filename': Output filename if saved
    """
    
    # Set default start date if not provided
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
    
    print(f"Generating FMU dataset:")
    print(f"  - CDUs: {n_cdus}")
    print(f"  - Duration: {duration_hours} hours")
    print(f"  - Timestep: {timestep_seconds} seconds")
    print(f"  - Start date: {start_date}")
    print(f"  - Scenario distribution: Normal={scenario_distribution[0]:.0%}, "
          f"Edge={scenario_distribution[1]:.0%}, Fault={scenario_distribution[2]:.0%}")
    
    # Step 1: Generate power consumption data
    print("\n1. Generating power data...")
    power_df, scenarios = generate_continuous_power_data(
        n_cdus=n_cdus,
        duration_hours=duration_hours,
        timestep_seconds=timestep_seconds,
        initial_base_power=initial_base_power,
        seed=seed,
        distribution=scenario_distribution,
        config=config
    )
    print(f"   Power data shape: {power_df.shape}")
    
    # Step 2: Generate temperature data based on power consumption
    print("\n2. Generating temperature data...")
    fmu_input_df, filename = generate_temperature_dataset(
        power_df=power_df,
        start_date=start_date,
        timestep_seconds=timestep_seconds,
        seed=seed,
        output_dir=output_dir,
        output_format="parquet",
        save_output=save_output,
        config=config
    )
    print(f"   FMU input shape: {fmu_input_df.shape}")
    
    # Sample column names for verification
    print("\n   Sample FMU columns:")
    for col in list(fmu_input_df.columns)[:5]:
        print(f"     - {col}")
    
    if save_output:
        print(f"\n4. Dataset saved to: {output_dir}/{filename}")
    
    return {
        'fmu_input': fmu_input_df,
        'power_data': power_df,
        'scenarios': scenarios,
        'filename': filename
    }
