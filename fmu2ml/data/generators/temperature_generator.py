import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import qmc
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import os


class TemperatureGenerator:
    """CDU temperature generator using heat balance model"""
    
    def __init__(self, config: Dict, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        self.config = config
        
        # Temperature ranges (ASHRAE standards)
        self.t_air_normal = (18, 27)
        self.t_air_extreme = (15, 32)
        self.incident_prob = 0.005
        
        # Physical constants
        self.cp = 1005  # Specific heat of air (J/kg·K)
        self.dt = 1     # Time step (seconds)
    
    def generate_cdu_parameters(self, n_cdus: int) -> list:
        """Generate physical parameters for each CDU using Latin Hypercube Sampling"""
        
        param_ranges = {
            'baseline': (21, 23),
            'k_ext': (0.15, 0.4),
            'beta': (0.4, 0.9),
            'tau': (90, 300),
        }
        
        sampler = qmc.LatinHypercube(d=len(param_ranges), seed=self.seed)
        samples = sampler.random(n_cdus)
        
        cdu_params = []
        param_names = list(param_ranges.keys())
        
        for i in range(n_cdus):
            params = {}
            for j, param in enumerate(param_names):
                min_val, max_val = param_ranges[param]
                params[param] = min_val + samples[i, j] * (max_val - min_val)
            
            params['alpha'] = self.dt / params['tau']
            params['cdu_id'] = i + 1
            
            cdu_params.append(params)
        
        return cdu_params
    
    def smooth_external_temperature(self, t_ext: np.ndarray, window_hours: int = 3) -> np.ndarray:
        """Smooth external temperature using Savitzky-Golay filter"""
        
        window_samples = int(window_hours * 60)
        
        if window_samples % 2 == 0:
            window_samples += 1
        
        window_samples = max(5, min(window_samples, len(t_ext) // 2))
        
        try:
            t_ext_smooth = savgol_filter(t_ext, window_samples, polyorder=3)
        except:
            t_ext_smooth = np.convolve(t_ext, np.ones(window_samples)/window_samples, mode='same')
        
        return t_ext_smooth
    
    def calculate_temperature_response(
        self,
        power_kw: np.ndarray,
        t_ext: np.ndarray,
        cdu_params: Dict
    ) -> np.ndarray:
        """Calculate temperature response using heat balance"""
        
        n_timesteps = len(power_kw)
        temperature = np.zeros(n_timesteps)
        temperature[0] = cdu_params['baseline']
        
        incident_prob_per_step = self.incident_prob / 1440
        
        for t in range(1, n_timesteps):
            t_supply = cdu_params['baseline'] + cdu_params['k_ext'] * (t_ext[t] - 25)
            
            if power_kw[t] > 0:
                m_dot = cdu_params['beta'] * power_kw[t]
                heat_effect = power_kw[t] * 1000 / (m_dot * self.cp)
            else:
                heat_effect = 0
            
            t_target = t_supply + heat_effect
            
            temperature[t] = (1 - cdu_params['alpha']) * temperature[t-1] + \
                           cdu_params['alpha'] * t_target + \
                           np.random.normal(0, 0.05)
            
            if np.random.random() < incident_prob_per_step:
                spike_temp = np.random.uniform(28, 32)
                spike_duration = np.random.randint(1, 5)
                
                for j in range(min(spike_duration, n_timesteps - t)):
                    decay_factor = np.exp(-j / spike_duration)
                    temperature[t + j] = temperature[t + j] * (1 - decay_factor) + spike_temp * decay_factor
        
        temperature = np.clip(temperature, self.t_air_extreme[0], self.t_air_extreme[1])
        temperature = savgol_filter(temperature, window_length=5, polyorder=2)
        
        return temperature
    
    def generate_temperatures(
        self,
        power_data: pd.DataFrame,
        start_date: Optional[str] = None,
        timestep_seconds: int = 60
    ) -> pd.DataFrame:
        """Generate air temperatures for CDUs based on power data"""
        
        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
        
        # Detect CDUs from power data
        cdu_columns = sorted([col for col in power_data.columns if col.startswith('CDU_')])
        n_cdus = len(cdu_columns)
        
        if n_cdus == 0:
            raise ValueError("No CDU columns found in power_data. Expected columns starting with 'CDU_'")
        
        print(f"Detected {n_cdus} CDUs in power data")
        
        num_timesteps = len(power_data)
        
        # Initialize Weather object
        try:
            from raps.weather import Weather
            weather = Weather(start_date, self.config)
        except ImportError:
            weather = None
        
        # Generate CDU parameters
        cdu_params = self.generate_cdu_parameters(n_cdus)
        
        # Generate timestamps
        start_datetime = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        timestamps = [start_datetime + timedelta(seconds=i*timestep_seconds) for i in range(num_timesteps)]
        
        # Get external temperature
        t_ext_raw = np.zeros(num_timesteps)
        
        if weather:
            unique_days = set(ts.strftime('%Y-%m-%d') for ts in timestamps)
            for day in sorted(unique_days):
                weather.retrieve_weather_data_for_day(day)
            
            for i, ts in enumerate(timestamps):
                temp = weather.get_temperature(ts)
                if temp is not None:
                    t_ext_raw[i] = temp - 273.15
                else:
                    t_ext_raw[i] = self.config.get('WET_BULB_TEMP', 298.15) - 273.15
        else:
            t_ext_raw = np.full(num_timesteps, 25.0)
        
        # Smooth external temperature
        t_ext = self.smooth_external_temperature(t_ext_raw)
        
        # Initialize output dictionary
        cdu_temp = {}
        
        # Generate temperature for each CDU
        for i, cdu_name in enumerate(cdu_columns):
            params = cdu_params[i]
            
            if cdu_name in power_data.columns:
                power_kw = power_data[cdu_name].values / 1000.0
            else:
                power_kw = np.ones(num_timesteps) * 25.0
            
            cdu_temp[cdu_name] = self.calculate_temperature_response(power_kw, t_ext, params)
        
        df_temps = pd.DataFrame(cdu_temp)
        df_temps['T_ext'] = t_ext 
        
        return df_temps + 273.15
    
    def format_for_fmu(self, temps_df: pd.DataFrame, power_df: pd.DataFrame) -> pd.DataFrame:
        """Format data for FMU input"""
        
        fmu_data = {}
        
        # Add external temperature
        fmu_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = temps_df['T_ext'].values
        
        # Get actual CDU columns
        cdu_columns = sorted([col for col in temps_df.columns if col.startswith('CDU_')])
        
        # Map CDU data
        for cdu_name in cdu_columns:
            cdu_num = int(cdu_name.split('_')[1])
            
            if cdu_name in temps_df.columns:
                col_name = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_T_Air'
                fmu_data[col_name] = temps_df[cdu_name].values
            
            if cdu_name in power_df.columns:
                col_name = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_Q_flow_total'
                fmu_data[col_name] = power_df[cdu_name].values
        
        fmu_input = pd.DataFrame(fmu_data, index=temps_df.index)
            
        return fmu_input


def generate_temperature_dataset(
    power_df: pd.DataFrame,
    start_date: Optional[str] = None,
    timestep_seconds: int = 60,
    seed: int = 42,
    output_dir: str = "data",
    output_format: str = "parquet",
    save_output: bool = False,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Temperature dataset generation"""
    
    if config is None:
        try:
            from raps.config import ConfigManager
            config = ConfigManager(system_name="marconi100").get_config()
            print("Using configuration from ConfigManager for 'marconi100'")
            
        except ImportError:
            config = {'WET_BULB_TEMP': 298.15}
    
    temp_gen = TemperatureGenerator(config=config, seed=seed)
    
    temps_df = temp_gen.generate_temperatures(
        power_data=power_df,
        start_date=start_date,
        timestep_seconds=timestep_seconds
    )
    
    fmu_input = temp_gen.format_for_fmu(temps_df, power_df)
    
    filename = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        system_name = config.get('system_name')
        duration_hours = len(power_df) * timestep_seconds / 3600
        n_cdus = len([col for col in power_df.columns if 'CDU' in col])
        
        if output_format.lower() == "parquet":
            filename = f"fmu_input_{int(duration_hours)}hrs_{system_name}_{n_cdus}CDU.parquet"
            filepath = os.path.join(output_dir, filename)
            fmu_input.to_parquet(filepath)
        else:
            filename = f"fmu_input_{int(duration_hours)}hrs_{system_name}_{n_cdus}CDU.csv"
            filepath = os.path.join(output_dir, filename)
            fmu_input.to_csv(filepath, index=False)
        
        print(f"Temperature dataset saved to: {filepath}")
    
    return fmu_input, filename