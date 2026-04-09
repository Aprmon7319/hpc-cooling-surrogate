from dataclasses import dataclass
from typing import Dict, Any
from raps.config import ConfigManager


@dataclass
class SystemConfig:
    """Base system configuration."""
    name: str
    min_power: float
    max_power: float
    min_normal_load: float
    max_normal_load: float
    wet_bulb_temp: float
    water_density: float = 997.0  # kg/m³
    water_specific_heat: float = 4186.0  # J/(kg·K)
    gpm_to_m3_s: float = 6.30902e-5
    
    @classmethod
    def from_raps_config(cls, system_name: str) -> 'SystemConfig':
        """Create SystemConfig from RAPS configuration."""
        config_manager = ConfigManager(system_name=system_name)
        raps_config = config_manager.get_config()
        
        return cls(
            name=system_name,
            min_power=raps_config['MIN_POWER'],
            max_power=raps_config['MAX_POWER'],
            min_normal_load=raps_config['MIN_NORMAL_LOAD'],
            max_normal_load=raps_config['MAX_NORMAL_LOAD'],
            wet_bulb_temp=raps_config.get('WET_BULB_TEMP', 298.15)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'min_power': self.min_power,
            'max_power': self.max_power,
            'min_normal_load': self.min_normal_load,
            'max_normal_load': self.max_normal_load,
            'wet_bulb_temp': self.wet_bulb_temp,
            'water_density': self.water_density,
            'water_specific_heat': self.water_specific_heat,
            'gpm_to_m3_s': self.gpm_to_m3_s
        }


def get_system_config(system_name: str) -> SystemConfig:
    """
    Get system configuration by name.
    
    Parameters
    ----------
    system_name : str
        Name of the system (e.g., 'marconi100', 'summit')
    
    Returns
    -------
    SystemConfig
        System configuration object
    """
    return SystemConfig.from_raps_config(system_name)