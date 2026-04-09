"""
System Profiler for Cooling Model Analysis.

Extracts and profiles key characteristics of individual cooling systems.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class SystemProfiler:
    """
    Profiles individual data center cooling systems.
    
    Extracts system configuration, simulates characteristic operating
    conditions, and summarizes key behavioral traits.
    """
    
    def __init__(self, system_name: str, **config_overrides):
        """
        Initialize the system profiler.
        
        Args:
            system_name: System configuration name (e.g., 'marconi100', 'summit')
            **config_overrides: Additional configuration overrides
        """
        self.system_name = system_name
        
        # Load system configuration
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        # Extract key configuration values
        self.num_cdus = self.config.get('NUM_CDUS', self.config.get('num_cdus', 1))
        self.cooling_efficiency = self.config.get('COOLING_EFFICIENCY', 0.0)
        self.wet_bulb_temp = self.config.get('WET_BULB_TEMP', 290.0)
        self.fmu_path = self.config.get('FMU_PATH', '')
        
        logger.info(f"SystemProfiler initialized for: {system_name}")
        logger.info(f"  - CDUs: {self.num_cdus}")
        logger.info(f"  - Cooling Efficiency: {self.cooling_efficiency}")
        
    def get_system_profile(self) -> Dict[str, Any]:
        """
        Get the complete system profile.
        
        Returns:
            Dictionary with system configuration and characteristics
        """
        profile = {
            'system_name': self.system_name,
            'configuration': {
                'num_cdus': self.num_cdus,
                'racks_per_cdu': self.config.get('RACKS_PER_CDU', 1),
                'nodes_per_rack': self.config.get('NODES_PER_RACK', 0),
                'cpus_per_node': self.config.get('CPUS_PER_NODE', 0),
                'gpus_per_node': self.config.get('GPUS_PER_NODE', 0),
                'cooling_efficiency': self.cooling_efficiency,
                'wet_bulb_temp_k': self.wet_bulb_temp,
                'location': {
                    'zip_code': self.config.get('ZIP_CODE', ''),
                    'country': self.config.get('COUNTRY_CODE', '')
                }
            },
            'compute_capacity': self._compute_capacity_metrics(),
            'power_characteristics': self._get_power_characteristics(),
            'fmu_model': {
                'path': self.fmu_path,
                'output_variables': list(self.config.get('FMU_COLUMN_MAPPING', {}).keys())
            }
        }
        
        return profile
    
    def _compute_capacity_metrics(self) -> Dict[str, Any]:
        """Compute total system capacity metrics."""
        num_cdus = self.num_cdus
        racks_per_cdu = self.config.get('RACKS_PER_CDU', 1)
        nodes_per_rack = self.config.get('NODES_PER_RACK', 0)
        cpus_per_node = self.config.get('CPUS_PER_NODE', 0)
        gpus_per_node = self.config.get('GPUS_PER_NODE', 0)
        
        total_racks = num_cdus * racks_per_cdu
        total_nodes = total_racks * nodes_per_rack
        total_cpus = total_nodes * cpus_per_node
        total_gpus = total_nodes * gpus_per_node
        
        # Compute theoretical peak FLOPS
        cpu_peak_flops = self.config.get('CPU_PEAK_FLOPS', 0)
        gpu_peak_flops = self.config.get('GPU_PEAK_FLOPS', 0)
        
        total_cpu_flops = total_cpus * cpu_peak_flops
        total_gpu_flops = total_gpus * gpu_peak_flops
        total_flops = total_cpu_flops + total_gpu_flops
        
        return {
            'total_racks': total_racks,
            'total_nodes': total_nodes,
            'total_cpus': total_cpus,
            'total_gpus': total_gpus,
            'peak_cpu_flops': total_cpu_flops,
            'peak_gpu_flops': total_gpu_flops,
            'peak_total_flops': total_flops,
            'peak_total_pflops': total_flops / 1e15 if total_flops > 0 else 0
        }
    
    def _get_power_characteristics(self) -> Dict[str, Any]:
        """Get power-related characteristics."""
        min_power = self.config.get('MIN_POWER', 0)
        max_power = self.config.get('MAX_POWER', 0)
        
        nodes_per_rack = self.config.get('NODES_PER_RACK', 0)
        
        return {
            'min_power_per_node_kw': min_power,
            'max_power_per_node_kw': max_power,
            'estimated_min_rack_power_kw': min_power * nodes_per_rack,
            'estimated_max_rack_power_kw': max_power * nodes_per_rack,
            'min_normal_load': self.config.get('MIN_NORMAL_LOAD', 0),
            'max_normal_load': self.config.get('MAX_NORMAL_LOAD', 0)
        }
    
    def get_input_column_templates(self) -> Dict[str, str]:
        """
        Get input column name templates for this system.
        
        Returns:
            Dictionary mapping input types to column templates
        """
        return {
            'Q_flow': 'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_Q_flow_total',
            'T_Air': 'simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_T_Air',
            'T_ext': 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        }
    
    def get_output_column_templates(self) -> Dict[str, str]:
        """
        Get output column name templates for this system.
        
        Returns:
            Dictionary mapping output types to column templates
        """
        return {
            'V_flow_prim_GPM': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.V_flow_prim_GPM',
            'V_flow_sec_GPM': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.V_flow_sec_GPM',
            'W_flow_CDUP_kW': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.W_flow_CDUP_kW',
            'T_prim_s_C': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.T_prim_s_C',
            'T_prim_r_C': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.T_prim_r_C',
            'T_sec_s_C': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.T_sec_s_C',
            'T_sec_r_C': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.T_sec_r_C',
            'p_prim_s_psig': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.p_prim_s_psig',
            'p_prim_r_psig': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.p_prim_r_psig',
            'p_sec_s_psig': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.p_sec_s_psig',
            'p_sec_r_psig': 'simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.p_sec_r_psig'
        }
    
    def summarize(self) -> str:
        """
        Generate a human-readable summary of the system.
        
        Returns:
            Formatted summary string
        """
        profile = self.get_system_profile()
        
        lines = [
            f"=" * 60,
            f"System Profile: {self.system_name.upper()}",
            f"=" * 60,
            f"",
            f"Location: {profile['configuration']['location']['country']}",
            f"",
            f"Infrastructure:",
            f"  - CDUs: {profile['configuration']['num_cdus']}",
            f"  - Racks per CDU: {profile['configuration']['racks_per_cdu']}",
            f"  - Total Racks: {profile['compute_capacity']['total_racks']}",
            f"  - Nodes per Rack: {profile['configuration']['nodes_per_rack']}",
            f"  - Total Nodes: {profile['compute_capacity']['total_nodes']}",
            f"",
            f"Compute Capacity:",
            f"  - CPUs per Node: {profile['configuration']['cpus_per_node']}",
            f"  - GPUs per Node: {profile['configuration']['gpus_per_node']}",
            f"  - Total CPUs: {profile['compute_capacity']['total_cpus']}",
            f"  - Total GPUs: {profile['compute_capacity']['total_gpus']}",
            f"  - Peak Performance: {profile['compute_capacity']['peak_total_pflops']:.2f} PFLOPS",
            f"",
            f"Cooling:",
            f"  - Efficiency: {profile['configuration']['cooling_efficiency']}",
            f"  - Wet Bulb Temp: {profile['configuration']['wet_bulb_temp_k']:.2f} K",
            f"",
            f"Power Characteristics:",
            f"  - Min Node Power: {profile['power_characteristics']['min_power_per_node_kw']:.2f} kW",
            f"  - Max Node Power: {profile['power_characteristics']['max_power_per_node_kw']:.2f} kW",
            f"  - Max Rack Power: {profile['power_characteristics']['estimated_max_rack_power_kw']:.2f} kW",
            f"=" * 60
        ]
        
        return "\n".join(lines)
