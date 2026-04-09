"""
Metrics Calculator for Cooling Model Analysis.

Provides standardized metrics computation for comparing cooling systems.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates standardized metrics for cooling system comparison.
    
    Computes efficiency, thermal, and dynamic metrics that allow
    fair comparison across different system scales.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.output_vars = [
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
            'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
            'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
        ]
        
    def compute_efficiency_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict
    ) -> Dict[str, float]:
        """
        Compute cooling efficiency metrics.
        
        Args:
            data: Simulation data with cooling outputs
            system_config: System configuration dictionary
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Aggregate CDUP power across all CDUs
        cdup_power_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.W_flow_CDUP_kW'
            for i in range(1, num_cdus + 1)
        ]
        
        available_cdup_cols = [col for col in cdup_power_cols if col in data.columns]
        
        if available_cdup_cols:
            total_cdup_power = data[available_cdup_cols].sum(axis=1)
            metrics['mean_cdup_power_kw'] = float(total_cdup_power.mean())
            metrics['max_cdup_power_kw'] = float(total_cdup_power.max())
            metrics['cdup_power_std'] = float(total_cdup_power.std())
        
        # Calculate heat load (Q_flow) total
        q_flow_cols = [
            f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total'
            for i in range(1, num_cdus + 1)
        ]
        
        available_q_cols = [col for col in q_flow_cols if col in data.columns]
        
        if available_q_cols:
            total_heat_load = data[available_q_cols].sum(axis=1)
            metrics['mean_heat_load_kw'] = float(total_heat_load.mean())
            metrics['max_heat_load_kw'] = float(total_heat_load.max())
            
            # Cooling Power Usage Effectiveness (normalized by heat removed)
            if 'mean_cdup_power_kw' in metrics and metrics['mean_heat_load_kw'] > 0:
                metrics['cooling_power_ratio'] = (
                    metrics['mean_cdup_power_kw'] / metrics['mean_heat_load_kw']
                )
        
        # Per-CDU normalized metrics
        if 'mean_cdup_power_kw' in metrics:
            metrics['cdup_power_per_cdu_kw'] = metrics['mean_cdup_power_kw'] / num_cdus
            
        if 'mean_heat_load_kw' in metrics:
            metrics['heat_load_per_cdu_kw'] = metrics['mean_heat_load_kw'] / num_cdus
        
        return metrics
    
    def compute_thermal_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict
    ) -> Dict[str, float]:
        """
        Compute thermal performance metrics.
        
        Args:
            data: Simulation data with temperature outputs
            system_config: System configuration dictionary
            
        Returns:
            Dictionary of thermal metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Collect temperature data from all CDUs
        temp_return_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_sec_r_C'
            for i in range(1, num_cdus + 1)
        ]
        temp_supply_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_sec_s_C'
            for i in range(1, num_cdus + 1)
        ]
        
        available_return_cols = [col for col in temp_return_cols if col in data.columns]
        available_supply_cols = [col for col in temp_supply_cols if col in data.columns]
        
        if available_return_cols:
            all_return_temps = data[available_return_cols]
            metrics['mean_rack_return_temp_c'] = float(all_return_temps.mean().mean())
            metrics['max_rack_return_temp_c'] = float(all_return_temps.max().max())
            metrics['rack_return_temp_std'] = float(all_return_temps.std().mean())
            
        if available_supply_cols:
            all_supply_temps = data[available_supply_cols]
            metrics['mean_rack_supply_temp_c'] = float(all_supply_temps.mean().mean())
            metrics['rack_supply_temp_range'] = float(
                all_supply_temps.max().max() - all_supply_temps.min().min()
            )
        
        # Temperature delta (return - supply)
        if available_return_cols and available_supply_cols:
            return_mean = data[available_return_cols].mean(axis=1)
            supply_mean = data[available_supply_cols].mean(axis=1)
            delta_t = return_mean - supply_mean
            metrics['mean_delta_t_c'] = float(delta_t.mean())
            metrics['max_delta_t_c'] = float(delta_t.max())
        
        # Facility temperatures
        facility_return_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_prim_r_C'
            for i in range(1, num_cdus + 1)
        ]
        available_fac_cols = [col for col in facility_return_cols if col in data.columns]
        
        if available_fac_cols:
            all_fac_temps = data[available_fac_cols]
            metrics['mean_facility_return_temp_c'] = float(all_fac_temps.mean().mean())
            
        return metrics
    
    def compute_flow_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict
    ) -> Dict[str, float]:
        """
        Compute flow rate metrics.
        
        Args:
            data: Simulation data with flow outputs
            system_config: System configuration dictionary
            
        Returns:
            Dictionary of flow metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Secondary flow rates (rack side)
        flow_sec_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.V_flow_sec_GPM'
            for i in range(1, num_cdus + 1)
        ]
        
        available_sec_cols = [col for col in flow_sec_cols if col in data.columns]
        
        if available_sec_cols:
            total_sec_flow = data[available_sec_cols].sum(axis=1)
            metrics['total_sec_flow_gpm'] = float(total_sec_flow.mean())
            metrics['sec_flow_per_cdu_gpm'] = float(total_sec_flow.mean() / num_cdus)
            metrics['sec_flow_std'] = float(total_sec_flow.std())
        
        # Primary flow rates (facility side)
        flow_prim_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.V_flow_prim_GPM'
            for i in range(1, num_cdus + 1)
        ]
        
        available_prim_cols = [col for col in flow_prim_cols if col in data.columns]
        
        if available_prim_cols:
            total_prim_flow = data[available_prim_cols].sum(axis=1)
            metrics['total_prim_flow_gpm'] = float(total_prim_flow.mean())
            metrics['prim_flow_per_cdu_gpm'] = float(total_prim_flow.mean() / num_cdus)
        
        return metrics
    
    def compute_pressure_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict
    ) -> Dict[str, float]:
        """
        Compute pressure differential metrics.
        
        Args:
            data: Simulation data with pressure outputs
            system_config: System configuration dictionary
            
        Returns:
            Dictionary of pressure metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Secondary pressure (rack side)
        p_sec_supply_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.p_sec_s_psig'
            for i in range(1, num_cdus + 1)
        ]
        p_sec_return_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.p_sec_r_psig'
            for i in range(1, num_cdus + 1)
        ]
        
        available_s_cols = [col for col in p_sec_supply_cols if col in data.columns]
        available_r_cols = [col for col in p_sec_return_cols if col in data.columns]
        
        if available_s_cols and available_r_cols:
            supply_mean = data[available_s_cols].mean(axis=1)
            return_mean = data[available_r_cols].mean(axis=1)
            delta_p_sec = supply_mean - return_mean
            metrics['mean_delta_p_sec_psig'] = float(delta_p_sec.mean())
            metrics['max_delta_p_sec_psig'] = float(delta_p_sec.max())
        
        return metrics
    
    def compute_dynamic_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict,
        time_step: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute dynamic response metrics.
        
        Args:
            data: Simulation data
            system_config: System configuration dictionary
            time_step: Time step in seconds
            
        Returns:
            Dictionary of dynamic metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Compute rate of change for key outputs
        temp_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_sec_r_C'
            for i in range(1, num_cdus + 1)
        ]
        
        available_temp_cols = [col for col in temp_cols if col in data.columns]
        
        if available_temp_cols:
            # Average temperature across CDUs
            avg_temp = data[available_temp_cols].mean(axis=1)
            temp_rate = avg_temp.diff() / time_step
            
            metrics['mean_temp_rate_c_per_s'] = float(temp_rate.abs().mean())
            metrics['max_temp_rate_c_per_s'] = float(temp_rate.abs().max())
            
            # Estimate time constant from autocorrelation
            if len(avg_temp) > 100:
                autocorr = pd.Series(avg_temp).autocorr(lag=60)
                if not np.isnan(autocorr) and autocorr > 0:
                    metrics['thermal_time_constant_approx_s'] = float(-60 / np.log(autocorr))
        
        return metrics
    
    def compute_sensitivity_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict
    ) -> Dict[str, float]:
        """
        Compute input-output sensitivity metrics.
        
        Args:
            data: Simulation data
            system_config: System configuration dictionary
            
        Returns:
            Dictionary of sensitivity metrics
        """
        metrics = {}
        num_cdus = system_config.get('NUM_CDUS', 1)
        
        # Get input columns
        q_flow_cols = [
            f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total'
            for i in range(1, num_cdus + 1)
        ]
        
        # Get output columns
        cdup_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.W_flow_CDUP_kW'
            for i in range(1, num_cdus + 1)
        ]
        temp_cols = [
            f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.T_sec_r_C'
            for i in range(1, num_cdus + 1)
        ]
        
        available_q_cols = [col for col in q_flow_cols if col in data.columns]
        available_cdup_cols = [col for col in cdup_cols if col in data.columns]
        available_temp_cols = [col for col in temp_cols if col in data.columns]
        
        if available_q_cols and available_cdup_cols:
            total_q = data[available_q_cols].sum(axis=1)
            total_cdup = data[available_cdup_cols].sum(axis=1)
            corr = stats.pearsonr(total_q, total_cdup)
            metrics['qflow_cdup_correlation'] = float(corr[0])
            
        if available_q_cols and available_temp_cols:
            total_q = data[available_q_cols].sum(axis=1)
            avg_temp = data[available_temp_cols].mean(axis=1)
            corr = stats.pearsonr(total_q, avg_temp)
            metrics['qflow_temp_correlation'] = float(corr[0])
        
        return metrics
    
    def compute_all_metrics(
        self, 
        data: pd.DataFrame,
        system_config: Dict,
        time_step: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a system.
        
        Args:
            data: Simulation data
            system_config: System configuration dictionary
            time_step: Time step in seconds
            
        Returns:
            Dictionary with all metric categories
        """
        all_metrics = {
            'system_info': {
                'num_cdus': system_config.get('NUM_CDUS', 1),
                'nodes_per_rack': system_config.get('NODES_PER_RACK', 0),
                'gpus_per_node': system_config.get('GPUS_PER_NODE', 0),
                'cpus_per_node': system_config.get('CPUS_PER_NODE', 0),
                'cooling_efficiency': system_config.get('COOLING_EFFICIENCY', 0.0)
            },
            'efficiency': self.compute_efficiency_metrics(data, system_config),
            'thermal': self.compute_thermal_metrics(data, system_config),
            'flow': self.compute_flow_metrics(data, system_config),
            'pressure': self.compute_pressure_metrics(data, system_config),
            'dynamic': self.compute_dynamic_metrics(data, system_config, time_step),
            'sensitivity': self.compute_sensitivity_metrics(data, system_config)
        }
        
        return all_metrics
