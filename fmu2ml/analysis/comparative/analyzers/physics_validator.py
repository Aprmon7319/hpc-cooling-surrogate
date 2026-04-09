"""
Physics Constraint Validator for Cooling Model Analysis.

Evaluates whether physics constraints hold across different cooling models.
This is a numpy-based version for FMU output analysis (not training).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PhysicsValidatorConfig:
    """Configuration for physics validation."""
    
    # Physical constants
    water_density: float = 997  # kg/m³
    water_specific_heat: float = 4186  # J/(kg·K)
    gpm_to_m3_s: float = 6.30902e-5  # Conversion factor
    
    # Constraint thresholds
    min_approach_temp: float = 2.0  # Minimum approach temperature in °C
    min_cop: float = 2.0  # Minimum Coefficient of Performance
    max_cop: float = 10.0  # Maximum realistic COP
    min_pue: float = 1.0  # Minimum PUE (theoretical limit)
    max_pue: float = 3.0  # Maximum reasonable PUE
    
    # Tolerance for mass conservation
    mass_conservation_tolerance: float = 0.05  # 5% tolerance


class PhysicsConstraintValidator:
    """
    Validates physics constraints for cooling model outputs.
    
    Evaluates:
    1. Temperature ordering (supply < return in primary, etc.)
    2. Approach temperature constraints
    3. Mass/flow conservation
    4. PUE bounds and consistency
    5. Energy balance
    6. Heat exchanger effectiveness
    7. Monotonicity (higher load → higher flow)
    """
    
    def __init__(self, config: PhysicsValidatorConfig = None, num_cdus: int = 49):
        """Initialize validator."""
        self.config = config or PhysicsValidatorConfig()
        self.num_cdus = num_cdus
        self.epsilon = 1e-6
    
    def _get_column(self, df: pd.DataFrame, pattern: str, cdu_id: int = None) -> Optional[np.ndarray]:
        """Get column data by pattern matching."""
        if cdu_id is not None:
            pattern = pattern.format(cdu_id)
        
        # Try exact match first
        if pattern in df.columns:
            return df[pattern].values
        
        # Try partial match
        matches = [c for c in df.columns if pattern in c]
        if matches:
            return df[matches[0]].values
        
        return None
    
    def validate_temperature_ordering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate temperature ordering constraints.
        
        Physics:
        - Primary supply temp < Primary return temp (water heats up)
        - Secondary supply temp < Secondary return temp
        - Primary supply temp > External temp + approach
        """
        results = {
            'constraint': 'temperature_ordering',
            'passed': True,
            'violations': {},
            'statistics': {}
        }
        
        for cdu_id in range(1, min(self.num_cdus + 1, 50)):  # Limit to first 50 CDUs
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            
            T_prim_s = self._get_column(df, f'{prefix}.T_prim_s_C')
            T_prim_r = self._get_column(df, f'{prefix}.T_prim_r_C')
            T_sec_s = self._get_column(df, f'{prefix}.T_sec_s_C')
            T_sec_r = self._get_column(df, f'{prefix}.T_sec_r_C')
            
            if T_prim_s is None or T_prim_r is None:
                continue
            
            # Check primary: supply should be cooler than return
            prim_violations = np.sum(T_prim_s > T_prim_r + 0.1)
            if prim_violations > 0:
                results['violations'][f'cdu_{cdu_id}_primary'] = {
                    'count': int(prim_violations),
                    'rate': float(prim_violations / len(T_prim_s))
                }
                results['passed'] = False
            
            # Check secondary if available
            if T_sec_s is not None and T_sec_r is not None:
                sec_violations = np.sum(T_sec_s > T_sec_r + 0.1)
                if sec_violations > 0:
                    results['violations'][f'cdu_{cdu_id}_secondary'] = {
                        'count': int(sec_violations),
                        'rate': float(sec_violations / len(T_sec_s))
                    }
                    results['passed'] = False
        
        # Calculate overall statistics
        if results['violations']:
            total_violations = sum(v['count'] for v in results['violations'].values())
            results['statistics']['total_violations'] = total_violations
            results['statistics']['avg_violation_rate'] = np.mean(
                [v['rate'] for v in results['violations'].values()]
            )
        else:
            results['statistics']['total_violations'] = 0
            results['statistics']['avg_violation_rate'] = 0.0
        
        return results
    
    def validate_approach_temperature(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate approach temperature constraints.
        
        The approach temperature is the minimum temperature difference
        in a heat exchanger. For CDUs, this is typically 2-5°C.
        """
        results = {
            'constraint': 'approach_temperature',
            'passed': True,
            'violations': {},
            'statistics': {'approach_temps': []}
        }
        
        for cdu_id in range(1, min(self.num_cdus + 1, 50)):
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            
            T_prim_s = self._get_column(df, f'{prefix}.T_prim_s_C')
            T_prim_r = self._get_column(df, f'{prefix}.T_prim_r_C')
            T_sec_s = self._get_column(df, f'{prefix}.T_sec_s_C')
            T_sec_r = self._get_column(df, f'{prefix}.T_sec_r_C')
            
            if T_prim_s is None or T_sec_r is None:
                continue
            
            # Approach 1: T_sec_r - T_prim_s (cold side approach)
            approach1 = T_sec_r - T_prim_s
            violations1 = np.sum(approach1 < self.config.min_approach_temp)
            
            if violations1 > 0:
                results['violations'][f'cdu_{cdu_id}_approach1'] = {
                    'count': int(violations1),
                    'rate': float(violations1 / len(approach1)),
                    'min_approach': float(np.min(approach1))
                }
                results['passed'] = False
            
            results['statistics']['approach_temps'].append(float(np.mean(approach1)))
        
        if results['statistics']['approach_temps']:
            results['statistics']['mean_approach'] = float(
                np.mean(results['statistics']['approach_temps'])
            )
        
        return results
    
    def validate_mass_conservation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate mass/flow conservation.
        
        Sum of individual CDU flows should equal datacenter total flow.
        """
        results = {
            'constraint': 'mass_conservation',
            'passed': True,
            'violations': {},
            'statistics': {}
        }
        
        # Get datacenter total flow
        dc_flow = self._get_column(df, 'simulator[1].datacenter[1].summary.V_flow_prim_GPM')
        if dc_flow is None:
            results['passed'] = None
            results['statistics']['error'] = 'Datacenter flow column not found'
            return results
        
        # Sum individual CDU flows
        cdu_flows = []
        for cdu_id in range(1, self.num_cdus + 1):
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            flow = self._get_column(df, f'{prefix}.V_flow_prim_GPM')
            if flow is not None:
                cdu_flows.append(flow)
        
        if not cdu_flows:
            results['passed'] = None
            results['statistics']['error'] = 'No CDU flow columns found'
            return results
        
        total_cdu_flow = np.sum(cdu_flows, axis=0)
        
        # Calculate relative error
        relative_error = np.abs(total_cdu_flow - dc_flow) / (np.abs(dc_flow) + self.epsilon)
        
        violations = np.sum(relative_error > self.config.mass_conservation_tolerance)
        if violations > 0:
            results['passed'] = False
            results['violations']['flow_mismatch'] = {
                'count': int(violations),
                'rate': float(violations / len(relative_error)),
                'max_error': float(np.max(relative_error)),
                'mean_error': float(np.mean(relative_error))
            }
        
        results['statistics']['mean_relative_error'] = float(np.mean(relative_error))
        results['statistics']['max_relative_error'] = float(np.max(relative_error))
        results['statistics']['n_cdus_found'] = len(cdu_flows)
        
        return results
    
    def validate_pue_bounds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate PUE is within physical bounds.
        
        PUE = Total Facility Power / IT Equipment Power
        Must be >= 1.0 (theoretical minimum)
        """
        results = {
            'constraint': 'pue_bounds',
            'passed': True,
            'violations': {},
            'statistics': {}
        }
        
        pue = self._get_column(df, 'pue')
        if pue is None:
            results['passed'] = None
            results['statistics']['error'] = 'PUE column not found'
            return results
        
        # Check lower bound
        below_min = np.sum(pue < self.config.min_pue)
        if below_min > 0:
            results['passed'] = False
            results['violations']['below_minimum'] = {
                'count': int(below_min),
                'rate': float(below_min / len(pue)),
                'min_value': float(np.min(pue))
            }
        
        # Check upper bound
        above_max = np.sum(pue > self.config.max_pue)
        if above_max > 0:
            results['violations']['above_maximum'] = {
                'count': int(above_max),
                'rate': float(above_max / len(pue)),
                'max_value': float(np.max(pue))
            }
        
        results['statistics']['mean_pue'] = float(np.mean(pue))
        results['statistics']['std_pue'] = float(np.std(pue))
        results['statistics']['min_pue'] = float(np.min(pue))
        results['statistics']['max_pue'] = float(np.max(pue))
        
        return results
    
    def validate_energy_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate energy balance at heat exchangers.
        
        Q = m_dot * Cp * delta_T
        Primary side heat rejection should approximately equal secondary side heat absorption.
        """
        results = {
            'constraint': 'energy_balance',
            'passed': True,
            'violations': {},
            'statistics': {'energy_balances': []}
        }
        
        for cdu_id in range(1, min(self.num_cdus + 1, 50)):
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            
            V_prim = self._get_column(df, f'{prefix}.V_flow_prim_GPM')
            T_prim_s = self._get_column(df, f'{prefix}.T_prim_s_C')
            T_prim_r = self._get_column(df, f'{prefix}.T_prim_r_C')
            V_sec = self._get_column(df, f'{prefix}.V_flow_sec_GPM')
            T_sec_s = self._get_column(df, f'{prefix}.T_sec_s_C')
            T_sec_r = self._get_column(df, f'{prefix}.T_sec_r_C')
            
            if V_prim is None or T_prim_s is None or T_prim_r is None:
                continue
            
            # Convert GPM to m³/s
            m_dot_prim = V_prim * self.config.gpm_to_m3_s * self.config.water_density
            
            # Primary side heat (water cools down: T_r > T_s)
            Q_prim = m_dot_prim * self.config.water_specific_heat * (T_prim_r - T_prim_s)
            
            if V_sec is not None and T_sec_s is not None and T_sec_r is not None:
                m_dot_sec = V_sec * self.config.gpm_to_m3_s * self.config.water_density
                # Secondary side heat (water heats up: T_r > T_s)
                Q_sec = m_dot_sec * self.config.water_specific_heat * (T_sec_r - T_sec_s)
                
                # Energy balance: Q_prim ≈ Q_sec
                energy_diff = np.abs(Q_prim - Q_sec) / (np.abs(Q_prim) + self.epsilon)
                mean_diff = float(np.mean(energy_diff))
                results['statistics']['energy_balances'].append(mean_diff)
                
                if mean_diff > 0.2:  # 20% tolerance
                    results['violations'][f'cdu_{cdu_id}'] = {
                        'mean_imbalance': mean_diff,
                        'max_imbalance': float(np.max(energy_diff))
                    }
                    results['passed'] = False
        
        if results['statistics']['energy_balances']:
            results['statistics']['mean_imbalance'] = float(
                np.mean(results['statistics']['energy_balances'])
            )
        
        return results
    
    def validate_monotonicity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate monotonic relationship between heat load and flow.
        
        Higher heat load should generally result in higher cooling flow.
        """
        results = {
            'constraint': 'monotonicity',
            'passed': True,
            'violations': {},
            'statistics': {}
        }
        
        # Collect heat loads and flows
        heat_loads = []
        flows = []
        
        for cdu_id in range(1, min(self.num_cdus + 1, 50)):
            # Heat load input
            q_col = f'simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total'
            q_data = self._get_column(df, q_col)
            
            # Flow output
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            flow = self._get_column(df, f'{prefix}.V_flow_prim_GPM')
            
            if q_data is not None and flow is not None:
                heat_loads.append(np.mean(q_data))
                flows.append(np.mean(flow))
        
        if len(heat_loads) < 3:
            results['passed'] = None
            results['statistics']['error'] = 'Insufficient data for monotonicity check'
            return results
        
        heat_loads = np.array(heat_loads)
        flows = np.array(flows)
        
        # Sort by heat load and check flow monotonicity
        sort_idx = np.argsort(heat_loads)
        sorted_flows = flows[sort_idx]
        
        # Count violations (flow decreases when heat increases)
        flow_diff = np.diff(sorted_flows)
        violations = np.sum(flow_diff < -0.5)  # Allow small tolerance
        
        if violations > len(flow_diff) * 0.2:  # More than 20% violations
            results['passed'] = False
            results['violations']['flow_monotonicity'] = {
                'count': int(violations),
                'rate': float(violations / len(flow_diff))
            }
        
        # Calculate correlation
        correlation = np.corrcoef(heat_loads, flows)[0, 1]
        results['statistics']['heat_flow_correlation'] = float(correlation)
        results['statistics']['violation_rate'] = float(violations / len(flow_diff)) if len(flow_diff) > 0 else 0.0
        
        return results
    
    def validate_heat_exchanger_effectiveness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate heat exchanger effectiveness is within physical bounds.
        
        Effectiveness = (actual heat transfer) / (max possible heat transfer)
        Should be between 0 and 1 (typically 0.6-0.95 for CDU heat exchangers).
        """
        results = {
            'constraint': 'hx_effectiveness',
            'passed': True,
            'violations': {},
            'statistics': {'effectiveness_values': []}
        }
        
        for cdu_id in range(1, min(self.num_cdus + 1, 50)):
            prefix = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary'
            
            T_prim_s = self._get_column(df, f'{prefix}.T_prim_s_C')
            T_prim_r = self._get_column(df, f'{prefix}.T_prim_r_C')
            T_sec_s = self._get_column(df, f'{prefix}.T_sec_s_C')
            T_sec_r = self._get_column(df, f'{prefix}.T_sec_r_C')
            
            if T_prim_s is None or T_prim_r is None or T_sec_s is None:
                continue
            
            # Effectiveness based on primary side
            # ε = (T_prim_r - T_prim_s) / (T_sec_r - T_prim_s)
            delta_T_max = T_sec_r - T_prim_s + self.epsilon
            delta_T_actual = T_prim_r - T_prim_s
            
            effectiveness = delta_T_actual / delta_T_max
            
            # Filter valid values
            valid_mask = (delta_T_max > 0.5) & (delta_T_actual > 0)
            if np.sum(valid_mask) > 0:
                eff_valid = effectiveness[valid_mask]
                mean_eff = float(np.mean(eff_valid))
                results['statistics']['effectiveness_values'].append(mean_eff)
                
                # Check bounds
                below_min = np.sum(eff_valid < 0.3)
                above_max = np.sum(eff_valid > 1.0)
                
                if below_min > len(eff_valid) * 0.1 or above_max > 0:
                    results['violations'][f'cdu_{cdu_id}'] = {
                        'mean_effectiveness': mean_eff,
                        'below_min_rate': float(below_min / len(eff_valid)),
                        'above_max_rate': float(above_max / len(eff_valid))
                    }
                    results['passed'] = False
        
        if results['statistics']['effectiveness_values']:
            results['statistics']['mean_effectiveness'] = float(
                np.mean(results['statistics']['effectiveness_values'])
            )
            results['statistics']['std_effectiveness'] = float(
                np.std(results['statistics']['effectiveness_values'])
            )
        
        return results
    
    def validate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all physics validations and return summary.
        """
        results = {
            'temperature_ordering': self.validate_temperature_ordering(df),
            'approach_temperature': self.validate_approach_temperature(df),
            'mass_conservation': self.validate_mass_conservation(df),
            'pue_bounds': self.validate_pue_bounds(df),
            'energy_balance': self.validate_energy_balance(df),
            'monotonicity': self.validate_monotonicity(df),
            'hx_effectiveness': self.validate_heat_exchanger_effectiveness(df)
        }
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.get('passed') is True)
        failed_count = sum(1 for r in results.values() if r.get('passed') is False)
        skipped_count = sum(1 for r in results.values() if r.get('passed') is None)
        
        results['summary'] = {
            'total_constraints': len(results) - 1,  # Exclude summary itself
            'passed': passed_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'pass_rate': passed_count / (passed_count + failed_count) if (passed_count + failed_count) > 0 else 0.0
        }
        
        return results
    
    def compare_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compare physics constraint adherence across multiple models.
        
        Args:
            data_dict: Dict mapping model name to DataFrame
            
        Returns:
            Comparison results
        """
        comparison = {
            'models': list(data_dict.keys()),
            'by_model': {},
            'by_constraint': {},
            'summary_table': []
        }
        
        constraint_names = [
            'temperature_ordering', 'approach_temperature', 'mass_conservation',
            'pue_bounds', 'energy_balance', 'monotonicity', 'hx_effectiveness'
        ]
        
        # Validate each model
        for model_name, df in data_dict.items():
            logger.info(f"Validating physics constraints for {model_name}...")
            
            # Update num_cdus based on model
            if 'summit' in model_name.lower() or 'lassen' in model_name.lower():
                self.num_cdus = 257
            elif 'frontier' in model_name.lower():
                self.num_cdus = 128
            else:
                self.num_cdus = 49
            
            validation_results = self.validate_all(df)
            comparison['by_model'][model_name] = validation_results
        
        # Organize by constraint for easy comparison
        for constraint in constraint_names:
            comparison['by_constraint'][constraint] = {}
            for model_name in data_dict.keys():
                model_result = comparison['by_model'][model_name].get(constraint, {})
                comparison['by_constraint'][constraint][model_name] = {
                    'passed': model_result.get('passed'),
                    'statistics': model_result.get('statistics', {})
                }
        
        # Create summary table
        for model_name in data_dict.keys():
            row = {'model': model_name}
            for constraint in constraint_names:
                result = comparison['by_model'][model_name].get(constraint, {})
                row[constraint] = '✓' if result.get('passed') else ('?' if result.get('passed') is None else '✗')
            row['pass_rate'] = comparison['by_model'][model_name].get('summary', {}).get('pass_rate', 0)
            comparison['summary_table'].append(row)
        
        return comparison
