"""
Physics-based validation for model outputs.
"""
import numpy as np
import torch
from typing import Dict, List


class PhysicsValidator:
    """Validate physics constraints in model outputs"""
    
    def __init__(self):
        self.WATER_DENSITY = 997  # kg/m³
        self.WATER_SPECIFIC_HEAT = 4186  # J/(kg·K)
        self.GPM_TO_M3_S = 6.30902e-5
        self.MIN_APPROACH_TEMP = 2.0
        self.epsilon = 1e-6
    
    def validate_outputs(self, outputs: np.ndarray, inputs: np.ndarray) -> Dict[str, List[float]]:
        """
        Validate physics constraints
        
        Returns dict with violation scores for each constraint
        """
        violations = {
            'temp_ordering': [],
            'approach_temp': [],
            'mass_conservation': [],
            'energy_balance': []
        }
        
        for i in range(outputs.shape[0]):
            output = outputs[i]
            input_data = inputs[i]
            
            # Extract variables
            T_prim_s = output[3::11][:49]
            T_prim_r = output[4::11][:49]
            T_sec_s = output[5::11][:49]
            T_sec_r = output[6::11][:49]
            V_flow_prim = output[0::11][:49]
            datacenter_flow = output[539]
            
            # Temperature ordering
            temp_viol = np.mean(np.maximum(0, T_prim_s - T_prim_r))
            violations['temp_ordering'].append(float(temp_viol))
            
            # Approach temperature
            approach1 = T_sec_r - T_prim_s
            approach2 = T_sec_s - T_prim_r
            approach_viol = np.mean(np.maximum(0, self.MIN_APPROACH_TEMP - approach1))
            approach_viol += np.mean(np.maximum(0, self.MIN_APPROACH_TEMP - approach2))
            violations['approach_temp'].append(float(approach_viol / 2))
            
            # Mass conservation
            total_cdu_flow = np.sum(V_flow_prim)
            flow_error = np.abs(total_cdu_flow - datacenter_flow) / (np.abs(datacenter_flow) + self.epsilon)
            violations['mass_conservation'].append(float(flow_error))
        
        return violations
    
    def print_summary(self, violations: Dict[str, List[float]]):
        """Print validation summary"""
        print("" + "="*60)
        print("PHYSICS VALIDATION SUMMARY")
        print("="*60)
        
        for key, values in violations.items():
            mean_val = np.mean(values)
            max_val = np.max(values)
            print(f"{key:<25} Mean: {mean_val:.4f}  Max: {max_val:.4f}")
        
        print("="*60)
