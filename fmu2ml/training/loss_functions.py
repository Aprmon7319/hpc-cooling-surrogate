import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PhysicsLoss(nn.Module):
    """Enhanced physics-informed loss calculator"""
    
    def __init__(self, norm_handler=None):
        super().__init__()
        self.norm_handler = norm_handler
        
        # Physical constants
        self.WATER_DENSITY = 997  # kg/m³
        self.WATER_SPECIFIC_HEAT = 4186  # J/(kg·K)
        self.GPM_TO_M3_S = 6.30902e-5  # Conversion factor
        self.MIN_APPROACH_TEMP = 2.0  # Minimum approach temperature in °C
        self.epsilon = 1e-6  # Small value to prevent division by zero
        self.MIN_COP = 2.0  # Minimum Coefficient of Performance
        
        # Physics loss weights (tunable)
        self.physics_weights = {
            'temp_ordering': 2.0,
            'approach_temp': 1.5,
            'pue_physics': 1.0,
            'mass_conservation': 1.5,
            'monotonicity': 0.5,
            'thermodynamic_cop': 1.0,
            'cooling_tower_effectiveness': 1.0
        }
    
    def compute_lmtd(self, T_hot_in, T_hot_out, T_cold_in, T_cold_out):
        """Compute Log Mean Temperature Difference with numerical stability"""
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        
        # Ensure positive temperature differences
        dT1 = torch.clamp(dT1, min=self.epsilon)
        dT2 = torch.clamp(dT2, min=self.epsilon)
        
        # Check if temperatures are close
        ratio = dT1 / dT2
        close_temps = torch.abs(ratio - 1) < 0.01
        
        # Calculate LMTD
        lmtd = torch.zeros_like(dT1)
        lmtd[close_temps] = (dT1[close_temps] + dT2[close_temps]) / 2
        lmtd[~close_temps] = (dT1[~close_temps] - dT2[~close_temps]) / torch.log(ratio[~close_temps])
        
        return torch.clamp(lmtd, min=0.1, max=50.0)
    
    def forward(self, outputs, inputs) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss
        
        Returns dict with individual loss components
        """
        # Denormalize if handler available
        if self.norm_handler is not None:
            outputs_denorm = self.norm_handler.denormalize_output(outputs)
            inputs_denorm = self.norm_handler.denormalize_input(inputs)
        else:
            outputs_denorm = outputs
            inputs_denorm = inputs
        
        # Ensure inputs are the right shape
        if inputs_denorm.dim() == 2:
            inputs_denorm = inputs_denorm.unsqueeze(0)
        
        # Initialize loss dictionary
        losses = {}
        
        try:
            # Extract inputs (DENORMALIZED)
            cdu_heat_load_W = inputs_denorm[:, -1, :98:2]  # Q_flow_total
            cdu_air_temp_K = inputs_denorm[:, -1, 1::2]    # T_Air
            external_temp_K = inputs_denorm[:, -1, -1]     # T_ext
            
            # Extract outputs (11 per CDU)
            V_flow_prim_GPM = outputs_denorm[:, 0::11][:, :49]
            T_prim_s_C = outputs_denorm[:, 3::11][:, :49]
            T_prim_r_C = outputs_denorm[:, 4::11][:, :49]
            T_sec_s_C = outputs_denorm[:, 5::11][:, :49]
            T_sec_r_C = outputs_denorm[:, 6::11][:, :49]
            pump_power_kW = outputs_denorm[:, 2::11][:, :49]
            
            # Global outputs (start at 49*11 = 539)
            datacenter_flow = outputs_denorm[:, 539]
            pue = outputs_denorm[:, 540]
            
            # Temperature ordering
            prim_violation = F.relu(T_prim_s_C - T_prim_r_C)
            sec_violation = F.relu(T_sec_s_C - T_sec_r_C)
            external_temp_C = external_temp_K - 273.15
            external_temp_C_expanded = external_temp_C.unsqueeze(1).expand_as(T_prim_s_C)
            cooling_tower_violation = F.relu(external_temp_C_expanded + self.MIN_APPROACH_TEMP - T_prim_s_C)
            losses['temp_ordering'] = (prim_violation.mean() + sec_violation.mean() + 
                                      cooling_tower_violation.mean()) / 3
            
            # Approach temperature
            approach1 = T_sec_r_C - T_prim_s_C
            approach2 = T_sec_s_C - T_prim_r_C
            violation1 = F.relu(self.MIN_APPROACH_TEMP - approach1)
            violation2 = F.relu(self.MIN_APPROACH_TEMP - approach2)
            losses['approach_temp'] = (violation1.mean() + violation2.mean()) / 2
            
            # PUE physics
            total_IT_power_kW = torch.sum(cdu_heat_load_W, dim=1) / 1000
            total_cooling_power_kW = torch.sum(pump_power_kW, dim=1)
            expected_pue = (total_IT_power_kW + total_cooling_power_kW) / (total_IT_power_kW + self.epsilon)
            expected_pue = torch.clamp(expected_pue, min=1.0, max=3.0)
            pue_min_violation = F.relu(1.0 - pue)
            pue_calc_error = torch.abs(pue - expected_pue) / expected_pue
            losses['pue_physics'] = pue_min_violation.mean() + pue_calc_error.mean()
            
            # Mass conservation
            total_cdu_flow = torch.sum(V_flow_prim_GPM, dim=1)
            flow_error = torch.abs(total_cdu_flow - datacenter_flow) / (torch.abs(datacenter_flow) + self.epsilon)
            losses['mass_conservation'] = torch.clamp(flow_error, 0, 1).mean()
            
            # Monotonicity
            monotonicity_scores = []
            batch_size = cdu_heat_load_W.shape[0]
            for b in range(batch_size):
                heat_sorted_idx = torch.argsort(cdu_heat_load_W[b])
                sorted_flows = V_flow_prim_GPM[b][heat_sorted_idx]
                flow_diff = torch.diff(sorted_flows)
                heat_diff = torch.diff(cdu_heat_load_W[b][heat_sorted_idx])
                
                heat_increases = heat_diff > 0
                flow_decreases = flow_diff < -0.1
                violations = heat_increases & flow_decreases
                violation_score = violations.float().sum() / len(heat_diff)
                monotonicity_scores.append(violation_score)
            losses['monotonicity'] = torch.stack(monotonicity_scores).mean()
            
            # Thermodynamic COP
            total_cooling_W = torch.sum(cdu_heat_load_W, dim=1)
            total_pump_work_W = torch.sum(pump_power_kW, dim=1) * 1000
            T_cold_K = torch.mean(T_prim_s_C + 273.15, dim=1)
            T_hot_K = torch.mean(cdu_air_temp_K, dim=1)
            carnot_cop = T_cold_K / (T_hot_K - T_cold_K + self.epsilon)
            realistic_cop = 0.5 * carnot_cop
            expected_total_work = total_cooling_W / torch.clamp(realistic_cop, min=self.MIN_COP)
            actual_cop = total_cooling_W / (total_pump_work_W + expected_total_work * 0.1 + self.epsilon)
            cop_bounds_penalty = F.relu(self.MIN_COP - actual_cop).mean() + F.relu(actual_cop - 10.0).mean()
            losses['thermodynamic_cop'] = cop_bounds_penalty
            
            # Cooling tower effectiveness
            cooling_tower_approach = T_prim_s_C - external_temp_C.unsqueeze(1).expand_as(T_prim_s_C)
            ct_approach_low = F.relu(self.MIN_APPROACH_TEMP - cooling_tower_approach).mean()
            ct_approach_high = F.relu(cooling_tower_approach - 10.0).mean()
            effectiveness = (T_prim_r_C - T_prim_s_C) / (T_prim_r_C - external_temp_C.unsqueeze(1) + self.epsilon)
            eff_bounds_penalty = F.relu(0.6 - effectiveness).mean() + F.relu(effectiveness - 0.95).mean()
            losses['cooling_tower_effectiveness'] = (ct_approach_low + ct_approach_high + eff_bounds_penalty) / 3
            
            # Calculate total weighted physics loss
            total_loss = sum(self.physics_weights[k] * losses[k] for k in losses if k in self.physics_weights)
            losses['total'] = total_loss
            
        except Exception as e:
            print(f"Error in physics loss calculation: {e}")
            # Return default values on error
            for key in self.physics_weights.keys():
                losses[key] = torch.tensor(0.0, device=outputs.device)
            losses['total'] = torch.tensor(0.0, device=outputs.device)
        
        return losses


class CombinedLoss(nn.Module):
    """Combined loss function with MSE, MAE, and physics"""
    
    def __init__(self, norm_handler=None, physics_weight: float = 0.3):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.physics_loss = PhysicsLoss(norm_handler)
        self.physics_weight = physics_weight
    
    def forward(self, outputs, targets, inputs):
        """
        Compute combined loss
        
        Returns total loss and dict of components
        """
        mse = self.mse_loss(outputs, targets)
        mae = self.mae_loss(outputs, targets)
        physics_losses = self.physics_loss(outputs, inputs)
        
        total_loss = mse + 0.1 * mae + self.physics_weight * physics_losses['total']
        
        return total_loss, {
            'mse': mse,
            'mae': mae,
            'physics': physics_losses,
            'total': total_loss
        }
