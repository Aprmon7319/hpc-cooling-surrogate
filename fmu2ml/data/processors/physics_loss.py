import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import json
import traceback
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

# Import your modules
from fmu2ml.config import ModelConfig
from fmu2ml.data.processors import DatacenterCoolingDataset, create_data_loaders

# Add these imports for parallel processing
import torch.multiprocessing as mp
import multiprocessing as cpu_mp
from functools import partial


class PhysicsValidator:
    """
    Physics-based validator for datacenter cooling system
    Works with both real FMU data and ML model predictions
    """
    
    def __init__(self, config: Optional[Dict] = None, device=None, system_name: str = 'marconi100'):
        """
        Initialize physics validator
        
        Parameters:
        -----------
        config : Dict, optional
            System configuration (NUM_CDUS, etc.)
        device : torch.device, optional
            Device for tensor operations
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # System configuration
        if config is None:
            from raps.config import ConfigManager
            config = ConfigManager(system_name=system_name).get_config()
        
        self.config = config
        self.num_cdus = config.get('NUM_CDUS', 49)
        
        # Physical constants
        self.WATER_DENSITY = 997  # kg/m³
        self.WATER_SPECIFIC_HEAT = 4186  # J/(kg·K)
        self.GPM_TO_M3_S = 6.30902e-5  # conversion factor
        self.MIN_APPROACH_TEMP = 2.0  # Minimum approach temperature in °C
        self.epsilon = 1e-6
        self.MIN_COP = 2.0  # Minimum Coefficient of Performance
        
        # Physics loss weights (tunable)
        self.physics_weights = {
            'temp_ordering': 2.0,
            'approach_temp': 1.5,
            'pue_physics': 1.0,
            'mass_conservation': 1.5,
            'energy_balance': 1.5,
            'monotonicity': 0.5,
            'thermodynamic_cop': 1.0,
            'cooling_tower_effectiveness': 1.0
        }
        
        self.logger.info(f"Physics validator initialized for {self.num_cdus} CDUs")
    
    def extract_variables(
        self, 
        outputs: torch.Tensor, 
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract variables from model outputs and inputs
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model outputs [batch, 590] containing CDU outputs + datacenter metrics
        inputs : torch.Tensor  
            Model inputs [batch, seq_len, 99] containing power and temperatures
        
        Returns:
        --------
        Dict with extracted variables
        """
        # Ensure correct dimensions
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        
        batch_size = outputs.shape[0]
        
        # Extract from inputs (last timestep)
        cdu_heat_load_W = inputs[:, -1, :self.num_cdus*2:2]  # Power (every 2nd from 0)
        cdu_air_temp_K = inputs[:, -1, 1:self.num_cdus*2:2]  # Temp (every 2nd from 1)
        external_temp_K = inputs[:, -1, -1]  # Last element
        
        # Convert to Celsius
        cdu_air_temp_C = cdu_air_temp_K - 273.15
        external_temp_C = external_temp_K - 273.15
        
        # Extract from outputs (11 parameters per CDU + 2 datacenter + 49 HTC)
        vars_dict = {}
        
        # CDU outputs (11 vars × 49 CDUs = 539 values)
        vars_dict['V_flow_prim_GPM'] = outputs[:, 0::11][:, :self.num_cdus]
        vars_dict['V_flow_sec_GPM'] = outputs[:, 1::11][:, :self.num_cdus]
        vars_dict['pump_power_kW'] = outputs[:, 2::11][:, :self.num_cdus]
        vars_dict['T_prim_s_C'] = outputs[:, 3::11][:, :self.num_cdus]
        vars_dict['T_prim_r_C'] = outputs[:, 4::11][:, :self.num_cdus]
        vars_dict['T_sec_s_C'] = outputs[:, 5::11][:, :self.num_cdus]
        vars_dict['T_sec_r_C'] = outputs[:, 6::11][:, :self.num_cdus]
        vars_dict['p_prim_s_psig'] = outputs[:, 7::11][:, :self.num_cdus]
        vars_dict['p_prim_r_psig'] = outputs[:, 8::11][:, :self.num_cdus]
        vars_dict['p_sec_s_psig'] = outputs[:, 9::11][:, :self.num_cdus]
        vars_dict['p_sec_r_psig'] = outputs[:, 10::11][:, :self.num_cdus]
        
        # Datacenter level (2 values)
        vars_dict['datacenter_flow_GPM'] = outputs[:, 11*self.num_cdus]
        vars_dict['pue'] = outputs[:, 11*self.num_cdus+1]
        
        # HTC values (49 values)
        vars_dict['htc'] = outputs[:, 11*self.num_cdus+2:12*self.num_cdus+2]
        
        # Input variables
        vars_dict['cdu_heat_load_W'] = cdu_heat_load_W
        vars_dict['cdu_air_temp_C'] = cdu_air_temp_C
        vars_dict['external_temp_C'] = external_temp_C
        
        return vars_dict
    
    def compute_physics_loss(
        self, 
        outputs: torch.Tensor, 
        inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute comprehensive physics-informed loss
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model outputs [batch, 590]
        inputs : torch.Tensor
            Model inputs [batch, seq_len, 99]
        
        Returns:
        --------
        Dict with individual and total physics losses
        """
        try:
            vars_dict = self.extract_variables(outputs, inputs)
            physics_losses = {}
            
            # 1. Temperature Ordering
            physics_losses['temp_ordering'] = self._loss_temperature_ordering(vars_dict)
            
            # 2. Approach Temperature
            physics_losses['approach_temp'] = self._loss_approach_temperature(vars_dict)
            
            # 3. PUE Physics
            physics_losses['pue_physics'] = self._loss_pue_physics(vars_dict)
            
            # 4. Mass Conservation
            physics_losses['mass_conservation'] = self._loss_mass_conservation(vars_dict)
            
            # 5. Energy Balance
            physics_losses['energy_balance'] = self._loss_energy_balance(vars_dict)
            
            # 6. Monotonicity
            physics_losses['monotonicity'] = self._loss_monotonicity(vars_dict)
            
            # 7. Thermodynamic COP
            physics_losses['thermodynamic_cop'] = self._loss_thermodynamic_cop(vars_dict)
            
            # 8. Cooling Tower Effectiveness
            physics_losses['cooling_tower_effectiveness'] = self._loss_cooling_tower_effectiveness(vars_dict)
            
            # Calculate total weighted loss
            total_loss = sum(
                self.physics_weights[k] * physics_losses[k] 
                for k in physics_losses if k != 'total'
            )
            physics_losses['total'] = float(total_loss)
            
        except Exception as e:
            self.logger.error(f"Error in physics loss calculation: {e}")
            traceback.print_exc()
            physics_losses = {k: 0.0 for k in self.physics_weights.keys()}
            physics_losses['total'] = 0.0
        
        return physics_losses
    
    def analyze_physics_violations(self, outputs: torch.Tensor, inputs: torch.Tensor) -> Dict:
        """
        Detailed analysis of physics violations
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model outputs
        inputs : torch.Tensor
            Model inputs
            
        Returns:
        --------
        Dict with detailed violation analysis
        """
        vars_dict = self.extract_variables(outputs, inputs)
        
        analysis = {
            'temp_ordering': {},
            'approach_temp': {},
            'pue': {},
            'mass_conservation': {},
            'monotonicity': {},
            'energy_balance': {},
            'cooling_tower': {}
        }
        
        try:
            # Temperature Ordering Analysis
            prim_violations = (vars_dict['T_prim_s_C'] > vars_dict['T_prim_r_C']).float()
            sec_violations = (vars_dict['T_sec_s_C'] > vars_dict['T_sec_r_C']).float()
            external_temp_C = vars_dict['external_temp_C'].unsqueeze(1).expand_as(vars_dict['T_prim_s_C'])
            cooling_tower_violations = (vars_dict['T_prim_s_C'] < external_temp_C + self.MIN_APPROACH_TEMP).float()
            
            analysis['temp_ordering'] = {
                'primary_violations': prim_violations.sum(dim=1).detach().cpu().numpy(),
                'secondary_violations': sec_violations.sum(dim=1).detach().cpu().numpy(),
                'cooling_tower_violations': cooling_tower_violations.sum(dim=1).detach().cpu().numpy(),
                'total_violation_rate': (prim_violations.sum() + sec_violations.sum() + 
                                        cooling_tower_violations.sum()).item() / (3 * self.num_cdus * inputs.shape[0])
            }
            
            # Approach Temperature Analysis
            approach1 = vars_dict['T_sec_r_C'] - vars_dict['T_prim_s_C']
            approach2 = vars_dict['T_sec_s_C'] - vars_dict['T_prim_r_C']
            approach1_violations = (approach1 < self.MIN_APPROACH_TEMP).float()
            approach2_violations = (approach2 < self.MIN_APPROACH_TEMP).float()
            
            analysis['approach_temp'] = {
                'approach1_temps': approach1.detach().cpu().numpy(),
                'approach2_temps': approach2.detach().cpu().numpy(),
                'approach1_violations': approach1_violations.sum(dim=1).detach().cpu().numpy(),
                'approach2_violations': approach2_violations.sum(dim=1).detach().cpu().numpy(),
                'violation_rate': (approach1_violations.sum() + approach2_violations.sum()).item() / (2 * self.num_cdus * inputs.shape[0])
            }
            
            # PUE Analysis
            total_IT_power_kW = torch.sum(vars_dict['cdu_heat_load_W'], dim=1) / 1000
            total_cooling_power_kW = torch.sum(vars_dict['pump_power_kW'], dim=1)
            expected_pue = (total_IT_power_kW + total_cooling_power_kW) / (total_IT_power_kW + self.epsilon)
            
            analysis['pue'] = {
                'actual_pue': vars_dict['pue'].detach().cpu().numpy(),
                'expected_pue': expected_pue.detach().cpu().numpy(),
                'pue_error': torch.abs(vars_dict['pue'] - expected_pue).detach().cpu().numpy(),
                'below_1_violations': (vars_dict['pue'] < 1.0).float().mean().item(),
                'IT_power_kW': total_IT_power_kW.detach().cpu().numpy(),
                'cooling_power_kW': total_cooling_power_kW.detach().cpu().numpy()
            }
            
            # Mass Conservation Analysis
            total_cdu_flow = torch.sum(vars_dict['V_flow_prim_GPM'], dim=1)
            flow_imbalance = torch.abs(total_cdu_flow - vars_dict['datacenter_flow_GPM'])
            
            analysis['mass_conservation'] = {
                'total_cdu_flow': total_cdu_flow.detach().cpu().numpy(),
                'datacenter_flow': vars_dict['datacenter_flow_GPM'].detach().cpu().numpy(),
                'flow_imbalance': flow_imbalance.detach().cpu().numpy(),
                'relative_error': (flow_imbalance / (vars_dict['datacenter_flow_GPM'] + self.epsilon)).detach().cpu().numpy()
            }
            
            # Energy Balance Analysis
            energy_errors = []
            for b in range(inputs.shape[0]):
                for cdu_idx in range(self.num_cdus):
                    Q_load = vars_dict['cdu_heat_load_W'][b, cdu_idx]
                    V_flow_sec = vars_dict['V_flow_sec_GPM'][b, cdu_idx] * self.GPM_TO_M3_S
                    delta_T_sec = vars_dict['T_sec_r_C'][b, cdu_idx] - vars_dict['T_sec_s_C'][b, cdu_idx]
                    Q_removed = (self.WATER_DENSITY * V_flow_sec * self.WATER_SPECIFIC_HEAT * torch.abs(delta_T_sec))
                    energy_error = torch.abs(Q_load - Q_removed) / (Q_load + self.epsilon)
                    energy_errors.append(energy_error.item())
            
            analysis['energy_balance'] = {
                'mean_error': np.mean(energy_errors),
                'max_error': np.max(energy_errors),
                'violation_rate': np.mean([e > 0.1 for e in energy_errors])
            }
            
        except Exception as e:
            self.logger.error(f"Error in physics analysis: {e}")
            traceback.print_exc()
        
        return analysis
    
    def _loss_temperature_ordering(self, vars_dict: Dict) -> float:
        """T_prim_r > T_prim_s and T_sec_r > T_sec_s"""
        # Primary loop: return > supply
        prim_violation = F.relu(vars_dict['T_prim_s_C'] - vars_dict['T_prim_r_C'])
        
        # Secondary loop: return > supply
        sec_violation = F.relu(vars_dict['T_sec_s_C'] - vars_dict['T_sec_r_C'])
        
        # Cooling tower: primary supply should be above external + approach
        external_temp_expanded = vars_dict['external_temp_C'].unsqueeze(1).expand_as(
            vars_dict['T_prim_s_C']
        )
        ct_violation = F.relu(
            external_temp_expanded + self.MIN_APPROACH_TEMP - vars_dict['T_prim_s_C']
        )
        
        return ((prim_violation.mean() + sec_violation.mean() + ct_violation.mean()) / 3).item()
    
    def _loss_approach_temperature(self, vars_dict: Dict) -> float:
        """Minimum approach temperature between primary and secondary loops"""
        # Heat exchanger approach temperatures
        approach1 = vars_dict['T_sec_r_C'] - vars_dict['T_prim_s_C']
        approach2 = vars_dict['T_sec_s_C'] - vars_dict['T_prim_r_C']
        
        violation1 = F.relu(self.MIN_APPROACH_TEMP - approach1)
        violation2 = F.relu(self.MIN_APPROACH_TEMP - approach2)
        
        return ((violation1.mean() + violation2.mean()) / 2).item()
    
    def _loss_pue_physics(self, vars_dict: Dict) -> float:
        """PUE should be >= 1.0 and match calculated value"""
        pue = vars_dict['pue']
        
        # PUE must be >= 1.0
        pue_min_violation = F.relu(1.0 - pue)
        
        # Calculate expected PUE
        total_IT_power_kW = torch.sum(vars_dict['cdu_heat_load_W'], dim=1) / 1000
        total_cooling_power_kW = torch.sum(vars_dict['pump_power_kW'], dim=1)
        
        expected_pue = (total_IT_power_kW + total_cooling_power_kW) / (
            total_IT_power_kW + self.epsilon
        )
        expected_pue = torch.clamp(expected_pue, min=1.0, max=3.0)
        
        # Relative error
        pue_calc_error = torch.abs(pue - expected_pue) / (expected_pue + self.epsilon)
        
        return (pue_min_violation.mean() + pue_calc_error.mean()).item()
    
    def _loss_mass_conservation(self, vars_dict: Dict) -> float:
        """Total CDU flow should match datacenter flow"""
        total_cdu_flow = torch.sum(vars_dict['V_flow_prim_GPM'], dim=1)
        datacenter_flow = vars_dict['datacenter_flow_GPM']
        
        flow_error = torch.abs(total_cdu_flow - datacenter_flow) / (
            torch.abs(datacenter_flow) + self.epsilon
        )
        
        return torch.clamp(flow_error, 0, 1).mean().item()
    
    def _loss_energy_balance(self, vars_dict: Dict) -> float:
        """Energy removed by water should match heat load"""
        batch_size = vars_dict['cdu_heat_load_W'].shape[0]
        energy_errors = []
        
        for b in range(batch_size):
            for cdu_idx in range(self.num_cdus):
                # Heat load (W)
                Q_load = vars_dict['cdu_heat_load_W'][b, cdu_idx]
                
                # Heat removed by secondary loop
                V_flow_sec = vars_dict['V_flow_sec_GPM'][b, cdu_idx] * self.GPM_TO_M3_S
                delta_T_sec = vars_dict['T_sec_r_C'][b, cdu_idx] - vars_dict['T_sec_s_C'][b, cdu_idx]
                
                Q_removed = (self.WATER_DENSITY * V_flow_sec * self.WATER_SPECIFIC_HEAT * 
                           torch.abs(delta_T_sec))
                
                # Relative error
                energy_error = torch.abs(Q_load - Q_removed) / (Q_load + self.epsilon)
                energy_errors.append(energy_error)
        
        return torch.stack(energy_errors).mean().item()
    
    def _loss_monotonicity(self, vars_dict: Dict) -> float:
        """Higher power should generally lead to higher cooling requirements"""
        batch_size = vars_dict['cdu_heat_load_W'].shape[0]
        monotonicity_scores = []
        
        for b in range(batch_size):
            heat_sorted_idx = torch.argsort(vars_dict['cdu_heat_load_W'][b])
            sorted_flows_prim = vars_dict['V_flow_prim_GPM'][b][heat_sorted_idx]
            sorted_flows_sec = vars_dict['V_flow_sec_GPM'][b][heat_sorted_idx]
            heat_diff = torch.diff(vars_dict['cdu_heat_load_W'][b][heat_sorted_idx])
            
            flow_diff_prim = torch.diff(sorted_flows_prim)
            flow_diff_sec = torch.diff(sorted_flows_sec)
            
            heat_increases = heat_diff > 0
            flow_decreases_prim = flow_diff_prim < -0.1
            flow_decreases_sec = flow_diff_sec < -0.1
            
            violations_prim = heat_increases & flow_decreases_prim
            violations_sec = heat_increases & flow_decreases_sec
            
            violation_score = (violations_prim.float().sum() + violations_sec.float().sum()) / (2 * len(heat_diff))
            monotonicity_scores.append(violation_score)
        
        return torch.stack(monotonicity_scores).mean().item()
    
    def _loss_thermodynamic_cop(self, vars_dict: Dict) -> float:
        """Coefficient of Performance should be within realistic bounds"""
        total_cooling_W = torch.sum(vars_dict['cdu_heat_load_W'], dim=1)
        total_pump_work_W = torch.sum(vars_dict['pump_power_kW'], dim=1) * 1000
        
        # Simplified Carnot COP
        T_cold_K = torch.mean(vars_dict['T_prim_s_C'] + 273.15, dim=1)
        T_hot_K = torch.mean(vars_dict['cdu_air_temp_C'] + 273.15, dim=1)
        carnot_cop = T_cold_K / (T_hot_K - T_cold_K + self.epsilon)
        
        # Realistic COP (40-60% of Carnot)
        realistic_cop = 0.5 * carnot_cop
        expected_total_work = total_cooling_W / torch.clamp(realistic_cop, min=self.MIN_COP)
        
        # Actual COP
        actual_cop = total_cooling_W / (total_pump_work_W + expected_total_work * 0.1 + self.epsilon)
        
        # COP should be between 2 and 10
        cop_bounds_penalty = F.relu(self.MIN_COP - actual_cop).mean() + \
                            F.relu(actual_cop - 10.0).mean()
        
        return cop_bounds_penalty.item()
    
    def _loss_cooling_tower_effectiveness(self, vars_dict: Dict) -> float:
        """Cooling tower should cool water within reasonable approach of ambient"""
        external_temp_expanded = vars_dict['external_temp_C'].unsqueeze(1).expand_as(
            vars_dict['T_prim_s_C']
        )
        
        # Approach temperature (should be 2-10°C)
        cooling_tower_approach = vars_dict['T_prim_s_C'] - external_temp_expanded
        
        ct_approach_low = F.relu(self.MIN_APPROACH_TEMP - cooling_tower_approach).mean()
        ct_approach_high = torch.tanh(F.relu(cooling_tower_approach - 20.0) / 20.0).mean()

        # Effectiveness = (T_in - T_out) / (T_in - T_ambient)
        effectiveness = (vars_dict['T_prim_r_C'] - vars_dict['T_prim_s_C']) / (
            vars_dict['T_prim_r_C'] - external_temp_expanded + self.epsilon
        )
        
        # Effectiveness should be between 0.6 and 0.95
        eff_bounds_penalty = F.relu(0.6 - effectiveness).mean() + \
                            F.relu(effectiveness - 0.95).mean()
        
        return (ct_approach_low + ct_approach_high + eff_bounds_penalty).item()
    
    def validate_chunk(
        self, 
        chunk_idx: int,
        data_path: str = 'data/',
        max_samples: Optional[int] = None,
        batch_size: int = 256  # Add batch processing
    ) -> Dict:
        """
        Validate a data chunk with batch processing
        """
        self.logger.info(f"Validating chunk {chunk_idx}...")
        
        try:
            chunk_path = Path(data_path) / f"chunk_{chunk_idx}"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk {chunk_idx} directory not found at {chunk_path}")
            
            dataset = DatacenterCoolingDataset(
                data_path=str(Path(data_path)),
                chunk_indices=[chunk_idx],
                sequence_length=12,
                config=self.config,
                normalize=False
            )
            
            if len(dataset) == 0:
                self.logger.warning(f"No data found for chunk {chunk_idx}")
                return None
            
            samples = min(len(dataset), max_samples) if max_samples else len(dataset)
            self.logger.info(f"Processing {samples} samples from chunk {chunk_idx}")
            
            # Process in batches
            physics_losses_list = []
            physics_analyses = []
            
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
            
            for batch_idx, (inputs, outputs) in enumerate(tqdm(loader, desc=f"Chunk {chunk_idx}")):
                if batch_idx * batch_size >= samples:
                    break
                
                # Move to device
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                
                # Calculate physics losses for batch
                losses = self.compute_physics_loss(outputs, inputs)
                
                # Store per-sample losses
                for i in range(outputs.shape[0]):
                    sample_losses = {k: v if isinstance(v, float) else v 
                                for k, v in losses.items()}
                    physics_losses_list.append(sample_losses)
                
                # Detailed analysis every 10 batches
                if batch_idx % 10 == 0:
                    analysis = self.analyze_physics_violations(outputs, inputs)
                    physics_analyses.append(analysis)
            
            if not physics_losses_list:
                self.logger.warning(f"No valid results for chunk {chunk_idx}")
                return None
            
            self.logger.info(f"✓ Chunk {chunk_idx}: {len(physics_losses_list)} samples validated")
            
            return {
                'chunk_idx': chunk_idx,
                'num_samples': len(physics_losses_list),
                'physics_losses': physics_losses_list,
                'physics_analyses': physics_analyses
            }
            
        except Exception as e:
            self.logger.error(f"Error validating chunk {chunk_idx}: {e}")
            traceback.print_exc()
            return None
    
    def plot_physics_losses(
        self, 
        results: Dict[int, Dict],
        save_dir: str = 'physics_validation_results'
    ):
        """
        Create visualization plots for physics losses with subplots
        
        Parameters:
        -----------
        results : Dict[int, Dict]
            Results from multiple chunks
        save_dir : str
            Directory to save plots
        """
        if not results:
            self.logger.warning("No results to plot")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        components = list(self.physics_weights.keys())
        
        # Calculate subplot grid
        n_chunks = len(results)
        n_cols = min(3, n_chunks)
        n_rows = (n_chunks + n_cols - 1) // n_cols
        
        # Create figure for total loss across chunks
        fig_total, axes_total = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_chunks == 1:
            axes_total = np.array([[axes_total]])
        elif n_rows == 1:
            axes_total = axes_total.reshape(1, -1)
        elif n_cols == 1:
            axes_total = axes_total.reshape(-1, 1)
        axes_total = axes_total.flatten()
        
        fig_total.suptitle('Total Physics Loss Over Time by Chunk', fontsize=16, y=1.0)
        
        # Create figure for component breakdown
        fig_comp, axes_comp = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_chunks == 1:
            axes_comp = np.array([[axes_comp]])
        elif n_rows == 1:
            axes_comp = axes_comp.reshape(1, -1)
        elif n_cols == 1:
            axes_comp = axes_comp.reshape(-1, 1)
        axes_comp = axes_comp.flatten()
        
        fig_comp.suptitle('Physics Loss Components by Chunk', fontsize=16, y=1.0)
        
        # Color map for components
        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
        
        for idx, (chunk_idx, chunk_results) in enumerate(sorted(results.items())):
            if chunk_results is None:
                continue
                
            losses_df = pd.DataFrame(chunk_results['physics_losses'])
            num_samples = len(losses_df)
            time_axis = np.arange(num_samples)
            
            # Plot total loss
            ax_total = axes_total[idx]
            ax_total.plot(time_axis, losses_df['total'], 
                        color='navy', linewidth=0.5, alpha=0.7, label='Total Loss')
            
            # Add moving average
            if num_samples > 20:
                window_size = max(10, num_samples // 50)
                moving_avg = losses_df['total'].rolling(window=window_size, min_periods=1).mean()
                ax_total.plot(time_axis, moving_avg, 
                            color='red', linewidth=1.5, linestyle='--', 
                            alpha=0.8, label=f'Moving Avg (w={window_size})')
            
            ax_total.set_xlabel('Sample Index', fontsize=10)
            ax_total.set_ylabel('Total Loss', fontsize=10)
            ax_total.set_title(f'Chunk {chunk_idx}\n({num_samples} samples)', fontsize=12)
            ax_total.grid(True, alpha=0.3)
            ax_total.legend(loc='upper right', fontsize=8)
            
            # Add statistics text
            mean_loss = losses_df['total'].mean()
            std_loss = losses_df['total'].std()
            max_loss = losses_df['total'].max()
            min_loss = losses_df['total'].min()
            stats_text = f'μ={mean_loss:.3f}\nσ={std_loss:.3f}\nmax={max_loss:.3f}\nmin={min_loss:.3f}'
            ax_total.text(0.02, 0.98, stats_text, transform=ax_total.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot component breakdown
            ax_comp = axes_comp[idx]
            
            for comp_idx, comp in enumerate(components):
                if comp in losses_df.columns:
                    ax_comp.plot(time_axis, losses_df[comp], 
                              color=colors[comp_idx], linewidth=0.8, 
                              alpha=0.7, label=comp.replace('_', ' ').title())
            
            ax_comp.set_xlabel('Sample Index', fontsize=10)
            ax_comp.set_ylabel('Loss Value', fontsize=10)
            ax_comp.set_title(f'Chunk {chunk_idx} - Component Breakdown', fontsize=12)
            ax_comp.grid(True, alpha=0.3)
            ax_comp.legend(loc='upper right', fontsize=7, ncol=2)
            ax_comp.set_xlim(0, num_samples)
        
        # Hide unused subplots
        for idx in range(n_chunks, len(axes_total)):
            axes_total[idx].set_visible(False)
            axes_comp[idx].set_visible(False)
        
        # Save figures
        fig_total.tight_layout()
        total_path = f"{save_dir}/total_loss_by_chunk.png"
        fig_total.savefig(total_path, dpi=150, bbox_inches='tight')
        plt.close(fig_total)
        self.logger.info(f"Saved total loss plot to {total_path}")
        
        fig_comp.tight_layout()
        comp_path = f"{save_dir}/component_loss_by_chunk.png"
        fig_comp.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig_comp)
        self.logger.info(f"Saved component loss plot to {comp_path}")
        
        # Create heatmap comparison
        self._create_chunk_comparison_heatmap(results, save_dir)
    
    def _create_chunk_comparison_heatmap(self, all_results: Dict, save_dir: str):
        """Create heatmap comparing loss components across chunks"""
        components = list(self.physics_weights.keys())
        
        # Prepare data for heatmap
        chunk_ids = sorted(all_results.keys())
        heatmap_data = []
        
        for chunk_idx in chunk_ids:
            if all_results[chunk_idx] is None:
                continue
            results_df = pd.DataFrame(all_results[chunk_idx]['physics_losses'])
            row_data = []
            for comp in components:
                if comp in results_df.columns:
                    row_data.append(results_df[comp].mean())
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        if not heatmap_data:
            return
            
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(chunk_ids) * 0.4)))
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(components)))
        ax.set_yticks(np.arange(len(chunk_ids)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45, ha='right')
        ax.set_yticklabels([f'Chunk {i}' for i in chunk_ids])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Loss Value', rotation=270, labelpad=20)
        
        # Add values in cells
        for i in range(len(chunk_ids)):
            for j in range(len(components)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Physics Loss Components Comparison Across Chunks', fontsize=14)
        plt.tight_layout()
        heatmap_path = f"{save_dir}/chunk_comparison_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comparison heatmap to {heatmap_path}")


def validate_chunk_worker(args):
    """Worker function for parallel chunk validation"""
    chunk_idx, data_path, max_samples, use_gpu, gpu_id, system_name = args
    
    try:
        # Set GPU if requested
        if use_gpu and gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        
        print(f"[Worker {chunk_idx}] Processing on device {device}")
        
        # Create validator
        validator = PhysicsValidator(device=device, system_name=system_name)
        
        # Validate chunk
        results = validator.validate_chunk(chunk_idx, data_path, max_samples)
        
        return results
        
    except Exception as e:
        print(f"[Worker {chunk_idx}] Error: {e}")
        traceback.print_exc()
        return None


def validate_chunks_parallel(
    chunk_indices: List[int],
    data_path: str = 'data/',
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    output_dir: str = 'physics_validation_results',
    use_gpu: bool = True,
    system_name: str = 'marconi100'
) -> Dict[int, Dict]:
    """
    Validate multiple chunks in parallel
    
    Parameters:
    -----------
    chunk_indices : List[int]
        List of chunk indices to validate
    data_path : str
        Path to data directory
    max_samples : int, optional
        Maximum samples per chunk
    num_workers : int
        Number of parallel workers
    output_dir : str
        Directory to save results
    use_gpu : bool
        Whether to use GPU
    system_name : str
        Name of the system for Importing NumCDUs
    
    Returns:
    --------
    Dict mapping chunk_idx to validation results
    """
    
    if use_gpu and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Using {num_workers} workers with {n_gpus} GPUs available")
        
        # Assign GPU IDs to workers
        worker_args = []
        for i, chunk_idx in enumerate(chunk_indices):
            gpu_id = i % n_gpus
            worker_args.append((chunk_idx, data_path, max_samples, True, gpu_id, system_name))
    else:
        print(f"Using {num_workers} CPU workers")
        worker_args = [(chunk_idx, data_path, max_samples, False, None, system_name) 
                      for chunk_idx in chunk_indices]
    
    # Use multiprocessing pool
    results_dict = {}
    
    if num_workers > 1:
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(validate_chunk_worker, worker_args)
            
            for result in results:
                if result is not None:
                    results_dict[result['chunk_idx']] = result
    else:
        # Sequential processing
        for args in worker_args:
            result = validate_chunk_worker(args)
            if result is not None:
                results_dict[result['chunk_idx']] = result
    
    if not results_dict:
        print("No results generated!")
        return results_dict
    
    # Create validator for plotting
    validator = PhysicsValidator(system_name=system_name)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    validator.plot_physics_losses(results_dict, save_dir=output_dir)
    
    # Save summary
    summary = {}
    for chunk_idx, r in results_dict.items():
        losses_df = pd.DataFrame(r['physics_losses'])
        summary[chunk_idx] = {
            'num_samples': r['num_samples'],
            'mean_total_loss': losses_df['total'].mean(),
            'std_total_loss': losses_df['total'].std(),
            'max_total_loss': losses_df['total'].max(),
            'min_total_loss': losses_df['total'].min(),
            'components': {
                comp: {
                    'mean': losses_df[comp].mean(),
                    'std': losses_df[comp].std(),
                    'max': losses_df[comp].max(),
                    'min': losses_df[comp].min()
                } for comp in losses_df.columns if comp != 'total'
            }
        }
    
    # Save summary to JSON
    summary_path = f"{output_dir}/validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")
    
    # Create summary CSV
    summary_df = pd.DataFrame([
        {
            'chunk': chunk_idx,
            'samples': s['num_samples'],
            'mean_total_loss': s['mean_total_loss'],
            'std_total_loss': s['std_total_loss'],
            'max_total_loss': s['max_total_loss'],
            'min_total_loss': s['min_total_loss']
        } for chunk_idx, s in summary.items()
    ])
    csv_path = f"{output_dir}/validation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"CSV summary saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return results_dict


def analyze_violations(results_dict: Dict, output_dir: str = 'physics_validation_results'):
    """
    Analyze physics violations in detail
    
    Parameters:
    -----------
    results_dict : Dict
        Results from validation
    output_dir : str
        Directory to save analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    violation_stats = {}
    
    for chunk_idx, results in results_dict.items():
        if results is None or 'physics_analyses' not in results:
            continue
            
        analyses = results['physics_analyses']
        if not analyses:
            continue
            
        # Aggregate violation statistics
        temp_violations = []
        approach_violations = []
        pue_errors = []
        flow_errors = []
        
        for analysis in analyses:
            if 'temp_ordering' in analysis:
                temp_violations.append(analysis['temp_ordering']['total_violation_rate'])
            if 'approach_temp' in analysis:
                approach_violations.append(analysis['approach_temp']['violation_rate'])
            if 'pue' in analysis:
                pue_errors.extend(analysis['pue']['pue_error'])
            if 'mass_conservation' in analysis:
                flow_errors.extend(analysis['mass_conservation']['relative_error'])
        
        violation_stats[chunk_idx] = {
            'temp_violation_rate': np.mean(temp_violations) if temp_violations else 0,
            'approach_violation_rate': np.mean(approach_violations) if approach_violations else 0,
            'mean_pue_error': np.mean(pue_errors) if pue_errors else 0,
            'mean_flow_error': np.mean(flow_errors) if flow_errors else 0
        }
    
    # Create violation summary plot
    if violation_stats:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        chunks = sorted(violation_stats.keys())
        
        # Temperature violations
        ax = axes[0, 0]
        temp_rates = [violation_stats[c]['temp_violation_rate'] * 100 for c in chunks]
        ax.bar(range(len(chunks)), temp_rates, color='red', alpha=0.7)
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Violation Rate (%)')
        ax.set_title('Temperature Ordering Violations')
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels(chunks)
        ax.grid(True, alpha=0.3)
        
        # Approach temperature violations
        ax = axes[0, 1]
        approach_rates = [violation_stats[c]['approach_violation_rate'] * 100 for c in chunks]
        ax.bar(range(len(chunks)), approach_rates, color='orange', alpha=0.7)
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Violation Rate (%)')
        ax.set_title('Approach Temperature Violations')
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels(chunks)
        ax.grid(True, alpha=0.3)
        
        # PUE errors
        ax = axes[1, 0]
        pue_errors = [violation_stats[c]['mean_pue_error'] for c in chunks]
        ax.bar(range(len(chunks)), pue_errors, color='blue', alpha=0.7)
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Mean Error')
        ax.set_title('PUE Calculation Errors')
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels(chunks)
        ax.grid(True, alpha=0.3)
        
        # Flow conservation errors
        ax = axes[1, 1]
        flow_errors = [violation_stats[c]['mean_flow_error'] for c in chunks]
        ax.bar(range(len(chunks)), flow_errors, color='green', alpha=0.7)
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Relative Error')
        ax.set_title('Mass Conservation Errors')
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels(chunks)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        violation_path = f"{output_dir}/violation_analysis.png"
        plt.savefig(violation_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Violation analysis saved to {violation_path}")
    
    # Save violation statistics
    violation_df = pd.DataFrame.from_dict(violation_stats, orient='index')
    violation_csv = f"{output_dir}/violation_stats.csv"
    violation_df.to_csv(violation_csv)
    print(f"Violation statistics saved to {violation_csv}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics-Informed Validation for Datacenter Cooling')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/', 
                       help='Path to data chunks')
    parser.add_argument('--chunks', type=str, default='0',
                       help='Chunk indices (comma-separated) or "all"')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per chunk')
    
    # Cooling Model Arguments
    parser.add_argument('--system_name', type=str, default='marconi100',
                       help='Name of the datacenter cooling system configuration')
    
    # Parallelization arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='physics_validation_results',
                       help='Directory to save results')
    parser.add_argument('--analyze_violations', action='store_true',
                       help='Perform detailed violation analysis')
    
    args = parser.parse_args()
    
    # Parse chunks
    if args.chunks.lower() == 'all':
        data_path = Path(args.data_path)
        chunk_indices = []
        for chunk_dir in sorted(data_path.glob("chunk_*")):
            try:
                chunk_idx = int(chunk_dir.name.split("_")[1])
                chunk_indices.append(chunk_idx)
            except ValueError:
                continue
        print(f"Found {len(chunk_indices)} chunks: {chunk_indices}")
    else:
        chunk_indices = [int(x.strip()) for x in args.chunks.split(',')]
    
    if not chunk_indices:
        print("No chunks to process!")
        exit(1)
    
    # Setup multiprocessing for CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print(f"\n{'='*80}")
    print("PHYSICS VALIDATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Chunks to process: {chunk_indices}")
    print(f"Max samples per chunk: {args.max_samples if args.max_samples else 'All'}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Use GPU: {not args.no_gpu}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Validate chunks
    results = validate_chunks_parallel(
        chunk_indices=chunk_indices,
        data_path=args.data_path,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        system_name=args.system_name
    )
    
    # Analyze violations if requested
    if args.analyze_violations and results:
        print("\nPerforming violation analysis...")
        analyze_violations(results, args.output_dir)
    
    print(f"\n✓ Validation complete! Results saved to {args.output_dir}")