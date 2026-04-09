"""
Operating Regime Analyzer for CDU-Level Comparative Analysis.

Module D from the analysis framework:
- D1: Thermal Efficiency Analysis
- D2: Pumping Efficiency Analysis
- D3: Operating Envelope Detection
- D4: Constraint Boundary Analysis
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class OperatingRegimeConfig:
    """Configuration for operating regime analysis."""
    
    # Efficiency computation
    compute_thermal_efficiency: bool = True
    compute_pumping_efficiency: bool = True
    
    # Operating envelope
    envelope_method: str = "convex_hull"  # convex_hull, kde, dbscan
    kde_threshold: float = 0.1  # Density threshold for KDE envelope
    dbscan_eps: float = 0.3  # DBSCAN clustering epsilon
    dbscan_min_samples: int = 10
    
    # Constraint detection
    saturation_threshold: float = 0.02  # % of range to consider saturated
    constraint_min_samples: int = 50  # Minimum samples to identify constraint
    
    # PUE computation
    compute_pue: bool = True
    baseline_power_kw: float = 0.0  # Non-cooling infrastructure power


@dataclass
class OperatingRegimeResult:
    """Results from operating regime analysis."""
    
    model_name: str
    cdu_id: int
    
    # Efficiency metrics
    thermal_efficiency: Dict[str, float] = field(default_factory=dict)
    pumping_efficiency: Dict[str, float] = field(default_factory=dict)
    overall_efficiency: Dict[str, float] = field(default_factory=dict)
    
    # Operating envelope
    envelope_vertices: Optional[np.ndarray] = None
    envelope_volume: float = 0.0
    operating_centroid: Optional[np.ndarray] = None
    
    # Constraint boundaries
    saturated_outputs: List[Dict[str, Any]] = field(default_factory=list)
    constraint_regions: List[Dict[str, Any]] = field(default_factory=list)
    
    # PUE
    pue_stats: Dict[str, float] = field(default_factory=dict)
    
    # Operating ranges
    input_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    output_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class OperatingRegimeAnalyzer:
    """
    Analyzes operating regimes and efficiency of CDU cooling systems.
    
    Key capabilities:
    1. Thermal efficiency: Heat removal effectiveness
    2. Pumping efficiency: Flow work effectiveness
    3. Operating envelope: Valid operating region characterization
    4. Constraint boundaries: Where system hits limits
    5. Cross-model regime comparison
    """
    
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    OUTPUT_VARS = [
        "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW",
        "T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C",
        "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig", "p_sec_r_psig"
    ]
    
    # Units and conversions
    GPM_TO_M3S = 6.309e-5  # GPM to m³/s
    PSIG_TO_PA = 6894.76  # psig to Pa
    
    def __init__(self, config: Optional[OperatingRegimeConfig] = None):
        """Initialize operating regime analyzer."""
        self.config = config or OperatingRegimeConfig()
        self.results: Dict[str, Dict[int, OperatingRegimeResult]] = {}
        
    def _get_input_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get input column names for a CDU."""
        return {
            "Q_flow": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total",
            "T_Air": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_T_Air",
            "T_ext": "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext"
        }
        
    def _get_output_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get output column names for a CDU."""
        return {var: f"simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.{var}"
                for var in self.OUTPUT_VARS}
    
    def compute_thermal_efficiency(
        self,
        data: pd.DataFrame,
        cdu_id: int
    ) -> Dict[str, float]:
        """
        Compute thermal efficiency metrics.
        
        Thermal Efficiency = Q_removed / Q_input
        Where Q_removed is calculated from flow rate and temperature difference.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            
        Returns:
            Dict with thermal efficiency metrics
        """
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        metrics = {}
        
        # Get heat load
        q_flow_col = input_cols["Q_flow"]
        if q_flow_col not in data.columns:
            return metrics
            
        Q_input = data[q_flow_col].values  # kW
        
        # Compute heat removed from primary loop
        # Q = m_dot * Cp * delta_T
        # Using water: Cp ≈ 4.186 kJ/(kg·K), rho ≈ 1000 kg/m³
        
        v_prim_col = output_cols.get("V_flow_prim_GPM")
        t_prim_s_col = output_cols.get("T_prim_s_C")
        t_prim_r_col = output_cols.get("T_prim_r_C")
        
        if all(c in data.columns for c in [v_prim_col, t_prim_s_col, t_prim_r_col]):
            # Flow rate in m³/s
            V_prim = data[v_prim_col].values * self.GPM_TO_M3S
            T_prim_s = data[t_prim_s_col].values  # °C
            T_prim_r = data[t_prim_r_col].values  # °C
            
            # Heat removed (kW)
            # Q = rho * V * Cp * delta_T = 1000 * V * 4.186 * (T_r - T_s) / 1000
            Q_prim_removed = 1000 * V_prim * 4.186 * np.abs(T_prim_r - T_prim_s)
            
            # Efficiency
            valid_mask = Q_input > 0
            if np.any(valid_mask):
                eff_prim = Q_prim_removed[valid_mask] / Q_input[valid_mask]
                metrics["primary_loop_efficiency_mean"] = float(np.mean(eff_prim))
                metrics["primary_loop_efficiency_std"] = float(np.std(eff_prim))
                metrics["primary_loop_efficiency_min"] = float(np.min(eff_prim))
                metrics["primary_loop_efficiency_max"] = float(np.max(eff_prim))
        
        # Compute heat removed from secondary loop
        v_sec_col = output_cols.get("V_flow_sec_GPM")
        t_sec_s_col = output_cols.get("T_sec_s_C")
        t_sec_r_col = output_cols.get("T_sec_r_C")
        
        if all(c in data.columns for c in [v_sec_col, t_sec_s_col, t_sec_r_col]):
            V_sec = data[v_sec_col].values * self.GPM_TO_M3S
            T_sec_s = data[t_sec_s_col].values
            T_sec_r = data[t_sec_r_col].values
            
            Q_sec_removed = 1000 * V_sec * 4.186 * np.abs(T_sec_r - T_sec_s)
            
            valid_mask = Q_input > 0
            if np.any(valid_mask):
                eff_sec = Q_sec_removed[valid_mask] / Q_input[valid_mask]
                metrics["secondary_loop_efficiency_mean"] = float(np.mean(eff_sec))
                metrics["secondary_loop_efficiency_std"] = float(np.std(eff_sec))
        
        # Temperature approach (how close cooling water gets to air temp)
        t_air_col = input_cols["T_Air"]
        if t_air_col in data.columns and t_prim_r_col in data.columns:
            T_air = data[t_air_col].values - 273.15  # K to °C
            T_approach = data[t_prim_r_col].values - T_air
            metrics["temperature_approach_mean"] = float(np.mean(T_approach))
            metrics["temperature_approach_std"] = float(np.std(T_approach))
        
        return metrics
    
    def compute_pumping_efficiency(
        self,
        data: pd.DataFrame,
        cdu_id: int
    ) -> Dict[str, float]:
        """
        Compute pumping efficiency metrics.
        
        Pumping Efficiency = Q_removed / W_pump
        Also computes hydraulic efficiency: W_hydraulic / W_electrical
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            
        Returns:
            Dict with pumping efficiency metrics
        """
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        metrics = {}
        
        # Get CDU power consumption
        w_cdu_col = output_cols.get("W_flow_CDUP_kW")
        q_flow_col = input_cols["Q_flow"]
        
        if w_cdu_col not in data.columns or q_flow_col not in data.columns:
            return metrics
            
        W_pump = data[w_cdu_col].values  # kW
        Q_input = data[q_flow_col].values  # kW
        
        # Pumping efficiency: Heat removed per unit pump power
        valid_mask = W_pump > 0
        if np.any(valid_mask):
            eff = Q_input[valid_mask] / W_pump[valid_mask]
            metrics["cop_mean"] = float(np.mean(eff))  # Coefficient of Performance
            metrics["cop_std"] = float(np.std(eff))
            metrics["cop_min"] = float(np.min(eff))
            metrics["cop_max"] = float(np.max(eff))
        
        # Hydraulic power (W_h = V * delta_P)
        v_prim_col = output_cols.get("V_flow_prim_GPM")
        p_prim_s_col = output_cols.get("p_prim_s_psig")
        p_prim_r_col = output_cols.get("p_prim_r_psig")
        
        if all(c in data.columns for c in [v_prim_col, p_prim_s_col, p_prim_r_col]):
            V_prim = data[v_prim_col].values * self.GPM_TO_M3S  # m³/s
            delta_P = np.abs(data[p_prim_s_col].values - data[p_prim_r_col].values) * self.PSIG_TO_PA  # Pa
            
            W_hydraulic = V_prim * delta_P / 1000  # kW
            
            valid_mask = W_pump > 0
            if np.any(valid_mask):
                hydraulic_eff = W_hydraulic[valid_mask] / W_pump[valid_mask]
                metrics["hydraulic_efficiency_mean"] = float(np.mean(hydraulic_eff))
                metrics["hydraulic_efficiency_std"] = float(np.std(hydraulic_eff))
        
        # Power per unit flow
        if v_prim_col in data.columns:
            V_prim_gpm = data[v_prim_col].values
            valid_mask = V_prim_gpm > 0
            if np.any(valid_mask):
                power_per_flow = W_pump[valid_mask] / V_prim_gpm[valid_mask]
                metrics["power_per_gpm_mean"] = float(np.mean(power_per_flow))
                metrics["power_per_gpm_std"] = float(np.std(power_per_flow))
        
        return metrics
    
    def compute_operating_envelope(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        input_vars: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute operating envelope (valid operating region).
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            input_vars: Input variables to use (default: all)
            
        Returns:
            Dict with envelope characterization
        """
        input_cols = self._get_input_columns(cdu_id)
        input_vars = input_vars or self.INPUT_VARS
        
        # Extract input data
        available_vars = []
        X_data = []
        
        for var in input_vars:
            col = input_cols.get(var)
            if col and col in data.columns:
                X_data.append(data[col].values)
                available_vars.append(var)
                
        if len(X_data) < 2:
            return {"error": "Insufficient input data"}
            
        X = np.column_stack(X_data)
        
        # Remove NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        
        if len(X) < 10:
            return {"error": "Insufficient valid samples"}
        
        result = {
            "variables": available_vars,
            "n_samples": len(X),
            "ranges": {}
        }
        
        # Compute ranges
        for i, var in enumerate(available_vars):
            result["ranges"][var] = (float(X[:, i].min()), float(X[:, i].max()))
        
        # Compute centroid
        result["centroid"] = {var: float(X[:, i].mean()) for i, var in enumerate(available_vars)}
        
        # Compute envelope based on method
        if self.config.envelope_method == "convex_hull" and len(X) >= len(available_vars) + 1:
            try:
                # Scale data for better hull computation
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                hull = ConvexHull(X_scaled)
                
                # Convert vertices back to original scale
                vertices_scaled = X_scaled[hull.vertices]
                vertices = scaler.inverse_transform(vertices_scaled)
                
                result["envelope_vertices"] = vertices.tolist()
                result["envelope_volume"] = float(hull.volume)
                result["envelope_area"] = float(hull.area) if hasattr(hull, 'area') else 0.0
                result["n_vertices"] = len(hull.vertices)
                
            except Exception as e:
                logger.warning(f"Convex hull computation failed: {e}")
                
        elif self.config.envelope_method == "kde":
            try:
                # Scale data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit KDE
                kde = gaussian_kde(X_scaled.T)
                
                # Evaluate density at all points
                densities = kde(X_scaled.T)
                
                # Find high-density region
                threshold = np.percentile(densities, self.config.kde_threshold * 100)
                high_density_mask = densities >= threshold
                
                result["high_density_points"] = int(np.sum(high_density_mask))
                result["density_threshold"] = float(threshold)
                
            except Exception as e:
                logger.warning(f"KDE computation failed: {e}")
                
        elif self.config.envelope_method == "dbscan":
            try:
                # Scale data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # DBSCAN clustering
                clustering = DBSCAN(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples
                ).fit(X_scaled)
                
                # Find main cluster
                labels = clustering.labels_
                unique_labels = set(labels)
                unique_labels.discard(-1)  # Remove noise label
                
                if unique_labels:
                    main_label = max(unique_labels, key=lambda l: np.sum(labels == l))
                    main_cluster_mask = labels == main_label
                    
                    result["n_clusters"] = len(unique_labels)
                    result["main_cluster_size"] = int(np.sum(main_cluster_mask))
                    result["noise_points"] = int(np.sum(labels == -1))
                    
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {e}")
        
        return result
    
    def detect_constraint_boundaries(
        self,
        data: pd.DataFrame,
        cdu_id: int
    ) -> Dict[str, Any]:
        """
        Detect where outputs hit constraints/limits.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            
        Returns:
            Dict with constraint boundary analysis
        """
        output_cols = self._get_output_columns(cdu_id)
        
        result = {
            "saturated_outputs": [],
            "constraint_regions": []
        }
        
        for var in self.OUTPUT_VARS:
            col = output_cols.get(var)
            if col not in data.columns:
                continue
                
            values = data[col].dropna().values
            if len(values) < self.config.constraint_min_samples:
                continue
            
            var_min, var_max = values.min(), values.max()
            var_range = var_max - var_min
            
            if var_range == 0:
                continue
                
            # Check for saturation at min
            threshold_low = var_min + self.config.saturation_threshold * var_range
            n_at_min = np.sum(values <= threshold_low)
            
            if n_at_min >= self.config.constraint_min_samples:
                result["saturated_outputs"].append({
                    "variable": var,
                    "boundary": "min",
                    "value": float(var_min),
                    "n_samples": int(n_at_min),
                    "percentage": float(n_at_min / len(values) * 100)
                })
            
            # Check for saturation at max
            threshold_high = var_max - self.config.saturation_threshold * var_range
            n_at_max = np.sum(values >= threshold_high)
            
            if n_at_max >= self.config.constraint_min_samples:
                result["saturated_outputs"].append({
                    "variable": var,
                    "boundary": "max",
                    "value": float(var_max),
                    "n_samples": int(n_at_max),
                    "percentage": float(n_at_max / len(values) * 100)
                })
        
        # Find constraint regions by clustering saturated samples
        input_cols = self._get_input_columns(cdu_id)
        
        for sat_output in result["saturated_outputs"]:
            var = sat_output["variable"]
            col = output_cols[var]
            
            if sat_output["boundary"] == "min":
                threshold = data[col].min() + self.config.saturation_threshold * (data[col].max() - data[col].min())
                mask = data[col] <= threshold
            else:
                threshold = data[col].max() - self.config.saturation_threshold * (data[col].max() - data[col].min())
                mask = data[col] >= threshold
            
            # Get corresponding input values
            input_at_constraint = {}
            for inp_var, inp_col in input_cols.items():
                if inp_col in data.columns:
                    vals = data.loc[mask, inp_col].values
                    input_at_constraint[inp_var] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals))
                    }
            
            result["constraint_regions"].append({
                "output": var,
                "boundary": sat_output["boundary"],
                "input_conditions": input_at_constraint
            })
        
        return result
    
    def compute_pue(
        self,
        data: pd.DataFrame,
        cdu_id: int
    ) -> Dict[str, float]:
        """
        Compute Power Usage Effectiveness contribution.
        
        PUE = Total Facility Power / IT Equipment Power
        Here we compute CDU contribution to cooling overhead.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            
        Returns:
            Dict with PUE-related metrics
        """
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        metrics = {}
        
        q_flow_col = input_cols["Q_flow"]
        w_cdu_col = output_cols.get("W_flow_CDUP_kW")
        
        if q_flow_col not in data.columns or w_cdu_col not in data.columns:
            return metrics
        
        Q_IT = data[q_flow_col].values  # IT heat load (≈ IT power)
        W_CDU = data[w_cdu_col].values  # CDU power consumption
        
        # CDU overhead ratio
        valid_mask = Q_IT > 0
        if np.any(valid_mask):
            overhead_ratio = W_CDU[valid_mask] / Q_IT[valid_mask]
            metrics["cdu_overhead_mean"] = float(np.mean(overhead_ratio))
            metrics["cdu_overhead_std"] = float(np.std(overhead_ratio))
            metrics["cdu_overhead_max"] = float(np.max(overhead_ratio))
        
        # Partial PUE (IT + CDU) / IT
        total_power = Q_IT + W_CDU + self.config.baseline_power_kw
        valid_mask = Q_IT > 0
        if np.any(valid_mask):
            partial_pue = total_power[valid_mask] / Q_IT[valid_mask]
            metrics["partial_pue_mean"] = float(np.mean(partial_pue))
            metrics["partial_pue_std"] = float(np.std(partial_pue))
            metrics["partial_pue_min"] = float(np.min(partial_pue))
            metrics["partial_pue_max"] = float(np.max(partial_pue))
        
        return metrics
    
    def analyze_cdu(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model"
    ) -> OperatingRegimeResult:
        """
        Complete operating regime analysis for a CDU.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            
        Returns:
            OperatingRegimeResult
        """
        result = OperatingRegimeResult(model_name=model_name, cdu_id=cdu_id)
        
        logger.info(f"Analyzing operating regime for {model_name} CDU {cdu_id}")
        
        # Thermal efficiency
        if self.config.compute_thermal_efficiency:
            result.thermal_efficiency = self.compute_thermal_efficiency(data, cdu_id)
            logger.debug(f"Thermal efficiency computed: {len(result.thermal_efficiency)} metrics")
        
        # Pumping efficiency
        if self.config.compute_pumping_efficiency:
            result.pumping_efficiency = self.compute_pumping_efficiency(data, cdu_id)
            logger.debug(f"Pumping efficiency computed: {len(result.pumping_efficiency)} metrics")
        
        # Operating envelope
        envelope = self.compute_operating_envelope(data, cdu_id)
        if "envelope_vertices" in envelope:
            result.envelope_vertices = np.array(envelope["envelope_vertices"])
            result.envelope_volume = envelope.get("envelope_volume", 0.0)
        if "centroid" in envelope:
            result.operating_centroid = np.array(list(envelope["centroid"].values()))
        
        # Input/output ranges
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        for var, col in input_cols.items():
            if col in data.columns:
                result.input_ranges[var] = (float(data[col].min()), float(data[col].max()))
        
        for var, col in output_cols.items():
            if col in data.columns:
                result.output_ranges[var] = (float(data[col].min()), float(data[col].max()))
        
        # Constraint boundaries
        constraints = self.detect_constraint_boundaries(data, cdu_id)
        result.saturated_outputs = constraints.get("saturated_outputs", [])
        result.constraint_regions = constraints.get("constraint_regions", [])
        
        # PUE
        if self.config.compute_pue:
            result.pue_stats = self.compute_pue(data, cdu_id)
        
        # Combine efficiency metrics
        result.overall_efficiency = {
            **result.thermal_efficiency,
            **result.pumping_efficiency,
            **result.pue_stats
        }
        
        # Store result
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][cdu_id] = result
        
        return result
    
    def compare_across_models(
        self,
        cdu_id: int
    ) -> Dict[str, Any]:
        """
        Compare operating regimes across models for the same CDU.
        
        Args:
            cdu_id: CDU identifier
            
        Returns:
            Comparison results
        """
        comparison = {
            "cdu_id": cdu_id,
            "models": [],
            "efficiency_comparison": {},
            "envelope_comparison": {},
            "constraint_comparison": {}
        }
        
        for model_name, cdus in self.results.items():
            if cdu_id not in cdus:
                continue
                
            result = cdus[cdu_id]
            comparison["models"].append(model_name)
            
            # Efficiency comparison
            for metric, value in result.overall_efficiency.items():
                if metric not in comparison["efficiency_comparison"]:
                    comparison["efficiency_comparison"][metric] = {}
                comparison["efficiency_comparison"][metric][model_name] = value
            
            # Envelope comparison
            comparison["envelope_comparison"][model_name] = {
                "volume": result.envelope_volume,
                "centroid": result.operating_centroid.tolist() if result.operating_centroid is not None else None,
                "n_constraints": len(result.saturated_outputs)
            }
        
        # Compute comparison statistics
        for metric, values in comparison["efficiency_comparison"].items():
            if len(values) > 1:
                vals = list(values.values())
                comparison["efficiency_comparison"][metric]["_range"] = max(vals) - min(vals)
                comparison["efficiency_comparison"][metric]["_mean"] = np.mean(vals)
        
        return comparison
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Convert results to serializable format
        def serialize_result(result: OperatingRegimeResult) -> Dict:
            return {
                "model_name": result.model_name,
                "cdu_id": result.cdu_id,
                "thermal_efficiency": result.thermal_efficiency,
                "pumping_efficiency": result.pumping_efficiency,
                "overall_efficiency": result.overall_efficiency,
                "envelope_volume": result.envelope_volume,
                "operating_centroid": result.operating_centroid.tolist() if result.operating_centroid is not None else None,
                "saturated_outputs": result.saturated_outputs,
                "constraint_regions": result.constraint_regions,
                "pue_stats": result.pue_stats,
                "input_ranges": {k: list(v) for k, v in result.input_ranges.items()},
                "output_ranges": {k: list(v) for k, v in result.output_ranges.items()}
            }
        
        all_results = {}
        for model_name, cdus in self.results.items():
            all_results[model_name] = {
                str(cdu_id): serialize_result(result)
                for cdu_id, result in cdus.items()
            }
        
        with open(output_dir / "operating_regime_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
            
        logger.info(f"Operating regime results saved to {output_dir}")
