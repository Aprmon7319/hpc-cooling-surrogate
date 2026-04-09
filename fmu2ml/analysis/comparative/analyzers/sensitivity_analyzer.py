"""
Sensitivity Analyzer for CDU-Level Comparative Analysis.

Computes Jacobian matrices, sensitivity rankings, and identifies
high-sensitivity operating regions for neural operator training focus.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    
    # Jacobian computation
    finite_diff_step: float = 0.01  # Relative step for finite differences
    central_diff: bool = True  # Use central differences (more accurate)
    
    # Sensitivity ranking
    normalize_sensitivities: bool = True  # Normalize to [0, 1]
    aggregation_method: str = "mean"  # mean, max, rms
    
    # High-sensitivity regions
    sensitivity_threshold: float = 0.7  # Percentile for high sensitivity
    min_region_samples: int = 10  # Minimum samples to define a region
    
    # Monte Carlo sensitivity (Sobol indices)
    n_samples_sobol: int = 1024  # Samples for Sobol analysis
    compute_sobol: bool = True  # Whether to compute Sobol indices
    
    # Cross-model comparison
    compare_sensitivity_profiles: bool = True


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    
    model_name: str
    cdu_id: int
    
    # Jacobian matrices at different operating points
    jacobians: Dict[str, np.ndarray] = field(default_factory=dict)
    jacobian_operating_points: List[Dict[str, float]] = field(default_factory=list)
    
    # Aggregated sensitivities
    mean_sensitivity_matrix: Optional[np.ndarray] = None
    sensitivity_rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    # Sobol indices
    first_order_sobol: Optional[Dict[str, Dict[str, float]]] = None
    total_order_sobol: Optional[Dict[str, Dict[str, float]]] = None
    
    # High-sensitivity regions
    high_sensitivity_regions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    sensitivity_stats: Dict[str, Any] = field(default_factory=dict)


class SensitivityAnalyzer:
    """
    Analyzes sensitivity of CDU outputs to inputs across cooling models.
    
    Key capabilities:
    1. Jacobian computation at operating points
    2. Sensitivity ranking for each output
    3. Sobol indices for global sensitivity
    4. High-sensitivity region identification
    5. Cross-model sensitivity comparison
    """
    
    # CDU inputs and outputs
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    OUTPUT_VARS = [
        "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW",
        "T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C",
        "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig", "p_sec_r_psig"
    ]
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize sensitivity analyzer."""
        self.config = config or SensitivityConfig()
        self.results: Dict[str, Dict[int, SensitivityResult]] = {}  # model -> cdu -> result
        self._surrogate_models: Dict[str, Any] = {}
        
    def compute_jacobian(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        operating_point: Dict[str, float],
        model_name: str = "model"
    ) -> np.ndarray:
        """
        Compute Jacobian matrix at a specific operating point.
        
        J[i,j] = ∂output_i / ∂input_j
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            operating_point: Dict of input values {input_name: value}
            model_name: Model identifier
            
        Returns:
            Jacobian matrix (n_outputs x n_inputs)
        """
        # Build surrogate if not cached
        surrogate_key = f"{model_name}_{cdu_id}"
        if surrogate_key not in self._surrogate_models:
            self._build_surrogate(data, cdu_id, model_name)
            
        surrogate = self._surrogate_models[surrogate_key]
        
        # Compute Jacobian using finite differences
        n_inputs = len(self.INPUT_VARS)
        n_outputs = len(self.OUTPUT_VARS)
        jacobian = np.zeros((n_outputs, n_inputs))
        
        # Base point
        x0 = np.array([operating_point[var] for var in self.INPUT_VARS])
        
        for j, input_var in enumerate(self.INPUT_VARS):
            # Compute step size
            h = self.config.finite_diff_step * abs(x0[j]) if x0[j] != 0 else self.config.finite_diff_step
            
            if self.config.central_diff:
                # Central difference: (f(x+h) - f(x-h)) / (2h)
                x_plus = x0.copy()
                x_plus[j] += h
                x_minus = x0.copy()
                x_minus[j] -= h
                
                y_plus = surrogate(x_plus.reshape(1, -1))
                y_minus = surrogate(x_minus.reshape(1, -1))
                
                jacobian[:, j] = (y_plus - y_minus).flatten() / (2 * h)
            else:
                # Forward difference: (f(x+h) - f(x)) / h
                x_plus = x0.copy()
                x_plus[j] += h
                
                y0 = surrogate(x0.reshape(1, -1))
                y_plus = surrogate(x_plus.reshape(1, -1))
                
                jacobian[:, j] = (y_plus - y0).flatten() / h
                
        return jacobian
    
    def _build_surrogate(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str
    ) -> None:
        """Build RBF surrogate model for Jacobian computation."""
        surrogate_key = f"{model_name}_{cdu_id}"
        
        # Extract input columns
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        available_inputs = [c for c in input_cols.values() if c in data.columns]
        available_outputs = [c for c in output_cols.values() if c in data.columns]
        
        if not available_inputs or not available_outputs:
            logger.warning(f"Insufficient columns for surrogate: CDU {cdu_id}")
            return
            
        # Prepare data
        X = data[available_inputs].dropna()
        y = data[[output_cols[k] for k in self.OUTPUT_VARS if output_cols.get(k) in available_outputs]]
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].values
        y = y.loc[common_idx].values
        
        if len(X) < 10:
            logger.warning(f"Insufficient data for surrogate: CDU {cdu_id}, {len(X)} samples")
            return
            
        # Scale inputs
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Build RBF interpolator
        try:
            rbf = RBFInterpolator(X_scaled, y, kernel="thin_plate_spline", smoothing=0.1)
            
            # Wrap with scaling
            def surrogate_func(x):
                x_scaled = scaler.transform(x)
                return rbf(x_scaled)
                
            self._surrogate_models[surrogate_key] = surrogate_func
            logger.debug(f"Built surrogate for {surrogate_key}")
        except Exception as e:
            logger.error(f"Failed to build surrogate for {surrogate_key}: {e}")
            
    def compute_sensitivity_matrix(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model",
        n_sample_points: int = 20
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Compute average sensitivity matrix over multiple operating points.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            n_sample_points: Number of operating points to sample
            
        Returns:
            Tuple of (mean Jacobian matrix, list of operating points)
        """
        # Get input ranges
        input_cols = self._get_input_columns(cdu_id)
        available_cols = {k: v for k, v in input_cols.items() if v in data.columns}
        
        if not available_cols:
            logger.warning(f"No input columns found for CDU {cdu_id}")
            return np.zeros((len(self.OUTPUT_VARS), len(self.INPUT_VARS))), []
            
        # Sample operating points
        operating_points = []
        jacobians = []
        
        # Use Latin hypercube sampling for operating points
        from scipy.stats import qmc
        
        ranges = {}
        for var in self.INPUT_VARS:
            if var in available_cols:
                col = available_cols[var]
                ranges[var] = (data[col].min(), data[col].max())
            else:
                # Default ranges
                if var == "Q_flow":
                    ranges[var] = (10, 100)  # kW
                elif var == "T_Air":
                    ranges[var] = (290, 310)  # K
                else:  # T_ext
                    ranges[var] = (280, 305)  # K
                    
        sampler = qmc.LatinHypercube(d=len(self.INPUT_VARS))
        samples = sampler.random(n=n_sample_points)
        
        # Scale to actual ranges
        for i, sample in enumerate(samples):
            op = {}
            for j, var in enumerate(self.INPUT_VARS):
                low, high = ranges[var]
                op[var] = low + sample[j] * (high - low)
            operating_points.append(op)
            
            # Compute Jacobian at this point
            try:
                J = self.compute_jacobian(data, cdu_id, op, model_name)
                jacobians.append(J)
            except Exception as e:
                logger.warning(f"Failed Jacobian at point {i}: {e}")
                
        if not jacobians:
            return np.zeros((len(self.OUTPUT_VARS), len(self.INPUT_VARS))), operating_points
            
        # Aggregate Jacobians
        jacobian_array = np.array(jacobians)
        
        if self.config.aggregation_method == "mean":
            mean_jacobian = np.mean(jacobian_array, axis=0)
        elif self.config.aggregation_method == "max":
            mean_jacobian = np.max(np.abs(jacobian_array), axis=0)
        else:  # rms
            mean_jacobian = np.sqrt(np.mean(jacobian_array**2, axis=0))
            
        return mean_jacobian, operating_points
    
    def rank_sensitivities(
        self,
        jacobian: np.ndarray
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rank input sensitivities for each output.
        
        Args:
            jacobian: Jacobian matrix (n_outputs x n_inputs)
            
        Returns:
            Dict mapping output name to ranked list of (input, sensitivity)
        """
        rankings = {}
        
        # Normalize if requested
        if self.config.normalize_sensitivities:
            # Normalize each row independently
            row_maxs = np.max(np.abs(jacobian), axis=1, keepdims=True)
            row_maxs[row_maxs == 0] = 1  # Avoid division by zero
            jacobian_norm = np.abs(jacobian) / row_maxs
        else:
            jacobian_norm = np.abs(jacobian)
            
        for i, output_var in enumerate(self.OUTPUT_VARS):
            sensitivities = []
            for j, input_var in enumerate(self.INPUT_VARS):
                sensitivities.append((input_var, float(jacobian_norm[i, j])))
                
            # Sort by sensitivity (descending)
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            rankings[output_var] = sensitivities
            
        return rankings
    
    def compute_sobol_indices(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model"
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Compute Sobol sensitivity indices.
        
        First-order: Individual effect of each input
        Total-order: Effect including all interactions
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            
        Returns:
            Tuple of (first_order_indices, total_order_indices)
        """
        from scipy.stats import qmc
        
        # Build surrogate if needed
        surrogate_key = f"{model_name}_{cdu_id}"
        if surrogate_key not in self._surrogate_models:
            self._build_surrogate(data, cdu_id, model_name)
            
        if surrogate_key not in self._surrogate_models:
            return {}, {}
            
        surrogate = self._surrogate_models[surrogate_key]
        
        # Get input ranges
        input_cols = self._get_input_columns(cdu_id)
        ranges = []
        for var in self.INPUT_VARS:
            col = input_cols.get(var)
            if col and col in data.columns:
                ranges.append((data[col].min(), data[col].max()))
            else:
                if var == "Q_flow":
                    ranges.append((10, 100))
                elif var == "T_Air":
                    ranges.append((290, 310))
                else:
                    ranges.append((280, 305))
                    
        # Saltelli sampling for Sobol
        n_inputs = len(self.INPUT_VARS)
        sampler = qmc.Sobol(d=n_inputs, scramble=True)
        
        # Need N * (2*d + 2) samples
        N = self.config.n_samples_sobol
        samples = sampler.random(n=N * (2 * n_inputs + 2))
        
        # Scale samples
        for j in range(n_inputs):
            low, high = ranges[j]
            samples[:, j] = low + samples[:, j] * (high - low)
            
        # Split into A, B matrices
        A = samples[:N]
        B = samples[N:2*N]
        
        # Evaluate model
        try:
            Y_A = surrogate(A)
            Y_B = surrogate(B)
        except Exception as e:
            logger.error(f"Sobol evaluation failed: {e}")
            return {}, {}
            
        # Compute indices for each output
        first_order = {output: {} for output in self.OUTPUT_VARS[:Y_A.shape[1]]}
        total_order = {output: {} for output in self.OUTPUT_VARS[:Y_A.shape[1]]}
        
        for j in range(n_inputs):
            # AB_j: A with j-th column from B
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            
            try:
                Y_AB_j = surrogate(AB_j)
            except Exception:
                continue
                
            for i, output in enumerate(self.OUTPUT_VARS[:Y_A.shape[1]]):
                y_a = Y_A[:, i]
                y_b = Y_B[:, i]
                y_ab_j = Y_AB_j[:, i]
                
                var_y = np.var(np.concatenate([y_a, y_b]))
                if var_y == 0:
                    continue
                    
                # First-order: V[E[Y|X_j]] / V[Y]
                S_j = np.mean(y_b * (y_ab_j - y_a)) / var_y
                
                # Total-order: 1 - V[E[Y|X_~j]] / V[Y]
                ST_j = 0.5 * np.mean((y_a - y_ab_j)**2) / var_y
                
                first_order[output][self.INPUT_VARS[j]] = float(np.clip(S_j, 0, 1))
                total_order[output][self.INPUT_VARS[j]] = float(np.clip(ST_j, 0, 1))
                
        return first_order, total_order
    
    def identify_high_sensitivity_regions(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model"
    ) -> List[Dict[str, Any]]:
        """
        Identify operating regions with high sensitivity.
        
        These regions are important for neural operator training.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            
        Returns:
            List of high-sensitivity region descriptors
        """
        # Compute Jacobians over a grid
        input_cols = self._get_input_columns(cdu_id)
        available = {k: v for k, v in input_cols.items() if v in data.columns}
        
        if not available:
            return []
            
        # Grid of points
        n_grid = 10
        grid_points = []
        grid_jacobians = []
        
        ranges = {}
        for var in self.INPUT_VARS:
            if var in available:
                col = available[var]
                ranges[var] = np.linspace(data[col].min(), data[col].max(), n_grid)
            else:
                if var == "Q_flow":
                    ranges[var] = np.linspace(10, 100, n_grid)
                elif var == "T_Air":
                    ranges[var] = np.linspace(290, 310, n_grid)
                else:
                    ranges[var] = np.linspace(280, 305, n_grid)
                    
        # Sample grid
        from itertools import product
        
        for combo in product(*[ranges[var] for var in self.INPUT_VARS]):
            op = {var: val for var, val in zip(self.INPUT_VARS, combo)}
            try:
                J = self.compute_jacobian(data, cdu_id, op, model_name)
                grid_points.append(op)
                grid_jacobians.append(np.linalg.norm(J))  # Frobenius norm
            except Exception:
                continue
                
        if not grid_jacobians:
            return []
            
        # Find high-sensitivity regions
        norms = np.array(grid_jacobians)
        threshold = np.percentile(norms, self.config.sensitivity_threshold * 100)
        
        high_sens_indices = np.where(norms >= threshold)[0]
        
        regions = []
        for idx in high_sens_indices:
            region = {
                "operating_point": grid_points[idx],
                "sensitivity_norm": float(norms[idx]),
                "percentile": float(stats.percentileofscore(norms, norms[idx]))
            }
            regions.append(region)
            
        # Cluster nearby regions
        if len(regions) > self.config.min_region_samples:
            regions = self._cluster_regions(regions)
            
        return regions
    
    def _cluster_regions(
        self,
        regions: List[Dict[str, Any]],
        max_clusters: int = 5
    ) -> List[Dict[str, Any]]:
        """Cluster similar high-sensitivity regions."""
        from sklearn.cluster import KMeans
        
        # Extract operating points
        X = np.array([
            [r["operating_point"][var] for var in self.INPUT_VARS]
            for r in regions
        ])
        
        n_clusters = min(max_clusters, len(regions))
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        
        clustered = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_regions = [r for r, m in zip(regions, cluster_mask) if m]
            
            if not cluster_regions:
                continue
                
            # Aggregate cluster info
            center = {
                var: float(np.mean([r["operating_point"][var] for r in cluster_regions]))
                for var in self.INPUT_VARS
            }
            
            clustered.append({
                "center": center,
                "size": len(cluster_regions),
                "mean_sensitivity": float(np.mean([r["sensitivity_norm"] for r in cluster_regions])),
                "max_sensitivity": float(np.max([r["sensitivity_norm"] for r in cluster_regions])),
                "bounds": {
                    var: {
                        "min": float(np.min([r["operating_point"][var] for r in cluster_regions])),
                        "max": float(np.max([r["operating_point"][var] for r in cluster_regions]))
                    }
                    for var in self.INPUT_VARS
                }
            })
            
        return clustered
    
    def analyze_cdu(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model"
    ) -> SensitivityResult:
        """
        Complete sensitivity analysis for a single CDU.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            
        Returns:
            SensitivityResult with all sensitivity metrics
        """
        result = SensitivityResult(model_name=model_name, cdu_id=cdu_id)
        
        # Compute average sensitivity matrix
        logger.info(f"Computing sensitivity matrix for CDU {cdu_id}")
        mean_jacobian, operating_points = self.compute_sensitivity_matrix(
            data, cdu_id, model_name
        )
        result.mean_sensitivity_matrix = mean_jacobian
        result.jacobian_operating_points = operating_points
        
        # Rank sensitivities
        result.sensitivity_rankings = self.rank_sensitivities(mean_jacobian)
        
        # Compute Sobol indices if requested
        if self.config.compute_sobol:
            logger.info(f"Computing Sobol indices for CDU {cdu_id}")
            first_order, total_order = self.compute_sobol_indices(
                data, cdu_id, model_name
            )
            result.first_order_sobol = first_order
            result.total_order_sobol = total_order
            
        # Identify high-sensitivity regions
        logger.info(f"Identifying high-sensitivity regions for CDU {cdu_id}")
        result.high_sensitivity_regions = self.identify_high_sensitivity_regions(
            data, cdu_id, model_name
        )
        
        # Compute summary statistics
        result.sensitivity_stats = {
            "jacobian_frobenius_norm": float(np.linalg.norm(mean_jacobian)),
            "max_sensitivity": float(np.max(np.abs(mean_jacobian))),
            "condition_number": float(np.linalg.cond(mean_jacobian)) if mean_jacobian.shape[0] == mean_jacobian.shape[1] else None,
            "n_operating_points": len(operating_points),
            "n_high_sensitivity_regions": len(result.high_sensitivity_regions)
        }
        
        # Cache result
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][cdu_id] = result
        
        return result
    
    def compare_sensitivities_across_models(
        self,
        model_results: Dict[str, SensitivityResult]
    ) -> Dict[str, Any]:
        """
        Compare sensitivity profiles across different cooling models.
        
        Args:
            model_results: Dict mapping model name to SensitivityResult
            
        Returns:
            Comparison metrics
        """
        comparison = {
            "models": list(model_results.keys()),
            "jacobian_comparison": {},
            "ranking_agreement": {},
            "sobol_comparison": {}
        }
        
        models = list(model_results.keys())
        
        if len(models) < 2:
            return comparison
            
        # Compare Jacobians
        jacobians = {}
        for model, result in model_results.items():
            if result.mean_sensitivity_matrix is not None:
                jacobians[model] = result.mean_sensitivity_matrix
                
        if len(jacobians) >= 2:
            # Compute pairwise differences
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    if m1 in jacobians and m2 in jacobians:
                        diff = np.abs(jacobians[m1] - jacobians[m2])
                        comparison["jacobian_comparison"][f"{m1}_vs_{m2}"] = {
                            "max_diff": float(np.max(diff)),
                            "mean_diff": float(np.mean(diff)),
                            "relative_diff": float(np.mean(diff) / np.mean(np.abs(jacobians[m1]) + np.abs(jacobians[m2]) + 1e-10))
                        }
                        
        # Compare sensitivity rankings
        for output in self.OUTPUT_VARS:
            rankings = {}
            for model, result in model_results.items():
                if output in result.sensitivity_rankings:
                    rankings[model] = [x[0] for x in result.sensitivity_rankings[output]]
                    
            if len(rankings) >= 2:
                # Compute Kendall's tau between rankings
                from scipy.stats import kendalltau
                
                for i, m1 in enumerate(models):
                    for m2 in models[i+1:]:
                        if m1 in rankings and m2 in rankings:
                            rank1 = [rankings[m1].index(v) for v in self.INPUT_VARS]
                            rank2 = [rankings[m2].index(v) for v in self.INPUT_VARS]
                            tau, p_value = kendalltau(rank1, rank2)
                            
                            key = f"{output}_{m1}_vs_{m2}"
                            comparison["ranking_agreement"][key] = {
                                "kendall_tau": float(tau),
                                "p_value": float(p_value)
                            }
                            
        return comparison
    
    def _get_input_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get input column names for a CDU."""
        return {
            "Q_flow": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total",
            "T_Air": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_T_Air",
            "T_ext": "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext"  # Global input
        }
        
    def _get_output_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get output column names for a CDU."""
        base = f"simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary"
        return {
            "V_flow_prim_GPM": f"{base}.V_flow_prim_GPM",
            "V_flow_sec_GPM": f"{base}.V_flow_sec_GPM",
            "W_flow_CDUP_kW": f"{base}.W_flow_CDUP_kW",
            "T_prim_s_C": f"{base}.T_prim_s_C",
            "T_prim_r_C": f"{base}.T_prim_r_C",
            "T_sec_s_C": f"{base}.T_sec_s_C",
            "T_sec_r_C": f"{base}.T_sec_r_C",
            "p_prim_s_psig": f"{base}.p_prim_s_psig",
            "p_prim_r_psig": f"{base}.p_prim_r_psig",
            "p_sec_s_psig": f"{base}.p_sec_s_psig",
            "p_sec_r_psig": f"{base}.p_sec_r_psig"
        }
        
    def to_dataframe(self, result: SensitivityResult) -> pd.DataFrame:
        """Convert sensitivity result to DataFrame."""
        rows = []
        
        if result.mean_sensitivity_matrix is not None:
            for i, output in enumerate(self.OUTPUT_VARS):
                for j, input_var in enumerate(self.INPUT_VARS):
                    rows.append({
                        "model": result.model_name,
                        "cdu_id": result.cdu_id,
                        "output": output,
                        "input": input_var,
                        "sensitivity": result.mean_sensitivity_matrix[i, j]
                    })
                    
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all sensitivity results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, cdu_results in self.results.items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save each CDU result
            all_rows = []
            for cdu_id, result in cdu_results.items():
                df = self.to_dataframe(result)
                all_rows.append(df)
                
                # Save detailed result
                result_dict = {
                    "cdu_id": cdu_id,
                    "sensitivity_rankings": result.sensitivity_rankings,
                    "first_order_sobol": result.first_order_sobol,
                    "total_order_sobol": result.total_order_sobol,
                    "high_sensitivity_regions": result.high_sensitivity_regions,
                    "stats": result.sensitivity_stats
                }
                
                import json
                with open(model_dir / f"cdu_{cdu_id}_sensitivity.json", "w") as f:
                    json.dump(result_dict, f, indent=2)
                    
            # Save combined DataFrame
            if all_rows:
                combined_df = pd.concat(all_rows, ignore_index=True)
                combined_df.to_csv(model_dir / "all_sensitivities.csv", index=False)
