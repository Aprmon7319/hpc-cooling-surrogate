"""
Transfer Function Analyzer for CDU-Level Comparative Analysis.

Characterizes system dynamics via transfer function modeling, including
gain matrices, pole-zero analysis, and I/O coupling quantification.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize
from scipy.linalg import svd

logger = logging.getLogger(__name__)


@dataclass
class TransferFunctionConfig:
    """Configuration for transfer function analysis."""
    
    # Model identification
    model_order: int = 2  # Default transfer function order
    max_order: int = 5  # Maximum order for order selection
    fit_method: str = "least_squares"  # least_squares, arx, subspace
    
    # Gain matrix
    compute_dc_gain: bool = True
    compute_hf_gain: bool = True  # High-frequency gain
    
    # Pole-zero analysis
    compute_poles_zeros: bool = True
    stability_margin: float = 0.0  # Distance from imaginary axis
    
    # Coupling analysis
    compute_coupling: bool = True
    rga_threshold: float = 0.1  # Relative Gain Array threshold
    
    # Frequency domain
    freq_points: int = 100
    freq_range: Tuple[float, float] = (0.001, 1.0)  # Hz


@dataclass
class TransferFunctionResult:
    """Results from transfer function analysis."""
    
    model_name: str
    cdu_id: int
    
    # Gain matrices
    dc_gain_matrix: Optional[np.ndarray] = None  # Static gain
    hf_gain_matrix: Optional[np.ndarray] = None  # High-frequency gain
    
    # Transfer function models (SISO)
    transfer_functions: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    
    # MIMO system
    mimo_system: Optional[Any] = None
    
    # Pole-zero analysis
    poles: Dict[str, np.ndarray] = field(default_factory=dict)
    zeros: Dict[str, np.ndarray] = field(default_factory=dict)
    dominant_poles: List[complex] = field(default_factory=list)
    stability_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Coupling analysis
    rga: Optional[np.ndarray] = None  # Relative Gain Array
    interaction_measure: Optional[np.ndarray] = None
    coupling_strength: Dict[str, float] = field(default_factory=dict)
    
    # Fit quality
    fit_metrics: Dict[str, float] = field(default_factory=dict)


class TransferFunctionAnalyzer:
    """
    Analyzes I/O relationships via transfer function characterization.
    
    Key capabilities:
    1. DC and high-frequency gain computation
    2. Transfer function identification (SISO)
    3. Pole-zero analysis
    4. I/O coupling via Relative Gain Array
    5. Cross-model transfer function comparison
    """
    
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    OUTPUT_VARS = [
        "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW",
        "T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C",
        "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig", "p_sec_r_psig"
    ]
    
    def __init__(self, config: Optional[TransferFunctionConfig] = None):
        """Initialize transfer function analyzer."""
        self.config = config or TransferFunctionConfig()
        self.results: Dict[str, Dict[int, TransferFunctionResult]] = {}
        
    def compute_dc_gain_matrix(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model"
    ) -> np.ndarray:
        """
        Compute DC (static) gain matrix.
        
        G_dc[i,j] = ∂y_i / ∂u_j at steady state
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            
        Returns:
            DC gain matrix (n_outputs x n_inputs)
        """
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        n_inputs = len(self.INPUT_VARS)
        n_outputs = len(self.OUTPUT_VARS)
        G_dc = np.zeros((n_outputs, n_inputs))
        
        # Use linear regression for DC gain
        from sklearn.linear_model import LinearRegression
        
        # Prepare input data
        X_cols = [input_cols[var] for var in self.INPUT_VARS if input_cols[var] in data.columns]
        if not X_cols:
            logger.warning(f"No input columns found for CDU {cdu_id}")
            return G_dc
            
        X = data[X_cols].dropna()
        
        for i, output_var in enumerate(self.OUTPUT_VARS):
            output_col = output_cols.get(output_var)
            if output_col not in data.columns:
                continue
                
            y = data[output_col]
            
            # Align data
            common_idx = X.index.intersection(y.dropna().index)
            if len(common_idx) < 10:
                continue
                
            X_fit = X.loc[common_idx].values
            y_fit = y.loc[common_idx].values
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X_fit, y_fit)
            
            # Extract gains
            for j, var in enumerate(self.INPUT_VARS):
                if input_cols[var] in X_cols:
                    col_idx = X_cols.index(input_cols[var])
                    G_dc[i, j] = model.coef_[col_idx]
                    
        return G_dc
    
    def identify_transfer_function(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
        order: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Identify SISO transfer function from I/O data.
        
        Args:
            time: Time array
            input_signal: Input signal
            output_signal: Output signal
            order: Transfer function order (numerator and denominator)
            
        Returns:
            Dict with num, den, poles, zeros, dc_gain
        """
        order = order or self.config.model_order
        
        # Estimate sample time
        dt = np.mean(np.diff(time))
        
        # Remove mean (work with deviations)
        u = input_signal - np.mean(input_signal)
        y = output_signal - np.mean(output_signal)
        
        # Normalize
        u_scale = np.std(u) if np.std(u) > 0 else 1.0
        y_scale = np.std(y) if np.std(y) > 0 else 1.0
        u_norm = u / u_scale
        y_norm = y / y_scale
        
        # ARX model identification
        # y[k] = -a1*y[k-1] - ... - an*y[k-n] + b0*u[k] + ... + bn*u[k-n]
        n = len(y_norm)
        n_params = 2 * order + 1
        
        if n < n_params + order:
            return {"error": "Insufficient data"}
            
        # Build regression matrix
        Phi = np.zeros((n - order, n_params))
        Y = y_norm[order:]
        
        for k in range(n - order):
            # AR part
            for i in range(order):
                Phi[k, i] = -y_norm[order + k - 1 - i]
            # X part
            for i in range(order + 1):
                Phi[k, order + i] = u_norm[order + k - i]
                
        # Least squares solution
        try:
            theta, residuals, rank, s = np.linalg.lstsq(Phi, Y, rcond=None)
        except np.linalg.LinAlgError:
            return {"error": "Least squares failed"}
            
        # Extract coefficients
        a = theta[:order]  # AR coefficients
        b = theta[order:]  # X coefficients
        
        # Build transfer function
        # H(z) = (b0 + b1*z^-1 + ...) / (1 + a1*z^-1 + ...)
        num = b * (y_scale / u_scale)  # Scale back
        den = np.concatenate([[1], a])
        
        # Convert to continuous time (if needed)
        try:
            sys_d = signal.TransferFunction(num, den, dt=dt)
            
            # Compute poles and zeros
            poles = np.roots(den)
            zeros = np.roots(num) if len(num) > 1 else np.array([])
            
            # DC gain
            dc_gain = np.sum(num) / np.sum(den) if np.sum(den) != 0 else 0
            
            # Fit quality
            y_pred = signal.lfilter(num, den, u_norm) * y_scale + np.mean(output_signal)
            ss_res = np.sum((output_signal - y_pred) ** 2)
            ss_tot = np.sum((output_signal - np.mean(output_signal)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            return {
                "num": num.tolist(),
                "den": den.tolist(),
                "poles": poles.tolist(),
                "zeros": zeros.tolist(),
                "dc_gain": dc_gain,
                "dt": dt,
                "order": order,
                "r_squared": r_squared
            }
            
        except Exception as e:
            logger.warning(f"Transfer function identification failed: {e}")
            return {"error": str(e)}
            
    def select_model_order(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray
    ) -> int:
        """
        Select optimal transfer function order using AIC/BIC.
        
        Args:
            time: Time array
            input_signal: Input signal
            output_signal: Output signal
            
        Returns:
            Optimal model order
        """
        best_order = 1
        best_aic = np.inf
        
        for order in range(1, self.config.max_order + 1):
            result = self.identify_transfer_function(
                time, input_signal, output_signal, order
            )
            
            if "error" in result:
                continue
                
            # Compute AIC
            n = len(output_signal)
            k = 2 * order + 1  # Number of parameters
            r_squared = result.get("r_squared", 0)
            
            if r_squared > 0:
                # Estimate variance
                mse = (1 - r_squared) * np.var(output_signal)
                aic = n * np.log(mse + 1e-10) + 2 * k
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    
        return best_order
    
    def compute_relative_gain_array(
        self,
        gain_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute Relative Gain Array (RGA) for coupling analysis.
        
        RGA = G .* (G^-T)
        
        Args:
            gain_matrix: DC gain matrix
            
        Returns:
            RGA matrix
        """
        # Need square matrix for RGA
        n_out, n_in = gain_matrix.shape
        n = min(n_out, n_in)
        G = gain_matrix[:n, :n]
        
        try:
            G_inv_T = np.linalg.inv(G).T
            rga = G * G_inv_T  # Element-wise multiplication
        except np.linalg.LinAlgError:
            # Use pseudoinverse
            G_pinv_T = np.linalg.pinv(G).T
            rga = G * G_pinv_T
            
        return rga
    
    def analyze_coupling(
        self,
        gain_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze I/O coupling from gain matrix.
        
        Args:
            gain_matrix: DC gain matrix
            
        Returns:
            Coupling analysis results
        """
        n_out, n_in = gain_matrix.shape
        
        # RGA (for square part)
        n = min(n_out, n_in)
        rga = self.compute_relative_gain_array(gain_matrix[:n, :n])
        
        # Coupling metrics
        results = {
            "rga": rga.tolist(),
            "diagonal_dominance": np.mean(np.diag(np.abs(rga))) / np.mean(np.abs(rga)) if np.mean(np.abs(rga)) > 0 else 0,
            "off_diagonal_sum": np.sum(np.abs(rga)) - np.sum(np.abs(np.diag(rga))),
            "strongly_coupled_pairs": []
        }
        
        # Identify strongly coupled pairs
        for i in range(n):
            for j in range(n):
                if i != j and np.abs(rga[i, j]) > self.config.rga_threshold:
                    results["strongly_coupled_pairs"].append({
                        "input": self.INPUT_VARS[j],
                        "output": self.OUTPUT_VARS[i],
                        "rga_value": float(rga[i, j])
                    })
                    
        # SVD analysis for interaction measure
        try:
            U, S, Vt = svd(gain_matrix[:n, :n])
            condition_number = S[0] / S[-1] if S[-1] > 0 else np.inf
            results["condition_number"] = float(condition_number)
            results["singular_values"] = S.tolist()
        except Exception:
            pass
            
        return results
    
    def analyze_stability(
        self,
        poles: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze system stability from poles.
        
        Args:
            poles: Array of system poles
            
        Returns:
            Stability analysis results
        """
        if len(poles) == 0:
            return {"stable": True, "margin": np.inf}
            
        # For discrete-time: stable if |poles| < 1
        # For continuous-time: stable if Re(poles) < 0
        
        # Assume discrete-time
        pole_magnitudes = np.abs(poles)
        max_magnitude = np.max(pole_magnitudes)
        
        stable = max_magnitude < 1.0
        margin = 1.0 - max_magnitude  # Distance from unit circle
        
        # Find dominant poles (closest to unit circle)
        sorted_idx = np.argsort(-pole_magnitudes)
        dominant = poles[sorted_idx[:min(3, len(poles))]]
        
        # Compute damping and frequency for complex poles
        pole_info = []
        for p in dominant:
            if np.iscomplex(p):
                mag = np.abs(p)
                phase = np.angle(p)
                damping = -np.log(mag)  # Approximate damping
                freq = np.abs(phase)  # Normalized frequency
                pole_info.append({
                    "pole": complex(p),
                    "magnitude": float(mag),
                    "phase_rad": float(phase),
                    "damping": float(damping),
                    "norm_freq": float(freq)
                })
            else:
                pole_info.append({
                    "pole": float(np.real(p)),
                    "magnitude": float(np.abs(p))
                })
                
        return {
            "stable": stable,
            "margin": float(margin),
            "max_pole_magnitude": float(max_magnitude),
            "dominant_poles": pole_info
        }
    
    def analyze_cdu(
        self,
        data: pd.DataFrame,
        cdu_id: int,
        model_name: str = "model",
        time_col: str = "time"
    ) -> TransferFunctionResult:
        """
        Complete transfer function analysis for a CDU.
        
        Args:
            data: DataFrame with CDU data
            cdu_id: CDU identifier
            model_name: Model identifier
            time_col: Time column name
            
        Returns:
            TransferFunctionResult
        """
        result = TransferFunctionResult(model_name=model_name, cdu_id=cdu_id)
        
        # Compute DC gain matrix
        logger.info(f"Computing DC gain matrix for CDU {cdu_id}")
        if self.config.compute_dc_gain:
            result.dc_gain_matrix = self.compute_dc_gain_matrix(data, cdu_id, model_name)
            
        # Get column mappings
        input_cols = self._get_input_columns(cdu_id)
        output_cols = self._get_output_columns(cdu_id)
        
        # Get time array
        time = data[time_col].values if time_col in data.columns else np.arange(len(data))
        
        # Identify transfer functions for each I/O pair
        logger.info(f"Identifying transfer functions for CDU {cdu_id}")
        all_poles = []
        
        for input_var in self.INPUT_VARS:
            input_col = input_cols[input_var]
            if input_col not in data.columns:
                continue
                
            input_signal = data[input_col].values
            result.transfer_functions[input_var] = {}
            
            for output_var in self.OUTPUT_VARS:
                output_col = output_cols[output_var]
                if output_col not in data.columns:
                    continue
                    
                output_signal = data[output_col].values
                
                # Select order and identify
                try:
                    order = self.select_model_order(time, input_signal, output_signal)
                    tf_result = self.identify_transfer_function(
                        time, input_signal, output_signal, order
                    )
                    
                    result.transfer_functions[input_var][output_var] = tf_result
                    
                    # Collect poles
                    if "poles" in tf_result:
                        poles = np.array(tf_result["poles"])
                        result.poles[f"{input_var}_{output_var}"] = poles
                        all_poles.extend(poles)
                        
                    if "zeros" in tf_result:
                        result.zeros[f"{input_var}_{output_var}"] = np.array(tf_result["zeros"])
                        
                except Exception as e:
                    logger.warning(f"TF identification failed for {input_var}->{output_var}: {e}")
                    
        # Find dominant poles across all channels
        if all_poles:
            all_poles = np.array(all_poles)
            sorted_idx = np.argsort(-np.abs(all_poles))
            result.dominant_poles = all_poles[sorted_idx[:5]].tolist()
            
            # Stability analysis
            result.stability_analysis = self.analyze_stability(all_poles)
            
        # Coupling analysis
        if self.config.compute_coupling and result.dc_gain_matrix is not None:
            logger.info(f"Analyzing coupling for CDU {cdu_id}")
            coupling = self.analyze_coupling(result.dc_gain_matrix)
            result.rga = np.array(coupling.get("rga", []))
            result.coupling_strength = {
                "diagonal_dominance": coupling.get("diagonal_dominance", 0),
                "condition_number": coupling.get("condition_number", 0),
                "n_strong_couplings": len(coupling.get("strongly_coupled_pairs", []))
            }
            
        # Compute fit metrics
        r_squared_values = []
        for input_var, outputs in result.transfer_functions.items():
            for output_var, tf_data in outputs.items():
                if "r_squared" in tf_data:
                    r_squared_values.append(tf_data["r_squared"])
                    
        if r_squared_values:
            result.fit_metrics = {
                "mean_r_squared": float(np.mean(r_squared_values)),
                "min_r_squared": float(np.min(r_squared_values)),
                "max_r_squared": float(np.max(r_squared_values))
            }
            
        # Cache result
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][cdu_id] = result
        
        return result
    
    def compare_transfer_functions_across_models(
        self,
        model_results: Dict[str, TransferFunctionResult]
    ) -> Dict[str, Any]:
        """
        Compare transfer functions across cooling models.
        
        Args:
            model_results: Dict mapping model name to TransferFunctionResult
            
        Returns:
            Comparison metrics
        """
        comparison = {
            "models": list(model_results.keys()),
            "dc_gain_comparison": {},
            "pole_comparison": {},
            "coupling_comparison": {},
            "stability_comparison": {}
        }
        
        models = list(model_results.keys())
        
        if len(models) < 2:
            return comparison
            
        # Compare DC gains
        gains = {}
        for model, result in model_results.items():
            if result.dc_gain_matrix is not None:
                gains[model] = result.dc_gain_matrix
                
        if len(gains) >= 2:
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    if m1 in gains and m2 in gains:
                        diff = np.abs(gains[m1] - gains[m2])
                        comparison["dc_gain_comparison"][f"{m1}_vs_{m2}"] = {
                            "max_diff": float(np.max(diff)),
                            "mean_diff": float(np.mean(diff)),
                            "relative_diff": float(
                                np.mean(diff) / (np.mean(np.abs(gains[m1]) + np.abs(gains[m2])) / 2 + 1e-10)
                            )
                        }
                        
        # Compare poles
        for model, result in model_results.items():
            comparison["pole_comparison"][model] = {
                "dominant_poles": [complex(p) if np.iscomplex(p) else float(p) for p in result.dominant_poles],
                "n_poles": len(result.dominant_poles),
                "stable": result.stability_analysis.get("stable", True),
                "stability_margin": result.stability_analysis.get("margin", 0)
            }
            
        # Compare coupling
        for model, result in model_results.items():
            comparison["coupling_comparison"][model] = result.coupling_strength
            
        return comparison
    
    def _get_input_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get input column names for a CDU."""
        return {
            "Q_flow": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total",
            "T_Air": f"simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_T_Air",
            "T_ext": "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext"
        }
        
    def _get_output_columns(self, cdu_id: int) -> Dict[str, str]:
        """Get output column names for a CDU."""
        base = f"simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary"
        return {var: f"{base}.{var}" for var in self.OUTPUT_VARS}
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert DC gain results to DataFrame."""
        rows = []
        
        for model_name, cdu_results in self.results.items():
            for cdu_id, result in cdu_results.items():
                if result.dc_gain_matrix is not None:
                    for i, output in enumerate(self.OUTPUT_VARS):
                        for j, input_var in enumerate(self.INPUT_VARS):
                            rows.append({
                                "model": model_name,
                                "cdu_id": cdu_id,
                                "input": input_var,
                                "output": output,
                                "dc_gain": result.dc_gain_matrix[i, j]
                            })
                            
        return pd.DataFrame(rows)
    
    def to_transfer_function_summary(self) -> pd.DataFrame:
        """Create summary of all identified transfer functions."""
        rows = []
        
        for model_name, cdu_results in self.results.items():
            for cdu_id, result in cdu_results.items():
                for input_var, outputs in result.transfer_functions.items():
                    for output_var, tf_data in outputs.items():
                        if "error" not in tf_data:
                            rows.append({
                                "model": model_name,
                                "cdu_id": cdu_id,
                                "input": input_var,
                                "output": output_var,
                                "order": tf_data.get("order", 0),
                                "dc_gain": tf_data.get("dc_gain", 0),
                                "n_poles": len(tf_data.get("poles", [])),
                                "n_zeros": len(tf_data.get("zeros", [])),
                                "r_squared": tf_data.get("r_squared", 0)
                            })
                            
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DC gains
        df_gains = self.to_dataframe()
        if not df_gains.empty:
            df_gains.to_csv(output_dir / "dc_gains.csv", index=False)
            
        # Save transfer function summary
        df_tf = self.to_transfer_function_summary()
        if not df_tf.empty:
            df_tf.to_csv(output_dir / "transfer_functions_summary.csv", index=False)
            
        # Save detailed results
        import json
        
        for model_name, cdu_results in self.results.items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            for cdu_id, result in cdu_results.items():
                result_dict = {
                    "cdu_id": cdu_id,
                    "model": model_name,
                    "dc_gain_matrix": result.dc_gain_matrix.tolist() if result.dc_gain_matrix is not None else None,
                    "transfer_functions": result.transfer_functions,
                    "stability_analysis": result.stability_analysis,
                    "coupling_strength": result.coupling_strength,
                    "fit_metrics": result.fit_metrics
                }
                
                with open(model_dir / f"cdu_{cdu_id}_transfer_functions.json", "w") as f:
                    json.dump(result_dict, f, indent=2, default=str)
