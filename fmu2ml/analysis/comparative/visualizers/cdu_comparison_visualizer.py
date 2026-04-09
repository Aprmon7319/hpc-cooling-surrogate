"""
CDU Comparison Visualizer for Comparative Analysis.

Creates visualizations comparing CDU behavior across cooling models,
including sensitivity heatmaps, response comparisons, and gain matrices.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Matplotlib imports with backend handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualization disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class CDUComparisonVisualizer:
    """
    Visualizes CDU-level analysis results for cross-model comparison.
    
    Provides:
    1. Sensitivity heatmaps
    2. DC gain matrix comparisons
    3. Step response overlays
    4. Pole-zero plots
    5. RGA visualizations
    6. Operating regime maps
    """
    
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    OUTPUT_VARS = [
        "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW",
        "T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C",
        "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig", "p_sec_r_psig"
    ]
    
    # Nice labels for variables
    INPUT_LABELS = {
        "Q_flow": "Heat Load (kW)",
        "T_Air": "Air Temp (K)",
        "T_ext": "External Temp (K)"
    }
    
    OUTPUT_LABELS = {
        "V_flow_prim_GPM": "Primary Flow (GPM)",
        "V_flow_sec_GPM": "Secondary Flow (GPM)",
        "W_flow_CDUP_kW": "CDU Power (kW)",
        "T_prim_s_C": "T_prim Supply (°C)",
        "T_prim_r_C": "T_prim Return (°C)",
        "T_sec_s_C": "T_sec Supply (°C)",
        "T_sec_r_C": "T_sec Return (°C)",
        "p_prim_s_psig": "P_prim Supply (psig)",
        "p_prim_r_psig": "P_prim Return (psig)",
        "p_sec_s_psig": "P_sec Supply (psig)",
        "p_sec_r_psig": "P_sec Return (psig)"
    }
    
    # Color schemes for models
    MODEL_COLORS = {
        "marconi100": "#1f77b4",  # Blue
        "summit": "#ff7f0e",      # Orange
        "frontier": "#2ca02c",    # Green
    }
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            figsize: Default figure size
            dpi: Figure resolution
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except Exception:
                pass  # Use default style
                
    def plot_sensitivity_heatmap(
        self,
        sensitivity_matrix: np.ndarray,
        model_name: str,
        cdu_id: int,
        title: Optional[str] = None,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot sensitivity matrix as heatmap.
        
        Args:
            sensitivity_matrix: (n_outputs x n_inputs) sensitivity values
            model_name: Model identifier
            cdu_id: CDU identifier
            title: Optional custom title
            save: Whether to save the figure
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Normalize for better visualization
        max_val = np.max(np.abs(sensitivity_matrix))
        if max_val > 0:
            norm_matrix = sensitivity_matrix / max_val
        else:
            norm_matrix = sensitivity_matrix
            
        # Create heatmap
        n_out, n_in = sensitivity_matrix.shape
        n_out = min(n_out, len(self.OUTPUT_VARS))
        n_in = min(n_in, len(self.INPUT_VARS))
        
        if SEABORN_AVAILABLE:
            sns.heatmap(
                norm_matrix[:n_out, :n_in],
                ax=ax,
                xticklabels=[self.INPUT_LABELS.get(v, v) for v in self.INPUT_VARS[:n_in]],
                yticklabels=[self.OUTPUT_LABELS.get(v, v) for v in self.OUTPUT_VARS[:n_out]],
                cmap="RdBu_r",
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Normalized Sensitivity"}
            )
        else:
            im = ax.imshow(norm_matrix[:n_out, :n_in], cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(n_in))
            ax.set_yticks(range(n_out))
            ax.set_xticklabels([self.INPUT_LABELS.get(v, v) for v in self.INPUT_VARS[:n_in]], rotation=45, ha="right")
            ax.set_yticklabels([self.OUTPUT_LABELS.get(v, v) for v in self.OUTPUT_VARS[:n_out]])
            plt.colorbar(im, ax=ax, label="Normalized Sensitivity")
            
        title = title or f"Sensitivity Matrix - {model_name} CDU {cdu_id}"
        ax.set_title(title)
        ax.set_xlabel("Inputs")
        ax.set_ylabel("Outputs")
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"sensitivity_heatmap_{model_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_sensitivity_comparison(
        self,
        sensitivity_matrices: Dict[str, np.ndarray],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare sensitivity matrices across models.
        
        Args:
            sensitivity_matrices: Dict mapping model name to sensitivity matrix
            cdu_id: CDU identifier
            save: Whether to save the figure
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not sensitivity_matrices:
            return None
            
        n_models = len(sensitivity_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
        
        if n_models == 1:
            axes = [axes]
            
        for ax, (model_name, matrix) in zip(axes, sensitivity_matrices.items()):
            max_val = np.max(np.abs(matrix))
            norm_matrix = matrix / max_val if max_val > 0 else matrix
            
            n_out = min(matrix.shape[0], len(self.OUTPUT_VARS))
            n_in = min(matrix.shape[1], len(self.INPUT_VARS))
            
            im = ax.imshow(norm_matrix[:n_out, :n_in], cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(n_in))
            ax.set_yticks(range(n_out))
            ax.set_xticklabels([self.INPUT_VARS[i][:6] for i in range(n_in)], rotation=45, ha="right")
            ax.set_yticklabels([self.OUTPUT_VARS[i][:10] for i in range(n_out)])
            ax.set_title(f"{model_name}")
            ax.set_xlabel("Inputs")
            if ax == axes[0]:
                ax.set_ylabel("Outputs")
                
        # Add colorbar
        fig.colorbar(im, ax=axes, label="Normalized Sensitivity", shrink=0.6)
        fig.suptitle(f"Sensitivity Comparison - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"sensitivity_comparison_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_dc_gain_comparison(
        self,
        gain_matrices: Dict[str, np.ndarray],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare DC gain matrices across models.
        
        Args:
            gain_matrices: Dict mapping model name to DC gain matrix
            cdu_id: CDU identifier
            save: Whether to save the figure
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not gain_matrices:
            return None
            
        # Create bar chart for each output
        models = list(gain_matrices.keys())
        n_models = len(models)
        
        # Get dimensions
        sample_matrix = list(gain_matrices.values())[0]
        n_out = min(sample_matrix.shape[0], len(self.OUTPUT_VARS))
        n_in = min(sample_matrix.shape[1], len(self.INPUT_VARS))
        
        fig, axes = plt.subplots(n_out, 1, figsize=(10, 2.5 * n_out), sharex=True)
        if n_out == 1:
            axes = [axes]
            
        x = np.arange(n_in)
        width = 0.8 / n_models
        
        for i, (ax, output) in enumerate(zip(axes, self.OUTPUT_VARS[:n_out])):
            for j, model in enumerate(models):
                gains = gain_matrices[model][i, :n_in]
                offset = (j - n_models / 2 + 0.5) * width
                color = self.MODEL_COLORS.get(model, f"C{j}")
                ax.bar(x + offset, gains, width, label=model if i == 0 else None, color=color)
                
            ax.set_ylabel(self.OUTPUT_LABELS.get(output, output)[:20])
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax.grid(axis="y", alpha=0.3)
            
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([self.INPUT_LABELS.get(v, v) for v in self.INPUT_VARS[:n_in]])
        axes[-1].set_xlabel("Inputs")
        axes[0].legend(loc="upper right")
        
        fig.suptitle(f"DC Gain Comparison - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"dc_gain_comparison_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_step_response_comparison(
        self,
        step_responses: Dict[str, Dict[str, Any]],
        cdu_id: int,
        input_var: str,
        output_vars: Optional[List[str]] = None,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare step responses across models.
        
        Args:
            step_responses: Dict[model] -> Dict[output] -> {time, response}
            cdu_id: CDU identifier
            input_var: Input that was stepped
            output_vars: Outputs to plot (None for all)
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not step_responses:
            return None
            
        output_vars = output_vars or self.OUTPUT_VARS
        n_outputs = len(output_vars)
        
        # Guard against empty outputs
        if n_outputs == 0:
            logger.warning(f"No output variables to plot for CDU {cdu_id}")
            return None
        
        n_cols = min(3, n_outputs)
        n_rows = (n_outputs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        
        for idx, output in enumerate(output_vars):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            for model, responses in step_responses.items():
                if output in responses:
                    resp = responses[output]
                    time = resp.get("time", np.arange(len(resp.get("response", []))))
                    response = resp.get("response", resp.get("step_response", []))
                    
                    if len(response) > 0:
                        color = self.MODEL_COLORS.get(model, None)
                        ax.plot(time, response, label=model, color=color)
                        
            ax.set_title(self.OUTPUT_LABELS.get(output, output)[:25])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Response")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            
        # Hide empty subplots
        for idx in range(n_outputs, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
            
        fig.suptitle(f"Step Response Comparison (Input: {input_var}) - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"step_response_{input_var}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_pole_zero_map(
        self,
        poles_zeros: Dict[str, Dict[str, Any]],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot pole-zero map for transfer functions.
        
        Args:
            poles_zeros: Dict[model] -> {poles: [...], zeros: [...]}
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not poles_zeros:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")
        
        for i, (model, pz) in enumerate(poles_zeros.items()):
            color = self.MODEL_COLORS.get(model, f"C{i}")
            
            # Plot poles
            poles = pz.get("poles", [])
            if poles:
                poles_arr = np.array(poles)
                ax.scatter(
                    np.real(poles_arr), np.imag(poles_arr),
                    marker="x", s=100, color=color, label=f"{model} poles"
                )
                
            # Plot zeros
            zeros = pz.get("zeros", [])
            if zeros:
                zeros_arr = np.array(zeros)
                ax.scatter(
                    np.real(zeros_arr), np.imag(zeros_arr),
                    marker="o", s=100, facecolors="none", edgecolors=color,
                    label=f"{model} zeros"
                )
                
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"Pole-Zero Map - CDU {cdu_id}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"pole_zero_map_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_rga_comparison(
        self,
        rga_matrices: Dict[str, np.ndarray],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare Relative Gain Arrays across models.
        
        Args:
            rga_matrices: Dict mapping model to RGA matrix
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not rga_matrices:
            return None
            
        n_models = len(rga_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
            
        for ax, (model, rga) in zip(axes, rga_matrices.items()):
            n = min(rga.shape[0], 3)  # RGA is square, show 3x3 max
            
            if SEABORN_AVAILABLE:
                sns.heatmap(
                    rga[:n, :n], ax=ax, annot=True, fmt=".2f",
                    cmap="coolwarm", center=1,
                    xticklabels=self.INPUT_VARS[:n],
                    yticklabels=self.OUTPUT_VARS[:n]
                )
            else:
                im = ax.imshow(rga[:n, :n], cmap="coolwarm", vmin=0, vmax=2)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(self.INPUT_VARS[:n])
                ax.set_yticklabels(self.OUTPUT_VARS[:n])
                plt.colorbar(im, ax=ax)
                
            ax.set_title(f"{model}")
            ax.set_xlabel("Inputs")
            if ax == axes[0]:
                ax.set_ylabel("Outputs")
                
        fig.suptitle(f"Relative Gain Array (RGA) Comparison - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"rga_comparison_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_time_constant_comparison(
        self,
        time_constants: Dict[str, Dict[str, float]],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare dominant time constants across models.
        
        Args:
            time_constants: Dict[model] -> Dict[output] -> time_constant
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not time_constants:
            return None
            
        # Prepare data
        models = list(time_constants.keys())
        outputs = set()
        for tc in time_constants.values():
            outputs.update(tc.keys())
        outputs = sorted(list(outputs))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(outputs))
        width = 0.8 / len(models)
        
        for j, model in enumerate(models):
            tc_values = [time_constants[model].get(out, 0) for out in outputs]
            offset = (j - len(models) / 2 + 0.5) * width
            color = self.MODEL_COLORS.get(model, f"C{j}")
            ax.bar(x + offset, tc_values, width, label=model, color=color)
            
        ax.set_xticks(x)
        ax.set_xticklabels([o[:15] for o in outputs], rotation=45, ha="right")
        ax.set_ylabel("Time Constant (s)")
        ax.set_title(f"Dominant Time Constants - CDU {cdu_id}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"time_constants_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_sobol_indices(
        self,
        sobol_results: Dict[str, Dict[str, Dict[str, float]]],
        cdu_id: int,
        order: str = "total",
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot Sobol sensitivity indices.
        
        Args:
            sobol_results: Dict[model] -> Dict[output] -> Dict[input] -> index
            cdu_id: CDU identifier
            order: "first" or "total"
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not sobol_results:
            return None
            
        models = list(sobol_results.keys())
        outputs = set()
        for sr in sobol_results.values():
            outputs.update(sr.keys())
        outputs = sorted(list(outputs))[:6]  # Limit to 6 outputs
        
        # Guard against empty outputs
        if not outputs:
            logger.warning(f"No Sobol outputs to plot for CDU {cdu_id}")
            return None
        
        n_cols = min(3, len(outputs))
        n_rows = (len(outputs) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, output in enumerate(outputs):
            ax = axes[idx]
            
            x = np.arange(len(self.INPUT_VARS))
            width = 0.8 / len(models)
            
            for j, model in enumerate(models):
                indices = sobol_results.get(model, {}).get(output, {})
                values = [indices.get(inp, 0) for inp in self.INPUT_VARS]
                offset = (j - len(models) / 2 + 0.5) * width
                color = self.MODEL_COLORS.get(model, f"C{j}")
                ax.bar(x + offset, values, width, label=model if idx == 0 else None, color=color)
                
            ax.set_xticks(x)
            ax.set_xticklabels(self.INPUT_VARS)
            ax.set_title(output[:20])
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            
        # Hide unused axes
        for idx in range(len(outputs), len(axes)):
            axes[idx].set_visible(False)
            
        axes[0].legend()
        fig.suptitle(f"Sobol Indices ({order}-order) - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"sobol_{order}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def create_summary_dashboard(
        self,
        analysis_results: Dict[str, Any],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Create summary dashboard with key metrics.
        
        Args:
            analysis_results: Combined analysis results
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Placeholder for different panels
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[2, :])
        
        # Panel 1: Model comparison metrics
        if "model_metrics" in analysis_results:
            metrics = analysis_results["model_metrics"]
            models = list(metrics.keys())
            values = [metrics[m].get("mean_sensitivity", 0) for m in models]
            colors = [self.MODEL_COLORS.get(m, "gray") for m in models]
            ax1.bar(models, values, color=colors)
            ax1.set_title("Mean Sensitivity")
            ax1.set_ylabel("Value")
            
        # Panel 2: Sensitivity comparison
        if "sensitivity_matrices" in analysis_results:
            matrices = analysis_results["sensitivity_matrices"]
            models = list(matrices.keys())
            for i, (model, matrix) in enumerate(matrices.items()):
                ax2.plot(np.abs(matrix).mean(axis=0), marker="o", label=model,
                        color=self.MODEL_COLORS.get(model, f"C{i}"))
            ax2.set_xticks(range(len(self.INPUT_VARS)))
            ax2.set_xticklabels(self.INPUT_VARS)
            ax2.set_title("Average Sensitivity to Each Input")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
        # Panel 3: Time constants
        if "time_constants" in analysis_results:
            tc = analysis_results["time_constants"]
            models = list(tc.keys())
            for model in models:
                if model in tc:
                    outputs = list(tc[model].keys())[:8]
                    values = [tc[model][o] for o in outputs]
                    ax3.plot(outputs, values, marker="o", label=model,
                            color=self.MODEL_COLORS.get(model, None))
            ax3.set_title("Dominant Time Constants")
            ax3.set_ylabel("Time (s)")
            ax3.tick_params(axis="x", rotation=45)
            ax3.legend()
            ax3.grid(alpha=0.3)
            
        # Panel 4: Stability summary
        if "stability" in analysis_results:
            stab = analysis_results["stability"]
            models = list(stab.keys())
            margins = [stab[m].get("margin", 0) for m in models]
            colors = ["green" if m > 0 else "red" for m in margins]
            ax4.bar(models, margins, color=colors)
            ax4.axhline(0, color="gray", linestyle="--")
            ax4.set_title("Stability Margin")
            ax4.set_ylabel("Margin")
            
        # Panel 5: Summary table
        ax5.axis("off")
        if "summary_table" in analysis_results:
            table_data = analysis_results["summary_table"]
            table = ax5.table(
                cellText=table_data["data"],
                colLabels=table_data["columns"],
                rowLabels=table_data["rows"],
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax5.set_title("Summary Metrics", pad=20)
            
        fig.suptitle(f"CDU {cdu_id} Analysis Dashboard", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"dashboard_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def close_all(self):
        """Close all matplotlib figures."""
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")
