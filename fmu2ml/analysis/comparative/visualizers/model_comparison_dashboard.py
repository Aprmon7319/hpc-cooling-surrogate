"""
Model Comparison Dashboard for CDU-Level Comparative Analysis.

Creates comprehensive dashboards for comparing cooling model behavior.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class ModelComparisonDashboard:
    """
    Creates comprehensive comparison dashboards for cooling models.
    
    Provides:
    1. Multi-model comparison summaries
    2. Key metrics radar charts
    3. Cross-model statistical comparisons
    4. Efficiency and performance matrices
    """
    
    MODEL_COLORS = {
        "marconi100": "#1f77b4",
        "summit": "#ff7f0e",
        "lassen": "#2ca02c",
        "frontier": "#d62728"
    }
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150
    ):
        """Initialize dashboard."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
    def create_executive_summary(
        self,
        analysis_results: Dict[str, Any],
        save: bool = True
    ) -> Optional[Figure]:
        """
        Create executive summary dashboard.
        
        Args:
            analysis_results: Complete analysis results
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle("CDU-Level Comparative Analysis: Executive Summary", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Panel 1: Model overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_overview(ax1, analysis_results)
        
        # Panel 2: Key metrics comparison (top middle-right)
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_key_metrics(ax2, analysis_results)
        
        # Panel 3: Radar chart (top right)
        ax3 = fig.add_subplot(gs[0, 3], projection='polar')
        self._plot_radar_chart(ax3, analysis_results)
        
        # Panel 4: Sensitivity heatmap (middle left)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_sensitivity_summary(ax4, analysis_results)
        
        # Panel 5: Time constants (middle right)
        ax5 = fig.add_subplot(gs[1, 2:])
        self._plot_time_constants_summary(ax5, analysis_results)
        
        # Panel 6: Efficiency comparison (bottom left)
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_efficiency_comparison(ax6, analysis_results)
        
        # Panel 7: Key findings text (bottom right)
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_key_findings(ax7, analysis_results)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "executive_summary.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def _plot_model_overview(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot model overview panel."""
        ax.axis('off')
        
        models = results.get('models', [])
        n_cdus = results.get('n_cdus_analyzed', 0)
        
        text = f"Models Compared: {len(models)}\n"
        for model in models:
            color = self.MODEL_COLORS.get(model, 'black')
            text += f"• {model}\n"
        text += f"\nCDUs Analyzed: {n_cdus}"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_title("Overview", fontweight='bold')
    
    def _plot_key_metrics(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot key metrics comparison."""
        metrics = results.get('summary_metrics', {})
        
        if not metrics:
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center')
            ax.set_title("Key Metrics", fontweight='bold')
            return
        
        models = list(metrics.keys())
        n_models = len(models)
        
        # Define key metrics to display
        key_metrics = ['mean_sensitivity', 'dominant_time_constant', 'cop_mean', 'stability_margin']
        metric_labels = ['Avg Sensitivity', 'Time Constant (s)', 'COP', 'Stability Margin']
        
        # Prepare data
        x = np.arange(len(key_metrics))
        width = 0.8 / n_models
        
        for i, model in enumerate(models):
            values = []
            for metric in key_metrics:
                val = metrics[model].get(metric, 0)
                values.append(val if val else 0)
            
            offset = (i - n_models / 2 + 0.5) * width
            color = self.MODEL_COLORS.get(model, f'C{i}')
            ax.bar(x + offset, values, width, label=model, color=color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(loc='upper right')
        ax.set_title("Key Metrics Comparison", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_radar_chart(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot radar chart for multi-dimensional comparison."""
        metrics = results.get('normalized_metrics', {})
        
        if not metrics:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            return
        
        categories = ['Sensitivity', 'Speed', 'Efficiency', 'Stability', 'Coverage']
        n_cats = len(categories)
        
        # Angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model, vals in metrics.items():
            values = [vals.get(cat.lower(), 0.5) for cat in categories]
            values += values[:1]  # Complete the circle
            
            color = self.MODEL_COLORS.get(model, None)
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.set_title("Performance Radar", fontweight='bold', pad=20)
    
    def _plot_sensitivity_summary(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot sensitivity heatmap summary."""
        sensitivity = results.get('sensitivity_matrices', {})
        
        if not sensitivity:
            ax.text(0.5, 0.5, "No sensitivity data", ha='center', va='center')
            ax.set_title("Sensitivity Summary", fontweight='bold')
            return
        
        # Average sensitivity across models
        models = list(sensitivity.keys())
        n_models = len(models)
        
        # Get dimensions from first model
        sample = list(sensitivity.values())[0]
        if isinstance(sample, np.ndarray):
            n_out, n_in = sample.shape
        else:
            ax.text(0.5, 0.5, "Invalid data format", ha='center', va='center')
            return
        
        # Compute difference between models (if 2 models)
        if n_models == 2:
            diff = np.abs(sensitivity[models[0]] - sensitivity[models[1]])
            im = ax.imshow(diff, cmap='Reds', aspect='auto')
            ax.set_title(f"Sensitivity Difference: {models[0]} vs {models[1]}", fontweight='bold')
        else:
            # Show first model
            im = ax.imshow(np.abs(sensitivity[models[0]]), cmap='Blues', aspect='auto')
            ax.set_title(f"Sensitivity: {models[0]}", fontweight='bold')
        
        ax.set_xlabel("Inputs")
        ax.set_ylabel("Outputs")
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_time_constants_summary(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot time constants summary."""
        time_constants = results.get('time_constants', {})
        
        if not time_constants:
            ax.text(0.5, 0.5, "No time constant data", ha='center', va='center')
            ax.set_title("Time Constants", fontweight='bold')
            return
        
        models = list(time_constants.keys())
        
        # Collect all outputs
        all_outputs = set()
        for tc in time_constants.values():
            all_outputs.update(tc.keys())
        outputs = sorted(list(all_outputs))[:8]
        
        x = np.arange(len(outputs))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            tc = time_constants[model]
            values = [tc.get(out, 0) for out in outputs]
            offset = (i - len(models) / 2 + 0.5) * width
            color = self.MODEL_COLORS.get(model, f'C{i}')
            ax.bar(x + offset, values, width, label=model, color=color)
        
        ax.set_xticks(x)
        ax.set_xticklabels([o[:10] for o in outputs], rotation=45, ha='right')
        ax.set_ylabel("Time Constant (s)")
        ax.legend()
        ax.set_title("Time Constants by Output", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_efficiency_comparison(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot efficiency comparison."""
        efficiency = results.get('efficiency', {})
        
        if not efficiency:
            ax.text(0.5, 0.5, "No efficiency data", ha='center', va='center')
            ax.set_title("Efficiency Metrics", fontweight='bold')
            return
        
        models = list(efficiency.keys())
        metrics = ['cop_mean', 'thermal_eff', 'hydraulic_eff']
        labels = ['COP', 'Thermal Eff.', 'Hydraulic Eff.']
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [efficiency[model].get(m, 0) for m in metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            color = self.MODEL_COLORS.get(model, f'C{i}')
            ax.bar(x + offset, values, width, label=model, color=color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Value")
        ax.legend()
        ax.set_title("Efficiency Metrics", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_key_findings(self, ax: plt.Axes, results: Dict[str, Any]) -> None:
        """Plot key findings text box."""
        ax.axis('off')
        
        findings = results.get('key_findings', [
            "• Analysis complete",
            "• See individual plots for details"
        ])
        
        text = "KEY FINDINGS\n" + "=" * 30 + "\n\n"
        for finding in findings[:8]:
            text += f"{finding}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title("Key Findings", fontweight='bold')
    
    def create_detailed_comparison(
        self,
        model1_results: Dict[str, Any],
        model2_results: Dict[str, Any],
        model1_name: str,
        model2_name: str,
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Create detailed pairwise comparison.
        
        Args:
            model1_results: First model results
            model2_results: Second model results
            model1_name: First model name
            model2_name: Second model name
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle(f"Detailed Comparison: {model1_name} vs {model2_name} (CDU {cdu_id})",
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Row 1: Sensitivity comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_pairwise_sensitivity(ax1, ax2, ax3, model1_results, model2_results,
                                        model1_name, model2_name)
        
        # Row 2: Gain comparison
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_pairwise_gains(ax4, model1_results, model2_results,
                                  model1_name, model2_name)
        
        # Row 3: Time constants and dynamics
        ax5 = fig.add_subplot(gs[2, :2])
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_pairwise_dynamics(ax5, ax6, model1_results, model2_results,
                                     model1_name, model2_name)
        
        # Row 4: Efficiency and operating regime
        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])
        ax9 = fig.add_subplot(gs[3, 2])
        self._plot_pairwise_efficiency(ax7, ax8, ax9, model1_results, model2_results,
                                       model1_name, model2_name)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"detailed_comparison_{model1_name}_{model2_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def _plot_pairwise_sensitivity(
        self, ax1, ax2, ax3,
        results1: Dict, results2: Dict,
        name1: str, name2: str
    ) -> None:
        """Plot pairwise sensitivity comparison."""
        sens1 = results1.get('sensitivity_matrix', np.zeros((3, 3)))
        sens2 = results2.get('sensitivity_matrix', np.zeros((3, 3)))
        
        # Model 1
        im1 = ax1.imshow(np.abs(sens1), cmap='Blues', aspect='auto')
        ax1.set_title(name1)
        ax1.set_xlabel("Inputs")
        ax1.set_ylabel("Outputs")
        
        # Model 2
        im2 = ax2.imshow(np.abs(sens2), cmap='Oranges', aspect='auto')
        ax2.set_title(name2)
        ax2.set_xlabel("Inputs")
        
        # Difference
        diff = sens1 - sens2
        max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
        if max_diff == 0:
            max_diff = 1
        im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-max_diff, vmax=max_diff)
        ax3.set_title(f"Difference ({name1} - {name2})")
        ax3.set_xlabel("Inputs")
        plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    def _plot_pairwise_gains(
        self, ax,
        results1: Dict, results2: Dict,
        name1: str, name2: str
    ) -> None:
        """Plot pairwise DC gain comparison."""
        gains1 = results1.get('dc_gains', {})
        gains2 = results2.get('dc_gains', {})
        
        all_pairs = set(gains1.keys()) | set(gains2.keys())
        pairs = sorted(list(all_pairs))[:10]
        
        x = np.arange(len(pairs))
        width = 0.35
        
        vals1 = [gains1.get(p, 0) for p in pairs]
        vals2 = [gains2.get(p, 0) for p in pairs]
        
        ax.bar(x - width/2, vals1, width, label=name1, color=self.MODEL_COLORS.get(name1, 'blue'))
        ax.bar(x + width/2, vals2, width, label=name2, color=self.MODEL_COLORS.get(name2, 'orange'))
        
        ax.set_xticks(x)
        ax.set_xticklabels([p[:15] for p in pairs], rotation=45, ha='right')
        ax.set_ylabel("DC Gain")
        ax.legend()
        ax.set_title("DC Gain Comparison")
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_pairwise_dynamics(
        self, ax1, ax2,
        results1: Dict, results2: Dict,
        name1: str, name2: str
    ) -> None:
        """Plot pairwise dynamics comparison."""
        tc1 = results1.get('time_constants', {})
        tc2 = results2.get('time_constants', {})
        
        # Time constants bar chart
        outputs = sorted(set(tc1.keys()) | set(tc2.keys()))[:8]
        x = np.arange(len(outputs))
        width = 0.35
        
        vals1 = [tc1.get(o, 0) for o in outputs]
        vals2 = [tc2.get(o, 0) for o in outputs]
        
        ax1.bar(x - width/2, vals1, width, label=name1, color=self.MODEL_COLORS.get(name1, 'blue'))
        ax1.bar(x + width/2, vals2, width, label=name2, color=self.MODEL_COLORS.get(name2, 'orange'))
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([o[:10] for o in outputs], rotation=45, ha='right')
        ax1.set_ylabel("Time Constant (s)")
        ax1.legend()
        ax1.set_title("Time Constants")
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter plot of time constants
        common = set(tc1.keys()) & set(tc2.keys())
        if common:
            x_vals = [tc1[k] for k in common]
            y_vals = [tc2[k] for k in common]
            ax2.scatter(x_vals, y_vals, alpha=0.7)
            
            # Add identity line
            max_val = max(max(x_vals), max(y_vals))
            ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
        
        ax2.set_xlabel(f"{name1} (s)")
        ax2.set_ylabel(f"{name2} (s)")
        ax2.set_title("Time Constant Correlation")
        ax2.legend()
    
    def _plot_pairwise_efficiency(
        self, ax1, ax2, ax3,
        results1: Dict, results2: Dict,
        name1: str, name2: str
    ) -> None:
        """Plot pairwise efficiency comparison."""
        eff1 = results1.get('efficiency', {})
        eff2 = results2.get('efficiency', {})
        
        # COP comparison
        cop1 = eff1.get('cop_mean', 0)
        cop2 = eff2.get('cop_mean', 0)
        ax1.bar([name1, name2], [cop1, cop2], 
                color=[self.MODEL_COLORS.get(name1, 'blue'), self.MODEL_COLORS.get(name2, 'orange')])
        ax1.set_ylabel("COP")
        ax1.set_title("Coefficient of Performance")
        
        # Thermal efficiency
        th1 = eff1.get('thermal_eff', 0)
        th2 = eff2.get('thermal_eff', 0)
        ax2.bar([name1, name2], [th1, th2],
                color=[self.MODEL_COLORS.get(name1, 'blue'), self.MODEL_COLORS.get(name2, 'orange')])
        ax2.set_ylabel("Efficiency")
        ax2.set_title("Thermal Efficiency")
        
        # Operating envelope volume
        vol1 = results1.get('envelope_volume', 0)
        vol2 = results2.get('envelope_volume', 0)
        ax3.bar([name1, name2], [vol1, vol2],
                color=[self.MODEL_COLORS.get(name1, 'blue'), self.MODEL_COLORS.get(name2, 'orange')])
        ax3.set_ylabel("Volume")
        ax3.set_title("Operating Envelope Size")
    
    def close_all(self):
        """Close all matplotlib figures."""
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")
