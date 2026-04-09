"""
Response Surface Visualizer for CDU-Level Comparative Analysis.

Creates 2D and 3D response surface plots for comparing model behavior.
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
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualization disabled")


class ResponseSurfaceVisualizer:
    """
    Visualizes response surfaces for comparing I/O relationships across models.
    
    Provides:
    1. 2D contour plots
    2. 3D surface plots
    3. Response surface difference plots
    4. Operating point overlay
    """
    
    INPUT_VARS = ["Q_flow", "T_Air", "T_ext"]
    
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
        "T_sec_r_C": "T_sec Return (°C)"
    }
    
    MODEL_COLORS = {
        "marconi100": "#1f77b4",
        "summit": "#ff7f0e",
        "lassen": "#2ca02c",
        "frontier": "#d62728"
    }
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150
    ):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_response_surface_2d(
        self,
        response_data: Dict[str, np.ndarray],
        model_name: str,
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot 2D contour response surface.
        
        Args:
            response_data: Dict with x1_centers, x2_centers, response_grid, x1_name, x2_name, output_name
            model_name: Model identifier
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x1 = response_data.get('x1_centers', np.array([]))
        x2 = response_data.get('x2_centers', np.array([]))
        Z = response_data.get('response_grid', np.array([[]]))
        
        if len(x1) == 0 or len(x2) == 0:
            return None
            
        X1, X2 = np.meshgrid(x1, x2)
        
        # Create contour plot
        levels = 20
        contour = ax.contourf(X1, X2, Z.T, levels=levels, cmap='viridis')
        ax.contour(X1, X2, Z.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        plt.colorbar(contour, ax=ax, label=self.OUTPUT_LABELS.get(
            response_data.get('output_name', 'Output'), 'Output'))
        
        ax.set_xlabel(self.INPUT_LABELS.get(response_data.get('x1_name', 'x1'), 'Input 1'))
        ax.set_ylabel(self.INPUT_LABELS.get(response_data.get('x2_name', 'x2'), 'Input 2'))
        ax.set_title(f"Response Surface: {response_data.get('output_name', 'Output')}\n"
                    f"{model_name} - CDU {cdu_id}")
        
        plt.tight_layout()
        
        if save:
            x1_name = response_data.get('x1_name', 'x1')
            x2_name = response_data.get('x2_name', 'x2')
            output_name = response_data.get('output_name', 'output')
            filepath = self.output_dir / f"response_2d_{model_name}_{output_name}_{x1_name}_{x2_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_response_surface_3d(
        self,
        response_data: Dict[str, np.ndarray],
        model_name: str,
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot 3D surface response.
        
        Args:
            response_data: Dict with x1_centers, x2_centers, response_grid
            model_name: Model identifier
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x1 = response_data.get('x1_centers', np.array([]))
        x2 = response_data.get('x2_centers', np.array([]))
        Z = response_data.get('response_grid', np.array([[]]))
        
        if len(x1) == 0 or len(x2) == 0:
            return None
            
        X1, X2 = np.meshgrid(x1, x2)
        
        # Create surface plot
        surf = ax.plot_surface(X1, X2, Z.T, cmap='viridis', 
                               edgecolor='none', alpha=0.8)
        
        ax.set_xlabel(self.INPUT_LABELS.get(response_data.get('x1_name', 'x1'), 'Input 1'))
        ax.set_ylabel(self.INPUT_LABELS.get(response_data.get('x2_name', 'x2'), 'Input 2'))
        ax.set_zlabel(self.OUTPUT_LABELS.get(response_data.get('output_name', 'Output'), 'Output'))
        ax.set_title(f"3D Response Surface: {response_data.get('output_name', 'Output')}\n"
                    f"{model_name} - CDU {cdu_id}")
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        if save:
            x1_name = response_data.get('x1_name', 'x1')
            x2_name = response_data.get('x2_name', 'x2')
            output_name = response_data.get('output_name', 'output')
            filepath = self.output_dir / f"response_3d_{model_name}_{output_name}_{x1_name}_{x2_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_response_comparison(
        self,
        response_data_dict: Dict[str, Dict[str, np.ndarray]],
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Compare response surfaces across models.
        
        Args:
            response_data_dict: Dict[model_name] -> response_data
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE or not response_data_dict:
            return None
            
        n_models = len(response_data_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
            
        # Get common output name
        sample_data = list(response_data_dict.values())[0]
        output_name = sample_data.get('output_name', 'Output')
        x1_name = sample_data.get('x1_name', 'x1')
        x2_name = sample_data.get('x2_name', 'x2')
        
        # Find global min/max for consistent colorbar
        vmin, vmax = np.inf, -np.inf
        for data in response_data_dict.values():
            Z = data.get('response_grid', np.array([[]]))
            if Z.size > 0:
                vmin = min(vmin, np.nanmin(Z))
                vmax = max(vmax, np.nanmax(Z))
        
        for ax, (model_name, data) in zip(axes, response_data_dict.items()):
            x1 = data.get('x1_centers', np.array([]))
            x2 = data.get('x2_centers', np.array([]))
            Z = data.get('response_grid', np.array([[]]))
            
            if len(x1) == 0 or len(x2) == 0:
                continue
                
            X1, X2 = np.meshgrid(x1, x2)
            
            contour = ax.contourf(X1, X2, Z.T, levels=15, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(model_name)
            ax.set_xlabel(self.INPUT_LABELS.get(x1_name, 'Input 1'))
            if ax == axes[0]:
                ax.set_ylabel(self.INPUT_LABELS.get(x2_name, 'Input 2'))
        
        # Add colorbar
        fig.colorbar(contour, ax=axes, label=self.OUTPUT_LABELS.get(output_name, 'Output'))
        fig.suptitle(f"Response Surface Comparison: {output_name} - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"response_comparison_{output_name}_{x1_name}_{x2_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_response_difference(
        self,
        response_data_1: Dict[str, np.ndarray],
        response_data_2: Dict[str, np.ndarray],
        model_1: str,
        model_2: str,
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot difference between two response surfaces.
        
        Args:
            response_data_1: First model's response data
            response_data_2: Second model's response data
            model_1: First model name
            model_2: Second model name
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x1 = response_data_1.get('x1_centers', np.array([]))
        x2 = response_data_1.get('x2_centers', np.array([]))
        Z1 = response_data_1.get('response_grid', np.array([[]]))
        Z2 = response_data_2.get('response_grid', np.array([[]]))
        
        if len(x1) == 0 or Z1.shape != Z2.shape:
            return None
            
        X1, X2 = np.meshgrid(x1, x2)
        
        # Compute difference
        Z_diff = Z1 - Z2
        
        # Get output name
        output_name = response_data_1.get('output_name', 'Output')
        x1_name = response_data_1.get('x1_name', 'x1')
        x2_name = response_data_1.get('x2_name', 'x2')
        
        # Find global range for models
        vmin = min(np.nanmin(Z1), np.nanmin(Z2))
        vmax = max(np.nanmax(Z1), np.nanmax(Z2))
        
        # Plot model 1
        c1 = axes[0].contourf(X1, X2, Z1.T, levels=15, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(model_1)
        axes[0].set_xlabel(self.INPUT_LABELS.get(x1_name, 'Input 1'))
        axes[0].set_ylabel(self.INPUT_LABELS.get(x2_name, 'Input 2'))
        
        # Plot model 2
        c2 = axes[1].contourf(X1, X2, Z2.T, levels=15, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(model_2)
        axes[1].set_xlabel(self.INPUT_LABELS.get(x1_name, 'Input 1'))
        
        # Plot difference
        diff_max = max(abs(np.nanmin(Z_diff)), abs(np.nanmax(Z_diff)))
        c3 = axes[2].contourf(X1, X2, Z_diff.T, levels=15, cmap='RdBu_r', 
                              vmin=-diff_max, vmax=diff_max)
        axes[2].set_title(f"Difference ({model_1} - {model_2})")
        axes[2].set_xlabel(self.INPUT_LABELS.get(x1_name, 'Input 1'))
        
        fig.colorbar(c1, ax=axes[:2], label=self.OUTPUT_LABELS.get(output_name, 'Output'), shrink=0.8)
        fig.colorbar(c3, ax=axes[2], label='Difference', shrink=0.8)
        
        fig.suptitle(f"Response Surface Difference: {output_name} - CDU {cdu_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"response_diff_{model_1}_{model_2}_{output_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def plot_operating_points_overlay(
        self,
        response_data: Dict[str, np.ndarray],
        operating_points: np.ndarray,
        model_name: str,
        cdu_id: int,
        save: bool = True
    ) -> Optional[Figure]:
        """
        Plot response surface with operating points overlaid.
        
        Args:
            response_data: Response surface data
            operating_points: Nx2 array of operating point coordinates
            model_name: Model identifier
            cdu_id: CDU identifier
            save: Whether to save
            
        Returns:
            Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x1 = response_data.get('x1_centers', np.array([]))
        x2 = response_data.get('x2_centers', np.array([]))
        Z = response_data.get('response_grid', np.array([[]]))
        
        if len(x1) == 0:
            return None
            
        X1, X2 = np.meshgrid(x1, x2)
        
        # Create contour plot
        contour = ax.contourf(X1, X2, Z.T, levels=20, cmap='viridis', alpha=0.7)
        ax.contour(X1, X2, Z.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Overlay operating points
        ax.scatter(operating_points[:, 0], operating_points[:, 1], 
                   c='red', s=20, alpha=0.5, label='Operating Points')
        
        plt.colorbar(contour, ax=ax, label=self.OUTPUT_LABELS.get(
            response_data.get('output_name', 'Output'), 'Output'))
        
        ax.set_xlabel(self.INPUT_LABELS.get(response_data.get('x1_name', 'x1'), 'Input 1'))
        ax.set_ylabel(self.INPUT_LABELS.get(response_data.get('x2_name', 'x2'), 'Input 2'))
        ax.set_title(f"Response Surface with Operating Points\n{model_name} - CDU {cdu_id}")
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            output_name = response_data.get('output_name', 'output')
            filepath = self.output_dir / f"response_with_ops_{model_name}_{output_name}_cdu{cdu_id}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved: {filepath}")
            
        return fig
    
    def close_all(self):
        """Close all matplotlib figures."""
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")
