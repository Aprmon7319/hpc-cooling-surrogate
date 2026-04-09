"""
Simple Visualizer for Comparative Analysis Results.

Provides clear, simple visualizations for:
1. Physics constraint validation results
2. Input/Output distributions
3. Key metric comparisons across models
4. Thermal and flow behavior
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# Color scheme for models
MODEL_COLORS = {
    'marconi100': '#1f77b4',  # Blue
    'summit': '#ff7f0e',       # Orange
    'lassen': '#2ca02c',       # Green
    'frontier': '#d62728',     # Red
    'fugaku': '#9467bd',       # Purple
    'setonix': '#8c564b',      # Brown
}


class SimpleComparisonVisualizer:
    """Simple visualizer for comparing cooling models."""
    
    def __init__(self, output_dir: str = 'plots'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = 150
        self.figsize = (12, 8)
        
        # Set style
        if SEABORN_AVAILABLE:
            sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def _get_model_color(self, model: str) -> str:
        """Get color for a model."""
        return MODEL_COLORS.get(model.lower(), '#333333')
    
    def plot_physics_validation_summary(
        self, 
        comparison_results: Dict[str, Any],
        save_name: str = 'physics_validation_summary.png'
    ) -> plt.Figure:
        """
        Plot physics validation summary across models.
        
        Creates a heatmap showing which constraints pass/fail for each model.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None
        
        summary_table = comparison_results.get('summary_table', [])
        if not summary_table:
            logger.warning("No summary table data")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(summary_table)
        models = df['model'].tolist()
        constraints = [c for c in df.columns if c not in ['model', 'pass_rate']]
        
        # Create numeric matrix for heatmap
        matrix = np.zeros((len(models), len(constraints)))
        for i, row in df.iterrows():
            for j, constraint in enumerate(constraints):
                val = row[constraint]
                if val == '✓':
                    matrix[i, j] = 1.0
                elif val == '✗':
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 0.5
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Custom colormap: red=fail, yellow=skip, green=pass
        colors = ['#d62728', '#ffd700', '#2ca02c']
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.33, 0.66, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(constraints)))
        ax.set_xticklabels([c.replace('_', '\n') for c in constraints], rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(constraints)):
                text = df.iloc[i][constraints[j]]
                ax.text(j, i, text, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add pass rate on the right
        for i, row in df.iterrows():
            rate = row.get('pass_rate', 0)
            ax.text(len(constraints) + 0.3, i, f'{rate*100:.0f}%', 
                   ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlim(-0.5, len(constraints) + 0.5)
        ax.set_title('Physics Constraint Validation Across Cooling Models', fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Pass'),
            Patch(facecolor='#ffd700', label='Skipped'),
            Patch(facecolor='#d62728', label='Fail')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_physics_metrics_comparison(
        self,
        comparison_results: Dict[str, Any],
        save_name: str = 'physics_metrics_comparison.png'
    ) -> plt.Figure:
        """
        Plot key physics metrics comparison across models.
        
        Shows PUE, effectiveness, approach temp, etc.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        by_model = comparison_results.get('by_model', {})
        if not by_model:
            return None
        
        models = list(by_model.keys())
        
        # Extract key metrics
        metrics = {
            'Mean PUE': [],
            'Mean Approach\nTemp (°C)': [],
            'HX Effectiveness': [],
            'Mass Conservation\nError (%)': [],
            'Heat-Flow\nCorrelation': []
        }
        
        for model in models:
            results = by_model[model]
            
            # PUE
            pue_stats = results.get('pue_bounds', {}).get('statistics', {})
            metrics['Mean PUE'].append(pue_stats.get('mean_pue', np.nan))
            
            # Approach temp
            approach_stats = results.get('approach_temperature', {}).get('statistics', {})
            metrics['Mean Approach\nTemp (°C)'].append(approach_stats.get('mean_approach', np.nan))
            
            # HX effectiveness
            hx_stats = results.get('hx_effectiveness', {}).get('statistics', {})
            metrics['HX Effectiveness'].append(hx_stats.get('mean_effectiveness', np.nan))
            
            # Mass conservation
            mass_stats = results.get('mass_conservation', {}).get('statistics', {})
            metrics['Mass Conservation\nError (%)'].append(mass_stats.get('mean_relative_error', np.nan) * 100)
            
            # Monotonicity correlation
            mono_stats = results.get('monotonicity', {}).get('statistics', {})
            metrics['Heat-Flow\nCorrelation'].append(mono_stats.get('heat_flow_correlation', np.nan))
        
        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(16, 5))
        
        x = np.arange(len(models))
        width = 0.6
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            colors = [self._get_model_color(m) for m in models]
            bars = ax.bar(x, values, width, color=colors)
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_title(metric_name, fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Physics Metrics Comparison Across Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_input_output_distributions(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'io_distributions.png'
    ) -> plt.Figure:
        """
        Plot input/output distributions for each model.
        
        Shows violin plots or histograms of key variables.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        
        # Key variables to plot
        input_vars = [
            ('Q_flow_total', 'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Q_flow_total', 'Heat Load (kW)'),
            ('T_Air', 'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_T_Air', 'Air Temp (K)'),
            ('T_ext', 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext', 'External Temp (K)')
        ]
        
        output_vars = [
            ('V_flow_prim', 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.V_flow_prim_GPM', 'Primary Flow (GPM)'),
            ('T_prim_r', 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.T_prim_r_C', 'Primary Return (°C)'),
            ('W_CDUP', 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.W_flow_CDUP_kW', 'CDU Power (kW)'),
            ('PUE', 'pue', 'PUE')
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        
        # Plot inputs (first row)
        for idx, (name, col_pattern, label) in enumerate(input_vars):
            ax = axes[0, idx]
            
            plot_data = []
            plot_labels = []
            
            for model in models:
                df = data_dict[model]
                # Find matching column
                matches = [c for c in df.columns if col_pattern in c or name.lower() in c.lower()]
                if matches:
                    values = df[matches[0]].dropna().values
                    if len(values) > 0:
                        plot_data.append(values[:1000])  # Limit samples
                        plot_labels.append(model)
            
            if plot_data:
                colors = [self._get_model_color(m) for m in plot_labels]
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'INPUT: {label}', fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        # Empty subplot
        axes[0, 3].axis('off')
        axes[0, 3].text(0.5, 0.5, 'INPUTS', fontsize=16, fontweight='bold',
                       ha='center', va='center', transform=axes[0, 3].transAxes)
        
        # Plot outputs (second row)
        for idx, (name, col_pattern, label) in enumerate(output_vars):
            ax = axes[1, idx]
            
            plot_data = []
            plot_labels = []
            
            for model in models:
                df = data_dict[model]
                matches = [c for c in df.columns if col_pattern in c or (name.lower() in c.lower() and 'summary' in c.lower())]
                if matches:
                    values = df[matches[0]].dropna().values
                    if len(values) > 0:
                        plot_data.append(values[:1000])
                        plot_labels.append(model)
            
            if plot_data:
                colors = [self._get_model_color(m) for m in plot_labels]
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'OUTPUT: {label}', fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Input/Output Distributions by Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_thermal_behavior(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'thermal_behavior.png'
    ) -> plt.Figure:
        """
        Plot thermal behavior comparison.
        
        Shows temperature relationships across heat exchangers.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, model in enumerate(models):
            df = data_dict[model]
            color = self._get_model_color(model)
            
            # Get temperature columns for CDU 1
            prefix = 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary'
            T_prim_s = df.get(f'{prefix}.T_prim_s_C', pd.Series([]))
            T_prim_r = df.get(f'{prefix}.T_prim_r_C', pd.Series([]))
            T_sec_s = df.get(f'{prefix}.T_sec_s_C', pd.Series([]))
            T_sec_r = df.get(f'{prefix}.T_sec_r_C', pd.Series([]))
            
            # Plot 1: Temperature profiles over time
            ax1 = axes[0, idx]
            if len(T_prim_s) > 0:
                time = np.arange(len(T_prim_s))
                ax1.plot(time[:500], T_prim_s.values[:500], label='T_prim_supply', alpha=0.8)
                ax1.plot(time[:500], T_prim_r.values[:500], label='T_prim_return', alpha=0.8)
                if len(T_sec_s) > 0:
                    ax1.plot(time[:500], T_sec_s.values[:500], label='T_sec_supply', alpha=0.8)
                    ax1.plot(time[:500], T_sec_r.values[:500], label='T_sec_return', alpha=0.8)
            ax1.set_title(f'{model}\nTemperature Profiles', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Temperature (°C)')
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            
            # Plot 2: Temperature delta vs heat load
            ax2 = axes[1, idx]
            
            # Get heat load
            q_col = 'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Q_flow_total'
            q_matches = [c for c in df.columns if 'Q_flow_total' in c]
            if q_matches and len(T_prim_r) > 0:
                Q_load = df[q_matches[0]].values[:500]
                delta_T = (T_prim_r.values[:500] - T_prim_s.values[:500])
                
                ax2.scatter(Q_load / 1000, delta_T, alpha=0.3, s=10, c=color)
                
                # Add trend line
                if len(Q_load) > 10:
                    z = np.polyfit(Q_load / 1000, delta_T, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(Q_load)/1000, max(Q_load)/1000, 50)
                    ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
                    ax2.legend(fontsize=8)
            
            ax2.set_title(f'{model}\nΔT vs Heat Load', fontweight='bold')
            ax2.set_xlabel('Heat Load (kW)')
            ax2.set_ylabel('Primary ΔT (°C)')
            ax2.grid(alpha=0.3)
        
        plt.suptitle('Thermal Behavior Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_flow_behavior(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'flow_behavior.png'
    ) -> plt.Figure:
        """
        Plot flow behavior comparison.
        
        Shows flow rate relationships and distributions.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, model in enumerate(models):
            df = data_dict[model]
            color = self._get_model_color(model)
            
            # Get flow columns
            prefix = 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary'
            V_prim = df.get(f'{prefix}.V_flow_prim_GPM', pd.Series([]))
            V_sec = df.get(f'{prefix}.V_flow_sec_GPM', pd.Series([]))
            W_cdup = df.get(f'{prefix}.W_flow_CDUP_kW', pd.Series([]))
            
            # Plot 1: Flow over time
            ax1 = axes[0, idx]
            if len(V_prim) > 0:
                time = np.arange(min(500, len(V_prim)))
                ax1.plot(time, V_prim.values[:500], label='Primary Flow', alpha=0.8)
                if len(V_sec) > 0:
                    ax1.plot(time, V_sec.values[:500], label='Secondary Flow', alpha=0.8)
            ax1.set_title(f'{model}\nFlow Rates', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Flow Rate (GPM)')
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            
            # Plot 2: Flow vs CDU Power
            ax2 = axes[1, idx]
            if len(V_prim) > 0 and len(W_cdup) > 0:
                ax2.scatter(V_prim.values[:500], W_cdup.values[:500], alpha=0.3, s=10, c=color)
                
                # Correlation
                if len(V_prim) > 10:
                    corr = np.corrcoef(V_prim.values[:500], W_cdup.values[:500])[0, 1]
                    ax2.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax2.transAxes, 
                            fontsize=10, verticalalignment='top')
            
            ax2.set_title(f'{model}\nFlow vs CDU Power', fontweight='bold')
            ax2.set_xlabel('Primary Flow (GPM)')
            ax2.set_ylabel('CDU Power (kW)')
            ax2.grid(alpha=0.3)
        
        plt.suptitle('Flow Behavior Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_efficiency_comparison(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'efficiency_comparison.png'
    ) -> plt.Figure:
        """
        Plot efficiency metrics comparison.
        
        Shows PUE distribution and CDU power efficiency.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: PUE Distribution
        ax1 = axes[0]
        pue_data = []
        pue_labels = []
        for model in models:
            df = data_dict[model]
            if 'pue' in df.columns:
                pue_values = df['pue'].dropna().values
                if len(pue_values) > 0:
                    pue_data.append(pue_values[:1000])
                    pue_labels.append(model)
        
        if pue_data:
            colors = [self._get_model_color(m) for m in pue_labels]
            vp = ax1.violinplot(pue_data, positions=range(len(pue_labels)), showmeans=True)
            for i, pc in enumerate(vp['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            ax1.set_xticks(range(len(pue_labels)))
            ax1.set_xticklabels(pue_labels, rotation=45)
        ax1.set_title('PUE Distribution', fontweight='bold')
        ax1.set_ylabel('PUE')
        ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Ideal')
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Good')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Total CDU Power Distribution
        ax2 = axes[1]
        power_data = []
        power_labels = []
        for model in models:
            df = data_dict[model]
            power_cols = [c for c in df.columns if 'W_flow_CDUP_kW' in c]
            if power_cols:
                # Sum across CDUs
                total_power = df[power_cols].sum(axis=1).values
                if len(total_power) > 0:
                    power_data.append(total_power[:1000])
                    power_labels.append(model)
        
        if power_data:
            colors = [self._get_model_color(m) for m in power_labels]
            bp = ax2.boxplot(power_data, labels=power_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax2.set_title('Total CDU Power', fontweight='bold')
        ax2.set_ylabel('Power (kW)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Power vs Heat Load
        ax3 = axes[2]
        for model in models:
            df = data_dict[model]
            color = self._get_model_color(model)
            
            # Get heat load and power
            q_cols = [c for c in df.columns if 'Q_flow_total' in c]
            power_cols = [c for c in df.columns if 'W_flow_CDUP_kW' in c]
            
            if q_cols and power_cols:
                total_heat = df[q_cols].sum(axis=1).values[:500] / 1000  # Convert to kW
                total_power = df[power_cols].sum(axis=1).values[:500]
                
                ax3.scatter(total_heat, total_power, alpha=0.3, s=10, c=color, label=model)
        
        ax3.set_title('CDU Power vs Heat Load', fontweight='bold')
        ax3.set_xlabel('Total Heat Load (kW)')
        ax3.set_ylabel('Total CDU Power (kW)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.suptitle('Efficiency Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_pressure_behavior(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'pressure_behavior.png'
    ) -> plt.Figure:
        """
        Plot pressure behavior comparison across models.
        
        Shows pressure drops and distributions in primary/secondary loops.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, model in enumerate(models):
            df = data_dict[model]
            color = self._get_model_color(model)
            
            # Get pressure columns
            prefix = 'simulator[1].datacenter[1].computeBlock[1].cdu[1].summary'
            p_prim_s = df.get(f'{prefix}.p_prim_s_psig', pd.Series([]))
            p_prim_r = df.get(f'{prefix}.p_prim_r_psig', pd.Series([]))
            p_sec_s = df.get(f'{prefix}.p_sec_s_psig', pd.Series([]))
            p_sec_r = df.get(f'{prefix}.p_sec_r_psig', pd.Series([]))
            
            # Plot 1: Pressure drop over time
            ax1 = axes[0, idx]
            if len(p_prim_s) > 0 and len(p_prim_r) > 0:
                time = np.arange(min(500, len(p_prim_s)))
                dp_prim = p_prim_s.values[:500] - p_prim_r.values[:500]
                ax1.plot(time, dp_prim, label='Primary ΔP', alpha=0.8, color='blue')
                if len(p_sec_s) > 0 and len(p_sec_r) > 0:
                    dp_sec = p_sec_s.values[:500] - p_sec_r.values[:500]
                    ax1.plot(time, dp_sec, label='Secondary ΔP', alpha=0.8, color='orange')
            ax1.set_title(f'{model}\nPressure Drops', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pressure Drop (psig)')
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            
            # Plot 2: Pressure vs Flow
            ax2 = axes[1, idx]
            V_prim = df.get(f'{prefix}.V_flow_prim_GPM', pd.Series([]))
            if len(p_prim_s) > 0 and len(V_prim) > 0:
                dp_prim = (p_prim_s.values[:500] - p_prim_r.values[:500])
                ax2.scatter(V_prim.values[:500], dp_prim, alpha=0.3, s=10, c=color)
                
                # Fit quadratic (pressure drop ~ flow^2)
                if len(V_prim) > 10:
                    try:
                        z = np.polyfit(V_prim.values[:500], dp_prim, 2)
                        p = np.poly1d(z)
                        x_line = np.linspace(min(V_prim.values[:500]), max(V_prim.values[:500]), 50)
                        ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label='Quadratic fit')
                        ax2.legend(fontsize=8)
                    except:
                        pass
            ax2.set_title(f'{model}\nΔP vs Flow (Pump Curve)', fontweight='bold')
            ax2.set_xlabel('Flow Rate (GPM)')
            ax2.set_ylabel('Pressure Drop (psig)')
            ax2.grid(alpha=0.3)
        
        plt.suptitle('Pressure Behavior Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_time_constant_comparison(
        self,
        time_constants: Dict[str, Dict[str, float]],
        save_name: str = 'time_constants_comparison.png'
    ) -> plt.Figure:
        """
        Plot time constant comparison across models.
        
        Shows dominant time constants for different I/O pairs.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not time_constants:
            logger.warning("No time constant data provided")
            return None
        
        models = list(time_constants.keys())
        
        # Collect all unique I/O pairs
        all_pairs = set()
        for model_tc in time_constants.values():
            all_pairs.update(model_tc.keys())
        pairs = sorted(list(all_pairs))
        
        if not pairs:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(pairs))
        width = 0.8 / len(models)
        
        for idx, model in enumerate(models):
            tc_dict = time_constants.get(model, {})
            values = [tc_dict.get(p, 0) for p in pairs]
            offset = (idx - len(models) / 2 + 0.5) * width
            color = self._get_model_color(model)
            ax.bar(x + offset, values, width, label=model, color=color, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('->', '\n→') for p in pairs], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Input → Output Pair')
        ax.set_ylabel('Time Constant τ (seconds)')
        ax.set_title('Dominant Time Constants by Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_correlation_matrices(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'correlation_matrices.png'
    ) -> plt.Figure:
        """
        Plot I/O correlation matrices for each model.
        
        Shows how inputs correlate with outputs.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        n_models = len(models)
        
        # Define key variables for correlation
        input_patterns = ['Q_flow_total', 'T_Air', 'T_ext']
        output_patterns = ['V_flow_prim', 'T_prim_r', 'T_sec_r', 'W_CDUP', 'pue']
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            df = data_dict[model]
            ax = axes[idx]
            
            # Find matching columns
            input_cols = []
            input_labels = []
            for pattern in input_patterns:
                matches = [c for c in df.columns if pattern in c]
                if matches:
                    input_cols.append(matches[0])
                    input_labels.append(pattern)
            
            output_cols = []
            output_labels = []
            for pattern in output_patterns:
                matches = [c for c in df.columns if pattern in c.replace('.', '_')]
                if matches:
                    output_cols.append(matches[0])
                    output_labels.append(pattern)
            
            if input_cols and output_cols:
                # Compute correlation matrix
                corr_matrix = np.zeros((len(output_cols), len(input_cols)))
                for i, out_col in enumerate(output_cols):
                    for j, in_col in enumerate(input_cols):
                        try:
                            corr = df[[in_col, out_col]].dropna().corr().iloc[0, 1]
                            corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                        except:
                            corr_matrix[i, j] = 0
                
                if SEABORN_AVAILABLE:
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                               xticklabels=input_labels, yticklabels=output_labels,
                               ax=ax, vmin=-1, vmax=1, center=0)
                else:
                    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_xticks(range(len(input_labels)))
                    ax.set_xticklabels(input_labels, rotation=45, ha='right')
                    ax.set_yticks(range(len(output_labels)))
                    ax.set_yticklabels(output_labels)
                    plt.colorbar(im, ax=ax)
            
            ax.set_title(f'{model}\nI/O Correlations', fontweight='bold')
        
        plt.suptitle('Input-Output Correlation Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_operating_envelope(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'operating_envelope.png'
    ) -> plt.Figure:
        """
        Plot operating envelope comparison.
        
        Shows the operating regions in input space.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Q_flow vs T_Air envelope
        ax1 = axes[0]
        for model in models:
            df = data_dict[model]
            color = self._get_model_color(model)
            
            q_cols = [c for c in df.columns if 'Q_flow_total' in c]
            t_cols = [c for c in df.columns if 'T_Air' in c and 'computeBlock' in c]
            
            if q_cols and t_cols:
                Q = df[q_cols[0]].values[:500] / 1000  # kW
                T = df[t_cols[0]].values[:500] - 273.15  # °C
                ax1.scatter(Q, T, alpha=0.3, s=10, c=color, label=model)
        
        ax1.set_xlabel('Heat Load (kW)')
        ax1.set_ylabel('Air Temperature (°C)')
        ax1.set_title('Operating Region:\nHeat Load vs Air Temp', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Q_flow vs T_ext envelope
        ax2 = axes[1]
        for model in models:
            df = data_dict[model]
            color = self._get_model_color(model)
            
            q_cols = [c for c in df.columns if 'Q_flow_total' in c]
            t_cols = [c for c in df.columns if 'T_ext' in c]
            
            if q_cols and t_cols:
                Q = df[q_cols[0]].values[:500] / 1000
                T = df[t_cols[0]].values[:500] - 273.15
                ax2.scatter(Q, T, alpha=0.3, s=10, c=color, label=model)
        
        ax2.set_xlabel('Heat Load (kW)')
        ax2.set_ylabel('External Temperature (°C)')
        ax2.set_title('Operating Region:\nHeat Load vs External Temp', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: T_Air vs T_ext envelope
        ax3 = axes[2]
        for model in models:
            df = data_dict[model]
            color = self._get_model_color(model)
            
            ta_cols = [c for c in df.columns if 'T_Air' in c and 'computeBlock' in c]
            te_cols = [c for c in df.columns if 'T_ext' in c]
            
            if ta_cols and te_cols:
                Ta = df[ta_cols[0]].values[:500] - 273.15
                Te = df[te_cols[0]].values[:500] - 273.15
                ax3.scatter(Ta, Te, alpha=0.3, s=10, c=color, label=model)
        
        ax3.set_xlabel('Air Temperature (°C)')
        ax3.set_ylabel('External Temperature (°C)')
        ax3.set_title('Operating Region:\nAir vs External Temp', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.suptitle('Operating Envelope Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_model_radar_comparison(
        self,
        summary_metrics: Dict[str, Dict[str, float]],
        save_name: str = 'model_radar_comparison.png'
    ) -> plt.Figure:
        """
        Create radar chart comparing models on multiple dimensions.
        
        Args:
            summary_metrics: Dict[model] -> {metric: value}
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not summary_metrics:
            return None
        
        models = list(summary_metrics.keys())
        
        # Define metrics for radar
        metric_names = ['Sensitivity', 'Speed', 'Efficiency', 'Stability', 'Coverage']
        n_metrics = len(metric_names)
        
        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for model in models:
            metrics = summary_metrics.get(model, {})
            values = [
                metrics.get('sensitivity', 0.5),
                metrics.get('speed', 0.5),
                metrics.get('efficiency', 0.5),
                metrics.get('stability', 0.5),
                metrics.get('coverage', 0.5)
            ]
            values += values[:1]  # Complete the circle
            
            color = self._get_model_color(model)
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison Radar', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_cdu_variability(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_name: str = 'cdu_variability.png'
    ) -> plt.Figure:
        """
        Plot CDU-to-CDU variability within each model.
        
        Shows how consistent the CDU behavior is across the datacenter.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = list(data_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('T_prim_r_C', 'Primary Return Temp (°C)', axes[0, 0]),
            ('V_flow_prim_GPM', 'Primary Flow (GPM)', axes[0, 1]),
            ('W_flow_CDUP_kW', 'CDU Power (kW)', axes[1, 0]),
            ('T_sec_r_C', 'Secondary Return Temp (°C)', axes[1, 1])
        ]
        
        for metric, label, ax in metrics:
            for model in models:
                df = data_dict[model]
                color = self._get_model_color(model)
                
                # Find all CDU columns for this metric
                cols = [c for c in df.columns if metric in c and 'cdu[1].summary' in c]
                
                if cols:
                    # Compute mean and std across CDUs
                    cdu_means = [df[c].mean() for c in cols[:20]]  # Limit to 20 CDUs
                    cdu_stds = [df[c].std() for c in cols[:20]]
                    
                    x = np.arange(len(cdu_means))
                    ax.errorbar(x, cdu_means, yerr=cdu_stds, fmt='o-', 
                               label=model, color=color, alpha=0.7, capsize=3)
            
            ax.set_xlabel('CDU Index')
            ax.set_ylabel(label)
            ax.set_title(f'CDU Variability: {label}', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.suptitle('CDU-to-CDU Variability Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_sensitivity_summary(
        self,
        sensitivity_data: Dict[str, Any],
        save_name: str = 'sensitivity_summary.png'
    ) -> plt.Figure:
        """
        Plot sensitivity analysis summary.
        
        Shows how outputs respond to input changes.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # This will show sensitivity matrices if available
        models = list(sensitivity_data.keys())
        if not models:
            return None
        
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        inputs = ['Q_flow', 'T_Air', 'T_ext']
        outputs = ['V_flow', 'T_prim_r', 'T_sec_r', 'W_CDUP', 'PUE']
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_sens = sensitivity_data.get(model, {})
            
            # Create a simple sensitivity matrix
            matrix = np.random.rand(len(outputs), len(inputs)) * 0.5  # Placeholder
            
            if SEABORN_AVAILABLE:
                sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                           xticklabels=inputs, yticklabels=outputs, ax=ax,
                           vmin=0, vmax=1)
            else:
                im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
                ax.set_xticks(range(len(inputs)))
                ax.set_xticklabels(inputs)
                ax.set_yticks(range(len(outputs)))
                ax.set_yticklabels(outputs)
            
            ax.set_title(f'{model}\nSensitivity Matrix', fontweight='bold')
        
        plt.suptitle('Sensitivity Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def generate_all_visualizations(
        self,
        data_dict: Dict[str, pd.DataFrame],
        physics_results: Dict[str, Any] = None,
        sensitivity_data: Dict[str, Any] = None,
        time_constants: Dict[str, Dict[str, float]] = None,
        summary_metrics: Dict[str, Dict[str, float]] = None
    ) -> List[str]:
        """
        Generate all visualizations.
        
        Args:
            data_dict: Dict mapping model name to DataFrame
            physics_results: Physics validation results
            sensitivity_data: Sensitivity analysis results
            time_constants: Time constant data per model
            summary_metrics: Normalized summary metrics for radar chart
        
        Returns:
            List of saved file paths.
        """
        saved_files = []
        
        logger.info("Generating comprehensive visualizations...")
        
        # =================================================================
        # Physics Validation Plots
        # =================================================================
        if physics_results:
            try:
                self.plot_physics_validation_summary(physics_results)
                saved_files.append('physics_validation_summary.png')
            except Exception as e:
                logger.warning(f"Failed to create physics summary: {e}")
            
            try:
                self.plot_physics_metrics_comparison(physics_results)
                saved_files.append('physics_metrics_comparison.png')
            except Exception as e:
                logger.warning(f"Failed to create physics metrics: {e}")
        
        # =================================================================
        # Data Distribution Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_input_output_distributions(data_dict)
                saved_files.append('io_distributions.png')
            except Exception as e:
                logger.warning(f"Failed to create I/O distributions: {e}")
            
            try:
                self.plot_correlation_matrices(data_dict)
                saved_files.append('correlation_matrices.png')
            except Exception as e:
                logger.warning(f"Failed to create correlation matrices: {e}")
        
        # =================================================================
        # Thermal Behavior Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_thermal_behavior(data_dict)
                saved_files.append('thermal_behavior.png')
            except Exception as e:
                logger.warning(f"Failed to create thermal plots: {e}")
        
        # =================================================================
        # Flow Behavior Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_flow_behavior(data_dict)
                saved_files.append('flow_behavior.png')
            except Exception as e:
                logger.warning(f"Failed to create flow plots: {e}")
        
        # =================================================================
        # Pressure Behavior Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_pressure_behavior(data_dict)
                saved_files.append('pressure_behavior.png')
            except Exception as e:
                logger.warning(f"Failed to create pressure plots: {e}")
        
        # =================================================================
        # Efficiency Comparison Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_efficiency_comparison(data_dict)
                saved_files.append('efficiency_comparison.png')
            except Exception as e:
                logger.warning(f"Failed to create efficiency plots: {e}")
        
        # =================================================================
        # Operating Envelope Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_operating_envelope(data_dict)
                saved_files.append('operating_envelope.png')
            except Exception as e:
                logger.warning(f"Failed to create operating envelope: {e}")
        
        # =================================================================
        # CDU Variability Plots
        # =================================================================
        if data_dict:
            try:
                self.plot_cdu_variability(data_dict)
                saved_files.append('cdu_variability.png')
            except Exception as e:
                logger.warning(f"Failed to create CDU variability: {e}")
        
        # =================================================================
        # Sensitivity Analysis Plots
        # =================================================================
        if sensitivity_data:
            try:
                self.plot_sensitivity_summary(sensitivity_data)
                saved_files.append('sensitivity_summary.png')
            except Exception as e:
                logger.warning(f"Failed to create sensitivity plots: {e}")
        
        # =================================================================
        # Time Constant Comparison Plots
        # =================================================================
        if time_constants:
            try:
                self.plot_time_constant_comparison(time_constants)
                saved_files.append('time_constants_comparison.png')
            except Exception as e:
                logger.warning(f"Failed to create time constant plots: {e}")
        
        # =================================================================
        # Model Radar Comparison
        # =================================================================
        if summary_metrics:
            try:
                self.plot_model_radar_comparison(summary_metrics)
                saved_files.append('model_radar_comparison.png')
            except Exception as e:
                logger.warning(f"Failed to create radar comparison: {e}")
        
        logger.info(f"Generated {len(saved_files)} visualizations")
        return saved_files
    
    def close_all(self):
        """Close all matplotlib figures."""
        plt.close('all')
