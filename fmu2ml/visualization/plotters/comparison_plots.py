import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ..utils import (
    VisualizationConfig, get_default_config, setup_plot_style,
    save_figure, get_cdu_column_name
)


class ComparisonPlotter:
    """Plotter for comparing FMU and ML model outputs."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize plotter with config."""
        self.config = config or get_default_config()
        setup_plot_style(self.config.style)
    
    def plot_predictions_vs_targets(self, predictions: np.ndarray,
                                    targets: np.ndarray,
                                    output_names: List[str],
                                    model_name: str = "Model",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot model predictions vs targets with scatter and residuals."""
        n_outputs = min(6, len(output_names))
        
        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize_medium)
        axes = axes.flatten()
        
        for i in range(n_outputs):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10, 
                      color='#1f77b4', edgecolors='none')
            
            # Perfect prediction line
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                   label='Perfect prediction')
            
            # Calculate R² score
            residuals = targets[:, i] - predictions[:, i]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((targets[:, i] - targets[:, i].mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            ax.set_xlabel('True Values', fontsize=10)
            ax.set_ylabel('Predicted Values', fontsize=10)
            ax.set_title(f'{output_names[i]}\nR² = {r2:.4f}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Predictions vs True Values', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=self.config.dpi)
        
        return fig
    
    def plot_model_comparison(self, fmu_data: pd.DataFrame,
                             ml_predictions: Dict[str, pd.DataFrame],
                             compute_blocks: List[int],
                             metrics: List[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Compare FMU outputs with multiple ML model predictions."""
        
        if metrics is None:
            metrics = ['V_flow_prim_GPM', 'W_flow_CDUP_kW', 'T_prim_s_C', 'T_sec_r_C']
        
        n_metrics = len(metrics)
        n_models = len(ml_predictions)
        
        fig, axes = plt.subplots(n_metrics, len(compute_blocks), 
                                figsize=(5 * len(compute_blocks), 4 * n_metrics))
        
        if len(compute_blocks) == 1:
            axes = axes.reshape(-1, 1)
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_models + 1))
        
        for col_idx, cb in enumerate(compute_blocks):
            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                # Get FMU data
                fmu_col = get_cdu_column_name(cb, metric, self.config.system_name)
                if fmu_col in fmu_data.columns:
                    fmu_series = fmu_data[fmu_col]
                    ax.plot(fmu_series.index, fmu_series, 
                           label='FMU', color=colors[0], linewidth=2.5, alpha=0.8)
                    
                    # Plot ML predictions
                    for model_idx, (model_name, pred_df) in enumerate(ml_predictions.items()):
                        if fmu_col in pred_df.columns:
                            ml_series = pred_df[fmu_col]
                            ax.plot(ml_series.index, ml_series,
                                   label=model_name, color=colors[model_idx + 1],
                                   linewidth=2, alpha=0.7, linestyle='--')
                            
                            # Calculate RMSE
                            if len(fmu_series) == len(ml_series):
                                rmse = np.sqrt(np.mean((fmu_series - ml_series)**2))
                                mae = np.mean(np.abs(fmu_series - ml_series))
                                
                                # Add metrics text
                                ax.text(0.02, 0.98 - model_idx * 0.1, 
                                       f'{model_name}: RMSE={rmse:.3f}, MAE={mae:.3f}',
                                       transform=ax.transAxes, 
                                       verticalalignment='top',
                                       fontsize=7,
                                       bbox=dict(boxstyle='round', 
                                               facecolor=colors[model_idx + 1], 
                                               alpha=0.3))
                
                if row_idx == 0:
                    ax.set_title(f'CB{cb}', fontweight='bold', fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(metric, fontsize=10)
                if row_idx == n_metrics - 1:
                    ax.set_xlabel('Time Index', fontsize=10)
                
                ax.legend(loc='best', fontsize=7)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('FMU vs ML Models Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=self.config.dpi)
        
        return fig
    
    def plot_error_distributions(self, fmu_data: pd.DataFrame,
                                 ml_predictions: Dict[str, pd.DataFrame],
                                 compute_blocks: List[int],
                                 metrics: List[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot error distributions for different models."""
        
        if metrics is None:
            metrics = ['V_flow_prim_GPM', 'W_flow_CDUP_kW', 'T_prim_s_C']
        
        n_models = len(ml_predictions)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, n_models, 
                                figsize=(5 * n_models, 4 * n_metrics))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, (model_name, pred_df) in enumerate(ml_predictions.items()):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[metric_idx, model_idx]
                
                errors = []
                for cb in compute_blocks:
                    fmu_col = get_cdu_column_name(cb, metric, self.config.system_name)
                    
                    if fmu_col in fmu_data.columns and fmu_col in pred_df.columns:
                        error = pred_df[fmu_col] - fmu_data[fmu_col]
                        errors.extend(error.dropna().values)
                
                if errors:
                    errors = np.array(errors)
                    
                    # Histogram
                    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', 
                           edgecolor='black', density=True)
                    
                    # Add vertical line at zero
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                             label='Zero error')
                    
                    # Add statistics
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)
                    ax.axvline(x=mean_error, color='green', linestyle=':', 
                             linewidth=2, label=f'Mean: {mean_error:.3f}')
                    
                    ax.text(0.02, 0.98, 
                           f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}\n'
                           f'RMSE: {np.sqrt(np.mean(errors**2)):.3f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=8)
                    
                    if metric_idx == 0:
                        ax.set_title(model_name, fontweight='bold', fontsize=12)
                    if model_idx == 0:
                        ax.set_ylabel(f'{metric}\nDensity', fontsize=10)
                    if metric_idx == n_metrics - 1:
                        ax.set_xlabel('Prediction Error', fontsize=10)
                    
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('Prediction Error Distributions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=self.config.dpi)
        
        return fig
    
    def plot_metrics_heatmap(self, metrics_dict: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap comparing metrics across models."""
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_dict).T
        
        fig, ax = plt.subplots(figsize=self.config.figsize_small)
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   ax=ax, cbar_kws={'label': 'Error Magnitude'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Models', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=self.config.dpi)
        
        return fig
    
    def plot_time_series_comparison(self, fmu_data: pd.DataFrame,
                                   ml_data: pd.DataFrame,
                                   compute_block: int,
                                   metric: str,
                                   model_name: str = "ML Model",
                                   time_range: Optional[Tuple[int, int]] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Detailed time series comparison for a single metric."""
        
        fig, axes = plt.subplots(2, 1, figsize=self.config.figsize_small, 
                                sharex=True, height_ratios=[3, 1])
        
        col_name = get_cdu_column_name(compute_block, metric, self.config.system_name)
        
        if col_name in fmu_data.columns and col_name in ml_data.columns:
            fmu_series = fmu_data[col_name]
            ml_series = ml_data[col_name]
            
            if time_range is not None:
                fmu_series = fmu_series.iloc[time_range[0]:time_range[1]]
                ml_series = ml_series.iloc[time_range[0]:time_range[1]]
            
            # Time series plot
            axes[0].plot(fmu_series.index, fmu_series, label='FMU', 
                        color='blue', linewidth=2, alpha=0.8)
            axes[0].plot(ml_series.index, ml_series, label=model_name, 
                        color='red', linewidth=2, alpha=0.7, linestyle='--')
            axes[0].set_ylabel(metric, fontsize=11)
            axes[0].set_title(f'Time Series Comparison - CB{compute_block}', 
                            fontweight='bold', fontsize=13)
            axes[0].legend(loc='best', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = ml_series - fmu_series
            axes[1].plot(residuals.index, residuals, color='green', 
                        linewidth=1.5, alpha=0.7)
            axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1].fill_between(residuals.index, residuals, 0, 
                                alpha=0.3, color='green')
            
            # Add statistics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            axes[1].text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=9)
            
            axes[1].set_xlabel('Time Index', fontsize=11)
            axes[1].set_ylabel('Residual', fontsize=11)
            axes[1].set_title('Prediction Error', fontweight='bold', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=self.config.dpi)
        
        return fig

def create_comparison_plots(
    fmu_data: pd.DataFrame,
    ml_predictions: Dict[str, pd.DataFrame],
    compute_blocks: List[int],
    output_dir: str,
    metrics: Optional[List[str]] = None,
    config: Optional[VisualizationConfig] = None
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive comparison plots
    
    Parameters:
    -----------
    fmu_data : pd.DataFrame
        FMU simulation outputs
    ml_predictions : Dict[str, pd.DataFrame]
        Dictionary of ML model predictions
    compute_blocks : List[int]
        Compute blocks to compare
    output_dir : str
        Directory to save plots
    metrics : List[str], optional
        Metrics to compare
    config : VisualizationConfig, optional
        Visualization configuration
    
    Returns:
    --------
    Dict[str, plt.Figure] : Dictionary of generated figures
    """
    plotter = ComparisonPlotter(config)
    
    figures = {}
    
    # Model comparison plot
    figures['comparison'] = plotter.plot_model_comparison(
        fmu_data, ml_predictions, compute_blocks, metrics,
        save_path=f"{output_dir}/model_comparison.png"
    )
    
    # Error distributions
    figures['errors'] = plotter.plot_error_distributions(
        fmu_data, ml_predictions, compute_blocks, metrics,
        save_path=f"{output_dir}/error_distributions.png"
    )
    
    # Time series comparisons for each compute block
    if metrics is None:
        metrics = ['V_flow_prim_GPM', 'W_flow_CDUP_kW', 'T_prim_s_C']
    
    for cb in compute_blocks[:3]:  # Limit to first 3 for brevity
        for metric in metrics:
            for model_name, pred_df in ml_predictions.items():
                fig_key = f'ts_{cb}_{metric}_{model_name}'
                figures[fig_key] = plotter.plot_time_series_comparison(
                    fmu_data, pred_df, cb, metric, model_name,
                    save_path=f"{output_dir}/ts_CB{cb}_{metric}_{model_name}.png"
                )
    
    return figures