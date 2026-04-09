"""
Model comparison visualizations.
Creates side-by-side comparisons of linear vs. non-linear model performance.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_model_comparison_plots(
    comparison_df: pd.DataFrame,
    polynomial_df: pd.DataFrame,
    output_dir: str
):
    """
    Create comprehensive model comparison visualizations.
    
    Args:
        comparison_df: DataFrame with linear vs. nonlinear comparison
        polynomial_df: DataFrame with polynomial fit results
        output_dir: Directory to save plots
    """
    logger.info("Creating model comparison plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. R² Comparison Heatmap (Linear vs Best Nonlinear)
    _create_r2_comparison_heatmap(comparison_df, output_dir)
    
    # 2. R² by Polynomial Degree Heatmap
    _create_degree_r2_heatmap(polynomial_df, output_dir)
    
    # 3. AIC/BIC Model Selection Plot
    _create_aic_bic_plot(polynomial_df, output_dir)
    
    # 4. Best Degree Distribution
    _create_best_degree_distribution(comparison_df, output_dir)
    
    logger.info("Created model comparison plots")


def _create_r2_comparison_heatmap(comparison_df: pd.DataFrame, output_dir: str):
    """Create heatmap comparing linear and best nonlinear R²."""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Linear R²
    pivot_linear = comparison_df.pivot(
        index='input',
        columns='output',
        values='linear_r2'
    )
    
    sns.heatmap(
        pivot_linear,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=axes[0],
        cbar_kws={'label': 'R²'}
    )
    axes[0].set_title('Linear Model R²', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Output Variables', fontsize=10)
    axes[0].set_ylabel('Input Variables', fontsize=10)
    
    # Best Nonlinear R²
    pivot_nonlinear = comparison_df.pivot(
        index='input',
        columns='output',
        values='best_nonlinear_r2'
    )
    
    sns.heatmap(
        pivot_nonlinear,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_kws={'label': 'R²'}
    )
    axes[1].set_title('Best Nonlinear Model R²', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Output Variables', fontsize=10)
    axes[1].set_ylabel('Input Variables', fontsize=10)
    
    # R² Improvement
    pivot_improvement = comparison_df.pivot(
        index='input',
        columns='output',
        values='r2_improvement_pct'
    )
    
    sns.heatmap(
        pivot_improvement,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        ax=axes[2],
        cbar_kws={'label': '% Improvement'}
    )
    axes[2].set_title('R² Improvement (%)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Output Variables', fontsize=10)
    axes[2].set_ylabel('Input Variables', fontsize=10)
    
    plt.suptitle('Linear vs. Nonlinear Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'r2_comparison_heatmap.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def _create_degree_r2_heatmap(polynomial_df: pd.DataFrame, output_dir: str):
    """Create heatmap showing R² for different polynomial degrees."""
    
    # Find R² columns
    r2_cols = [col for col in polynomial_df.columns if col.startswith('r2_degree_')]
    
    if not r2_cols:
        logger.warning("No R² columns found in polynomial_df")
        return
    
    # Create subplot for each input
    inputs = polynomial_df['input'].unique()
    n_inputs = len(inputs)
    
    fig, axes = plt.subplots(1, n_inputs, figsize=(7 * n_inputs, 8))
    
    if n_inputs == 1:
        axes = [axes]
    
    for idx, input_name in enumerate(inputs):
        ax = axes[idx]
        
        input_data = polynomial_df[polynomial_df['input'] == input_name]
        
        # Create matrix: outputs x degrees
        outputs = input_data['output'].unique()
        degrees = sorted([int(col.split('_')[-1]) for col in r2_cols])
        
        matrix = np.zeros((len(outputs), len(degrees)))
        
        for i, output in enumerate(outputs):
            row = input_data[input_data['output'] == output].iloc[0]
            for j, deg in enumerate(degrees):
                col_name = f'r2_degree_{deg}'
                if col_name in row.index:
                    matrix[i, j] = row[col_name]
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            xticklabels=[f'Deg {d}' for d in degrees],
            yticklabels=outputs,
            cbar_kws={'label': 'R²'}
        )
        
        ax.set_title(f'Input: {input_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Polynomial Degree', fontsize=10)
        ax.set_ylabel('Output Variables', fontsize=10)
    
    plt.suptitle('R² by Polynomial Degree', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'degree_r2_heatmap.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def _create_aic_bic_plot(polynomial_df: pd.DataFrame, output_dir: str):
    """Create AIC/BIC comparison plot for model selection."""
    
    aic_cols = [col for col in polynomial_df.columns if col.startswith('aic_degree_')]
    bic_cols = [col for col in polynomial_df.columns if col.startswith('bic_degree_')]
    
    if not aic_cols:
        logger.warning("No AIC columns found")
        return
    
    # Create a summary plot
    inputs = polynomial_df['input'].unique()
    n_inputs = len(inputs)
    
    fig, axes = plt.subplots(n_inputs, 2, figsize=(14, 5 * n_inputs))
    
    if n_inputs == 1:
        axes = axes.reshape(1, -1)
    
    degrees = sorted([int(col.split('_')[-1]) for col in aic_cols])
    
    for idx, input_name in enumerate(inputs):
        input_data = polynomial_df[polynomial_df['input'] == input_name]
        
        outputs = input_data['output'].values
        
        # AIC subplot
        ax_aic = axes[idx, 0]
        for i, output in enumerate(outputs):
            row = input_data[input_data['output'] == output].iloc[0]
            aic_values = [row[f'aic_degree_{d}'] for d in degrees if f'aic_degree_{d}' in row.index]
            ax_aic.plot(degrees[:len(aic_values)], aic_values, 'o-', label=output, alpha=0.7)
        
        ax_aic.set_xlabel('Polynomial Degree', fontsize=10)
        ax_aic.set_ylabel('AIC', fontsize=10)
        ax_aic.set_title(f'{input_name}: AIC by Degree', fontsize=11)
        ax_aic.grid(True, alpha=0.3)
        ax_aic.legend(fontsize=8, loc='best', ncol=2)
        
        # BIC subplot
        ax_bic = axes[idx, 1]
        for i, output in enumerate(outputs):
            row = input_data[input_data['output'] == output].iloc[0]
            bic_values = [row[f'bic_degree_{d}'] for d in degrees if f'bic_degree_{d}' in row.index]
            ax_bic.plot(degrees[:len(bic_values)], bic_values, 'o-', label=output, alpha=0.7)
        
        ax_bic.set_xlabel('Polynomial Degree', fontsize=10)
        ax_bic.set_ylabel('BIC', fontsize=10)
        ax_bic.set_title(f'{input_name}: BIC by Degree', fontsize=11)
        ax_bic.grid(True, alpha=0.3)
        ax_bic.legend(fontsize=8, loc='best', ncol=2)
    
    plt.suptitle('Model Selection Criteria (AIC/BIC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'aic_bic_comparison.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def _create_best_degree_distribution(comparison_df: pd.DataFrame, output_dir: str):
    """Create bar chart showing distribution of best polynomial degrees."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count of best degrees overall
    ax1 = axes[0]
    degree_counts = comparison_df['best_degree'].value_counts().sort_index()
    
    bars = ax1.bar(degree_counts.index, degree_counts.values, color='steelblue', edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('Number of Pairs', fontsize=12)
    ax1.set_title('Distribution of Optimal Polynomial Degrees', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Best degree by input variable
    ax2 = axes[1]
    inputs = comparison_df['input'].unique()
    x = np.arange(max(comparison_df['best_degree']) + 1)
    width = 0.25
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, input_name in enumerate(inputs):
        input_data = comparison_df[comparison_df['input'] == input_name]
        counts = input_data['best_degree'].value_counts()
        
        heights = [counts.get(deg, 0) for deg in x]
        ax2.bar(x + i * width, heights, width, label=input_name, color=colors[i], edgecolor='black')
    
    ax2.set_xlabel('Polynomial Degree', fontsize=12)
    ax2.set_ylabel('Number of Outputs', fontsize=12)
    ax2.set_title('Optimal Degree by Input Variable', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(d) for d in x])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Polynomial Degree Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'best_degree_distribution.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
