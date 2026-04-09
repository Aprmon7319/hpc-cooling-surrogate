"""
Non-linearity strength bar chart.
Shows percentage-point improvement in R² from non-linear models.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_nonlinearity_strength_chart(
    comparison_df: pd.DataFrame,
    output_dir: str
):
    """
    Create bar chart showing non-linearity strength across all pairs.
    
    Non-linearity strength = percentage-point improvement in R² from linear to best polynomial model.
    R² values are clamped to [0, 1] to avoid misleading results from negative R².
    
    Args:
        comparison_df: DataFrame with model comparison results
        output_dir: Directory to save plots
    """
    logger.info("Creating non-linearity strength chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate that r2_improvement_pct values are reasonable
    max_improvement = comparison_df['r2_improvement_pct'].max()
    if max_improvement > 100:
        logger.warning(
            f"Maximum R² improvement is {max_improvement:.1f} pp, which exceeds 100 pp. "
            f"This suggests R² values were not properly clamped. "
            f"Values will be capped at 100 pp for visualization."
        )
        # Cap at 100 percentage points (maximum possible for R² in [0, 1])
        comparison_df = comparison_df.copy()
        comparison_df['r2_improvement_pct'] = comparison_df['r2_improvement_pct'].clip(upper=100.0)
    
    # 1. Overall strength bar chart
    _create_overall_strength_chart(comparison_df, output_dir)
    
    # 2. Strength heatmap
    _create_strength_heatmap(comparison_df, output_dir)
    
    # 3. Strength by input
    _create_strength_by_input_chart(comparison_df, output_dir)
    
    # 4. Debug chart with raw R² values if available
    if 'linear_r2_raw' in comparison_df.columns:
        _create_raw_r2_debug_chart(comparison_df, output_dir)
    
    logger.info("Created non-linearity strength charts")


def _create_overall_strength_chart(comparison_df: pd.DataFrame, output_dir: str):
    """Create bar chart of non-linearity strength for each pair."""
    
    # Sort by non-linearity strength
    df_sorted = comparison_df.sort_values('r2_improvement_pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.3)))
    
    # Create labels
    labels = [f"{row['input']} → {row['output']}" for _, row in df_sorted.iterrows()]
    values = df_sorted['r2_improvement_pct'].values
    
    # Color bars by significance (percentage points)
    colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#2ecc71' for v in values]
    
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('R² Improvement (percentage points)', fontsize=12)
    ax.set_title('Non-linearity Strength by Input-Output Pair', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, label='Moderate (5 pp)')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='Strong (10 pp)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right', fontsize=10)
    
    # Add value annotations
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f} pp', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nonlinearity_strength_bars.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def _create_strength_heatmap(comparison_df: pd.DataFrame, output_dir: str):
    """Create heatmap of non-linearity strength."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² improvement percentage points
    ax1 = axes[0]
    pivot_pct = comparison_df.pivot(
        index='input',
        columns='output',
        values='r2_improvement_pct'
    )
    
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',  # Reversed so red = high non-linearity
        center=5,
        ax=ax1,
        cbar_kws={'label': 'Improvement (pp)'}
    )
    ax1.set_title('Non-linearity Strength (R² Improvement, pp)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Output Variables', fontsize=10)
    ax1.set_ylabel('Input Variables', fontsize=10)
    
    # Best polynomial degree
    ax2 = axes[1]
    pivot_degree = comparison_df.pivot(
        index='input',
        columns='output',
        values='best_degree'
    )
    
    sns.heatmap(
        pivot_degree,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        vmin=1,
        vmax=5,
        ax=ax2,
        cbar_kws={'label': 'Polynomial Degree'}
    )
    ax2.set_title('Recommended Polynomial Degree', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Output Variables', fontsize=10)
    ax2.set_ylabel('Input Variables', fontsize=10)
    
    plt.suptitle('Non-linearity Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nonlinearity_strength_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def _create_strength_by_input_chart(comparison_df: pd.DataFrame, output_dir: str):
    """Create grouped bar chart showing strength by input variable."""
    
    inputs = comparison_df['input'].unique()
    n_inputs = len(inputs)
    
    fig, axes = plt.subplots(1, n_inputs, figsize=(6 * n_inputs, 6))
    
    if n_inputs == 1:
        axes = [axes]
    
    for idx, input_name in enumerate(inputs):
        ax = axes[idx]
        
        input_data = comparison_df[comparison_df['input'] == input_name].copy()
        input_data = input_data.sort_values('r2_improvement_pct', ascending=True)
        
        outputs = input_data['output'].values
        improvements = input_data['r2_improvement_pct'].values
        
        # Color based on significance
        bar_colors = [
            '#e74c3c' if imp > 10 else '#f39c12' if imp > 5 else '#2ecc71' 
            for imp in improvements
        ]
        
        bars = ax.barh(range(len(outputs)), improvements, color=bar_colors, edgecolor='black')
        
        ax.set_yticks(range(len(outputs)))
        ax.set_yticklabels(outputs, fontsize=9)
        ax.set_xlabel('R² Improvement (pp)', fontsize=11)
        ax.set_title(f'Input: {input_name}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Moderate (5 pp)')
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Strong (10 pp)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value annotations
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{imp:.1f}%', va='center', fontsize=8,
                   fontweight='bold' if imp > 5 else 'normal')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Linear sufficient (<5 pp)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Moderate non-linearity (5-10 pp)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Strong non-linearity (>10 pp)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10, 
              bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Non-linearity Strength by Input Variable', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, 'nonlinearity_strength_by_input.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def _create_raw_r2_debug_chart(comparison_df: pd.DataFrame, output_dir: str):
    """Create debug chart showing raw R² values to diagnose issues."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(comparison_df) * 0.25)))
    
    df_sorted = comparison_df.sort_values('linear_r2_raw', ascending=True)
    labels = [f"{row['input']} → {row['output']}" for _, row in df_sorted.iterrows()]
    
    # Raw Linear R²
    ax1 = axes[0]
    linear_vals = df_sorted['linear_r2_raw'].values
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in linear_vals]
    ax1.barh(range(len(labels)), linear_vals, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Raw Linear R²', fontsize=11)
    ax1.set_title('Raw Linear R² (before clamping)', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=2)
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, val in enumerate(linear_vals):
        ax1.text(max(val, 0) + 0.01, i, f'{val:.4f}', va='center', fontsize=7)
    
    # Raw Best Nonlinear R²
    ax2 = axes[1]
    nonlinear_vals = df_sorted['best_nonlinear_r2_raw'].values
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in nonlinear_vals]
    ax2.barh(range(len(labels)), nonlinear_vals, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Raw Best Nonlinear R²', fontsize=11)
    ax2.set_title('Raw Best Nonlinear R² (before clamping)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, val in enumerate(nonlinear_vals):
        ax2.text(max(val, 0) + 0.01, i, f'{val:.4f}', va='center', fontsize=7)
    
    plt.suptitle('Debug: Raw R² Values (Negative = worse than mean prediction)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debug_raw_r2_values.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log summary
    n_negative_linear = (comparison_df['linear_r2_raw'] < 0).sum()
    n_negative_nonlinear = (comparison_df['best_nonlinear_r2_raw'] < 0).sum()
    logger.info(f"Debug R² summary: {n_negative_linear} pairs with negative linear R², "
               f"{n_negative_nonlinear} pairs with negative best nonlinear R²")


def create_neural_network_recommendations(
    comparison_df: pd.DataFrame,
    output_dir: str
):
    """
    Create recommendations for neural network architecture based on non-linearity analysis.
    """
    logger.info("Creating neural network recommendations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze non-linearity patterns
    strongly_nonlinear = comparison_df[comparison_df['r2_improvement_pct'] > 5]
    linear_sufficient = comparison_df[comparison_df['r2_improvement_pct'] <= 5]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Summary statistics
    ax1 = axes[0]
    
    categories = ['Linear Sufficient\n(<5 pp improvement)', 
                 'Moderate Non-linear\n(5-10 pp)',
                 'Strong Non-linear\n(>10 pp)']
    counts = [
        len(linear_sufficient),
        len(comparison_df[(comparison_df['r2_improvement_pct'] > 5) & 
                         (comparison_df['r2_improvement_pct'] <= 10)]),
        len(comparison_df[comparison_df['r2_improvement_pct'] > 10])
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Filter out zero-count categories for pie chart
    non_zero = [(c, cnt, col) for c, cnt, col in zip(categories, counts, colors) if cnt > 0]
    if non_zero:
        cats, cnts, cols = zip(*non_zero)
        wedges, texts, autotexts = ax1.pie(cnts, labels=cats, colors=cols,
                                           autopct='%1.1f%%', startangle=90)
    ax1.set_title('Non-linearity Classification', fontsize=12, fontweight='bold')
    
    # Right: Recommendations text
    ax2 = axes[1]
    ax2.axis('off')
    
    # Calculate statistics
    mean_improvement = comparison_df['r2_improvement_pct'].mean()
    max_improvement = comparison_df['r2_improvement_pct'].max()
    common_degree = comparison_df['best_degree'].mode().iloc[0] if len(comparison_df) > 0 else 1
    
    recommendations = f"""
NEURAL NETWORK ARCHITECTURE RECOMMENDATIONS
============================================

Based on non-linearity analysis:

1. OVERALL NON-LINEARITY: {'HIGH' if mean_improvement > 5 else 'MODERATE' if mean_improvement > 2 else 'LOW'}
   - Average R² improvement: {mean_improvement:.1f} pp
   - Maximum improvement: {max_improvement:.1f} pp
   
2. MOST COMMON OPTIMAL DEGREE: {common_degree}
   - Suggests {'non-linear' if common_degree > 1 else 'linear'} activations needed

3. RECOMMENDATIONS:
   
   {'• Use non-linear activation functions (ReLU, GELU, Swish)' if mean_improvement > 5 else '• Linear activations may suffice for most relationships'}
   
   • Consider {'polynomial feature expansion' if common_degree > 2 else 'standard dense layers'}
   
   • Suggested network depth: {'Deep (4+ layers)' if max_improvement > 10 else 'Moderate (2-3 layers)'}
   
4. KEY NON-LINEAR RELATIONSHIPS:
"""
    
    # Add top non-linear pairs
    top_nonlinear = comparison_df.nlargest(3, 'r2_improvement_pct')
    for _, row in top_nonlinear.iterrows():
        recommendations += f"\n   • {row['input']} → {row['output']}: {row['r2_improvement_pct']:.1f} pp improvement (degree {row['best_degree']})"
    
    ax2.text(0.05, 0.95, recommendations, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Neural Operator Design Recommendations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neural_network_recommendations.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as text file
    with open(os.path.join(output_dir, 'recommendations.txt'), 'w') as f:
        f.write(recommendations)
    
    logger.info("Created neural network recommendations")