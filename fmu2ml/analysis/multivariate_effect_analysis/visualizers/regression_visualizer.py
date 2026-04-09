"""
Multiple regression visualizations.
Creates beta coefficient plots, significance heatmaps, and R² charts.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_beta_coefficient_plot(
    regression_df: pd.DataFrame,
    output_dir: str
):
    """
    Create bar chart showing standardized beta coefficients for each output.
    
    Args:
        regression_df: DataFrame with regression results
        output_dir: Directory to save plots
    """
    logger.info("Creating beta coefficient plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract beta columns
    beta_cols = [col for col in regression_df.columns 
                 if col.startswith('beta_') and not col.startswith('beta_std_')]
    
    if not beta_cols:
        logger.warning("No beta coefficients found")
        return
    
    input_names = [col.replace('beta_', '') for col in beta_cols]
    n_outputs = len(regression_df)
    n_inputs = len(input_names)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(max(12, n_outputs * 0.8), 8))
    
    x = np.arange(n_outputs)
    width = 0.8 / n_inputs
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6'][:n_inputs]
    
    for i, (beta_col, input_name) in enumerate(zip(beta_cols, input_names)):
        offset = (i - n_inputs/2 + 0.5) * width
        values = regression_df[beta_col].values
        
        # Check significance
        sig_col = f'is_significant_{input_name}'
        if sig_col in regression_df.columns:
            hatches = ['' if sig else '///' for sig in regression_df[sig_col]]
        else:
            hatches = [''] * len(values)
        
        bars = ax.bar(x + offset, values, width, label=input_name, 
                      color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add hatch for non-significant
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    
    ax.set_xticks(x)
    ax.set_xticklabels(regression_df['output'], rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Output Variables', fontsize=12)
    ax.set_ylabel('Standardized Beta Coefficient', fontsize=12)
    ax.set_title('Standardized Beta Coefficients by Output\n(Hatched = Not Significant at p<0.05)',
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.legend(title='Input Variables', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beta_coefficients.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created beta coefficient plot")


def create_significance_heatmap(
    pvalue_matrix: pd.DataFrame,
    output_dir: str
):
    """
    Create heatmap showing p-values for each input-output pair.
    
    Args:
        pvalue_matrix: DataFrame with p-values (rows=inputs, cols=outputs)
        output_dir: Directory to save plots
    """
    logger.info("Creating significance heatmap...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create custom colormap (white = not significant, darker = more significant)
    # Use -log10(p) for visualization
    log_pvals = -np.log10(pvalue_matrix.clip(lower=1e-10))
    
    # Mask non-significant values
    mask = pvalue_matrix > 0.05
    
    sns.heatmap(
        log_pvals,
        annot=pvalue_matrix.round(3),
        fmt='',
        cmap='YlOrRd',
        mask=mask,
        ax=ax,
        cbar_kws={'label': '-log₁₀(p-value)'},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Add white cells for non-significant
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                          facecolor='white', edgecolor='lightgray'))
                ax.text(j + 0.5, i + 0.5, f'{pvalue_matrix.iloc[i, j]:.3f}',
                       ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Output Variables', fontsize=12)
    ax.set_ylabel('Input Variables', fontsize=12)
    ax.set_title('Predictor Significance Heatmap\n(White = Not Significant, p>0.05)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'significance_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created significance heatmap")


def create_variance_explained_chart(
    variance_df: pd.DataFrame,
    output_dir: str
):
    """
    Create bar chart showing R² for each output.
    
    Args:
        variance_df: DataFrame with R² values for each output
        output_dir: Directory to save plots
    """
    logger.info("Creating variance explained chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by R²
    df_sorted = variance_df.sort_values('r2', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df_sorted) * 0.3)))
    
    # R² bar chart
    ax1 = axes[0]
    
    colors = ['#2ecc71' if r2 > 0.7 else '#f39c12' if r2 > 0.3 else '#e74c3c' 
              for r2 in df_sorted['r2']]
    
    bars = ax1.barh(range(len(df_sorted)), df_sorted['r2'], color=colors,
                   edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['output'], fontsize=10)
    ax1.set_xlabel('R² (Variance Explained)', fontsize=12)
    ax1.set_title('Variance Explained by All Inputs\n(Green>0.7, Yellow>0.3, Red<0.3)',
                 fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value annotations
    for bar, val in zip(bars, df_sorted['r2']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # R² vs Adjusted R² comparison
    ax2 = axes[1]
    
    x = np.arange(len(df_sorted))
    width = 0.35
    
    ax2.barh(x - width/2, df_sorted['r2'], width, label='R²',
            color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.barh(x + width/2, df_sorted['adj_r2'], width, label='Adjusted R²',
            color='#9b59b6', edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(df_sorted['output'], fontsize=10)
    ax2.set_xlabel('R² Value', fontsize=12)
    ax2.set_title('R² vs Adjusted R²\n(Large difference suggests overfitting)',
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Multiple Regression: Variance Explained Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variance_explained.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created variance explained chart")


def create_vif_plot(
    vif_df: pd.DataFrame,
    output_dir: str,
    threshold: float = 10.0
):
    """
    Create bar chart showing VIF values with threshold line.
    
    Args:
        vif_df: DataFrame with VIF values
        output_dir: Directory to save plots
        threshold: VIF threshold for multicollinearity warning
    """
    logger.info("Creating VIF plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if vif_df.empty:
        logger.warning("No VIF data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(vif_df))
    width = 0.35
    
    colors = ['#e74c3c' if vif > threshold else '#2ecc71' 
              for vif in vif_df['max_vif']]
    
    ax.bar(x - width/2, vif_df['mean_vif'], width, label='Mean VIF',
           color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, vif_df['max_vif'], width, label='Max VIF',
           color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold (VIF={threshold})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(vif_df['input'], fontsize=12)
    ax.set_xlabel('Input Variables', fontsize=12)
    ax.set_ylabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_title('Multicollinearity Check\n(VIF > 10 indicates problematic multicollinearity)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vif_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created VIF plot")


def create_regression_summary_plot(
    regression_df: pd.DataFrame,
    output_dir: str
):
    """
    Create summary plot of regression analysis results.
    
    Args:
        regression_df: DataFrame with regression results
        output_dir: Directory to save plot
    """
    logger.info("Creating regression summary plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Check available columns and use appropriate ones
    available_cols = regression_df.columns.tolist()
    
    # 1. R² Distribution
    ax1 = axes[0, 0]
    if 'r2' in available_cols:
        r2_values = regression_df['r2'].values
        ax1.hist(r2_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.mean(r2_values), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(r2_values):.3f}')
        ax1.set_xlabel('R² Score', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of R² Scores', fontsize=14)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'R² data not available', ha='center', va='center', fontsize=12)
        ax1.set_title('Distribution of R² Scores', fontsize=14)
    
    # 2. Model Significance - check for f_pvalue or model_significant
    ax2 = axes[0, 1]
    if 'f_pvalue' in available_cols:
        # Determine significance based on F-test p-value
        significance_threshold = 0.05
        significant = (regression_df['f_pvalue'] < significance_threshold).sum()
        not_significant = len(regression_df) - significant
        
        colors = ['#2ecc71', '#e74c3c']
        ax2.pie([significant, not_significant], 
                labels=[f'Significant\n(p < {significance_threshold})', 'Not Significant'],
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Model Significance (F-test)\n{significant}/{len(regression_df)} significant', fontsize=14)
    elif 'model_significant' in available_cols:
        significant = regression_df['model_significant'].sum()
        not_significant = len(regression_df) - significant
        
        colors = ['#2ecc71', '#e74c3c']
        ax2.pie([significant, not_significant], 
                labels=['Significant', 'Not Significant'],
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Model Significance\n{significant}/{len(regression_df)} significant', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'Significance data not available', ha='center', va='center', fontsize=12)
        ax2.set_title('Model Significance', fontsize=14)
    
    # 3. Adjusted R² vs R²
    ax3 = axes[1, 0]
    if 'r2' in available_cols and 'adj_r2' in available_cols:
        ax3.scatter(regression_df['r2'], regression_df['adj_r2'], 
                   c='steelblue', alpha=0.7, s=100)
        
        # Add diagonal line
        min_val = min(regression_df['r2'].min(), regression_df['adj_r2'].min())
        max_val = max(regression_df['r2'].max(), regression_df['adj_r2'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax3.set_xlabel('R²', fontsize=12)
        ax3.set_ylabel('Adjusted R²', fontsize=12)
        ax3.set_title('R² vs Adjusted R²', fontsize=14)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'R² comparison data not available', ha='center', va='center', fontsize=12)
        ax3.set_title('R² vs Adjusted R²', fontsize=14)
    
    # 4. Top outputs by R²
    ax4 = axes[1, 1]
    if 'r2' in available_cols and 'output' in available_cols:
        top_n = min(10, len(regression_df))
        top_outputs = regression_df.nlargest(top_n, 'r2')
        
        y_pos = np.arange(len(top_outputs))
        ax4.barh(y_pos, top_outputs['r2'].values, color='steelblue', edgecolor='black')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_outputs['output'].values, fontsize=10)
        ax4.set_xlabel('R²', fontsize=12)
        ax4.set_title(f'Top {top_n} Outputs by R²', fontsize=14)
        ax4.invert_yaxis()
    else:
        ax4.text(0.5, 0.5, 'Output R² data not available', ha='center', va='center', fontsize=12)
        ax4.set_title('Top Outputs by R²', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created regression summary plot")
