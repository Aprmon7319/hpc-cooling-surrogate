"""
Response Characteristics Heatmap Visualizer

Creates heatmaps showing:
- Rise time for each input-output pair
- Gain for each input-output pair
- Time constants
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

logger = logging.getLogger(__name__)


def create_response_characteristics_heatmap(
    response_summary_df: pd.DataFrame,
    output_dir: str
):
    """
    Create heatmaps showing response characteristics.
    
    Args:
        response_summary_df: Summary of response characteristics
        output_dir: Directory to save plots
    """
    logger.info("Creating response characteristics heatmaps...")
    
    if response_summary_df.empty:
        logger.warning("No response summary data for heatmaps")
        return
    
    metrics = [
        ('rise_time_mean', 'Mean Rise Time (steps)', 'YlOrRd'),
        ('gain_mean', 'Mean Gain (Δout/Δin)', 'RdYlGn'),
        ('time_constant_mean', 'Mean Time Constant (τ)', 'YlGnBu'),
        ('settling_time_mean', 'Mean Settling Time (steps)', 'YlOrBr')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[idx]
        
        if metric not in response_summary_df.columns:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue
        
        try:
            pivot = response_summary_df.pivot(
                index='input',
                columns='output',
                values=metric
            )
            
            # Handle NaN values
            pivot = pivot.fillna(0)
            
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                ax=ax,
                cbar_kws={'label': title}
            )
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Output Variables', fontsize=10)
            ax.set_ylabel('Input Variables', fontsize=10)
            
        except Exception as e:
            logger.error(f"Error creating heatmap for {metric}: {e}")
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                   transform=ax.transAxes)
    
    plt.suptitle('Response Characteristics Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_characteristics_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created response characteristics heatmaps")


def create_dynamics_summary_heatmap(
    dynamics_summary_df: pd.DataFrame,
    output_dir: str
):
    """
    Create heatmap showing dynamics summary combining level/rate and impulse analysis.
    
    Args:
        dynamics_summary_df: Combined dynamics summary
        output_dir: Directory to save plots
    """
    logger.info("Creating dynamics summary heatmap...")
    
    if dynamics_summary_df.empty:
        logger.warning("No dynamics summary data for heatmap")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Dominant effect heatmap
    ax1 = axes[0]
    effect_map = {'level': 0, 'rate': 1, 'mixed': 0.5}
    df_temp = dynamics_summary_df.copy()
    df_temp['effect_numeric'] = df_temp['dominant_effect'].map(effect_map)
    
    try:
        pivot1 = df_temp.pivot(index='input', columns='output', values='effect_numeric')
        
        cmap = plt.cm.colors.ListedColormap(['#3498db', '#f39c12', '#e74c3c'])
        bounds = [0, 0.25, 0.75, 1]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        sns.heatmap(pivot1, ax=ax1, cmap='RdYlBu_r', center=0.5,
                   annot=True, fmt='.2f', cbar_kws={'label': 'Level (0) ↔ Rate (1)'})
        ax1.set_title('Dominant Effect Type', fontsize=12)
        ax1.set_xlabel('Output', fontsize=10)
        ax1.set_ylabel('Input', fontsize=10)
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Optimal lag heatmap
    ax2 = axes[1]
    if 'optimal_lag' in dynamics_summary_df.columns:
        try:
            pivot2 = dynamics_summary_df.pivot(
                index='input', columns='output', values='optimal_lag'
            )
            
            sns.heatmap(pivot2, ax=ax2, cmap='coolwarm', center=0,
                       annot=True, fmt='.0f', cbar_kws={'label': 'Optimal Lag (steps)'})
            ax2.set_title('Optimal Response Lag', fontsize=12)
            ax2.set_xlabel('Output', fontsize=10)
            ax2.set_ylabel('Input', fontsize=10)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No lag data', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Effect strength ratio heatmap
    ax3 = axes[2]
    if 'effect_ratio' in dynamics_summary_df.columns:
        try:
            df_temp2 = dynamics_summary_df.copy()
            df_temp2['effect_ratio_capped'] = df_temp2['effect_ratio'].clip(upper=3)
            
            pivot3 = df_temp2.pivot(
                index='input', columns='output', values='effect_ratio_capped'
            )
            
            sns.heatmap(pivot3, ax=ax3, cmap='RdYlGn', center=1,
                       annot=True, fmt='.2f', cbar_kws={'label': 'Rate/Level Ratio'})
            ax3.set_title('Effect Strength Ratio (Rate/Level)', fontsize=12)
            ax3.set_xlabel('Output', fontsize=10)
            ax3.set_ylabel('Input', fontsize=10)
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No ratio data', ha='center', va='center', transform=ax3.transAxes)
    
    plt.suptitle('Dynamic Effects Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamics_summary_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created dynamics summary heatmap")
