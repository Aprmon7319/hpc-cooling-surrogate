import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def create_importance_ranking_plots(rankings_df: pd.DataFrame, output_dir: str):
    """Create string/line plots showing input importance across all outputs."""
    logger.info("Creating importance ranking plots...")
    
    # Get unique inputs and outputs
    input_vars = rankings_df['input'].unique()
    output_vars = rankings_df['output'].unique()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color map for inputs
    colors = {'Q_flow': '#e74c3c', 'T_Air': '#3498db', 'T_ext': '#2ecc71'}
    markers = {'Q_flow': 'o', 'T_Air': 's', 'T_ext': '^'}
    
    # Subplot 1: String plot with lines connecting rankings
    for input_var in input_vars:
        input_data = rankings_df[rankings_df['input'] == input_var].sort_values('output')
        
        # Get values and output positions
        values = input_data['value'].values
        output_positions = np.arange(len(output_vars))
        
        # Plot line
        ax1.plot(output_positions, values, 
                marker=markers[input_var], 
                color=colors[input_var],
                linewidth=2.5, 
                markersize=10,
                label=input_var,
                alpha=0.8)
        
        # Add value labels at each point
        for x, y in zip(output_positions, values):
            ax1.text(x, y + 0.02, f'{y:.3f}', 
                    fontsize=8, 
                    ha='center', 
                    va='bottom',
                    color=colors[input_var],
                    fontweight='bold')
    
    ax1.set_xticks(output_positions)
    ax1.set_xticklabels(output_vars, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('Mutual Information', fontsize=13, fontweight='bold')
    ax1.set_title('Input Importance Across All Outputs (String Plot)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Subplot 2: Stacked bar chart showing relative importance
    # Pivot data for stacking
    pivot_data = rankings_df.pivot(index='output', columns='input', values='value')
    
    # Normalize to show relative importance (percentage)
    pivot_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(output_vars))
    width = 0.6
    
    bottom = np.zeros(len(output_vars))
    
    for input_var in input_vars:
        values = pivot_normalized[input_var].values
        ax2.bar(x, values, width, 
               label=input_var, 
               bottom=bottom,
               color=colors[input_var],
               alpha=0.8)
        
        # Add percentage labels
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:  # Only show label if segment is large enough
                ax2.text(i, bot + val/2, f'{val:.1f}%',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white')
        
        bottom += values
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(output_vars, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Relative Importance (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Relative Input Importance Distribution', 
                 fontsize=15, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'importance_rankings.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    logger.info(f"Saved importance ranking plots to {output_dir}")