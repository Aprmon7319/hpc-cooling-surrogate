"""
Partial correlation visualizations.
Creates comparison bar charts and network diagrams for direct effects.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from typing import Dict, List
import multiprocessing as mp

logger = logging.getLogger(__name__)


def create_correlation_comparison_chart(
    partial_df: pd.DataFrame,
    output_dir: str
):
    """
    Create bar chart comparing Pearson vs. Partial correlation.
    
    Args:
        partial_df: DataFrame with partial correlation results
        output_dir: Directory to save plots
    """
    logger.info("Creating correlation comparison chart...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by absolute partial correlation
    df_sorted = partial_df.copy()
    df_sorted['abs_partial'] = df_sorted['partial_corr'].abs()
    df_sorted = df_sorted.sort_values('abs_partial', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_sorted) * 0.35)))
    
    # Create labels
    labels = [f"{row['input']} → {row['output']}" for _, row in df_sorted.iterrows()]
    y_pos = np.arange(len(labels))
    
    # Bar width
    bar_height = 0.35
    
    # Pearson correlations
    pearson_vals = df_sorted['pearson_corr'].values
    partial_vals = df_sorted['partial_corr'].values
    
    # Colors based on significance
    pearson_colors = ['#3498db' if p < 0.05 else '#bdc3c7' 
                      for p in df_sorted['pearson_pvalue']]
    partial_colors = ['#e74c3c' if p < 0.05 else '#bdc3c7' 
                      for p in df_sorted['partial_pvalue']]
    
    # Create bars
    bars1 = ax.barh(y_pos - bar_height/2, pearson_vals, bar_height, 
                    color=pearson_colors, edgecolor='black', linewidth=0.5,
                    label='Pearson Correlation')
    bars2 = ax.barh(y_pos + bar_height/2, partial_vals, bar_height,
                    color=partial_colors, edgecolor='black', linewidth=0.5,
                    label='Partial Correlation')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Pearson vs. Partial Correlation Comparison\n(Colored = Significant at p<0.05)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Pearson (significant)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Partial (significant)'),
        mpatches.Patch(facecolor='#bdc3c7', edgecolor='black', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created correlation comparison chart")


def create_correlation_heatmaps(
    comparison_data: Dict[str, pd.DataFrame],
    output_dir: str
):
    """
    Create heatmaps for Pearson, Partial, and difference matrices.
    
    Args:
        comparison_data: Dictionary with correlation matrices
        output_dir: Directory to save plots
    """
    logger.info("Creating correlation heatmaps...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Pearson correlation heatmap
    ax1 = axes[0]
    if 'pearson' in comparison_data:
        sns.heatmap(
            comparison_data['pearson'],
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            ax=ax1,
            cbar_kws={'label': 'Correlation'}
        )
        ax1.set_title('Pearson Correlation', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Output Variables', fontsize=10)
        ax1.set_ylabel('Input Variables', fontsize=10)
    
    # Partial correlation heatmap
    ax2 = axes[1]
    if 'partial' in comparison_data:
        sns.heatmap(
            comparison_data['partial'],
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            ax=ax2,
            cbar_kws={'label': 'Partial Correlation'}
        )
        ax2.set_title('Partial Correlation\n(Controlling for Other Inputs)', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Output Variables', fontsize=10)
        ax2.set_ylabel('Input Variables', fontsize=10)
    
    # Difference heatmap
    ax3 = axes[2]
    if 'difference' in comparison_data:
        sns.heatmap(
            comparison_data['difference'],
            annot=True,
            fmt='.2f',
            cmap='PuOr',
            center=0,
            ax=ax3,
            cbar_kws={'label': 'Difference (|Pearson| - |Partial|)'}
        )
        ax3.set_title('Confounding Effect\n(Difference in Absolute Correlations)', 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('Output Variables', fontsize=10)
        ax3.set_ylabel('Input Variables', fontsize=10)
    
    plt.suptitle('Correlation Analysis: Identifying Confounded Relationships', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmaps.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created correlation heatmaps")


def create_network_diagram(
    network_data: Dict,
    output_dir: str,
    layout: str = 'spring'
):
    """
    Create network diagram showing direct effects.
    
    Args:
        network_data: Dictionary with nodes and edges
        output_dir: Directory to save plots
        layout: Network layout algorithm
    """
    logger.info("Creating network diagram...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    nodes = network_data.get('nodes', [])
    edges = network_data.get('edges', [])
    
    if not nodes or not edges:
        logger.warning("No network data available")
        return
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    input_nodes = [n['id'] for n in nodes if n['type'] == 'input']
    output_nodes = [n['id'] for n in nodes if n['type'] == 'output']
    
    for node in nodes:
        G.add_node(node['id'], node_type=node['type'])
    
    # Add edges
    for edge in edges:
        G.add_edge(
            edge['source'], 
            edge['target'],
            weight=edge['abs_weight'],
            correlation=edge['weight']
        )
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        # Bipartite layout
        pos = {}
        for i, node in enumerate(input_nodes):
            pos[node] = (0, i - len(input_nodes)/2)
        for i, node in enumerate(output_nodes):
            pos[node] = (2, i - len(output_nodes)/2)
    
    # Draw input nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=input_nodes,
        node_color='#3498db',
        node_size=2000,
        alpha=0.8,
        ax=ax
    )
    
    # Draw output nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=output_nodes,
        node_color='#2ecc71',
        node_size=1500,
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges with colors based on correlation sign
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['correlation'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['correlation'] < 0]
    
    # Edge widths based on strength
    pos_widths = [G[u][v]['weight'] * 5 for u, v in positive_edges]
    neg_widths = [G[u][v]['weight'] * 5 for u, v in negative_edges]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=positive_edges,
        edge_color='#27ae60',
        width=pos_widths if pos_widths else 1,
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.1"
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=negative_edges,
        edge_color='#e74c3c',
        width=neg_widths if neg_widths else 1,
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight='bold',
        ax=ax
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='Input Variables'),
        mpatches.Patch(facecolor='#2ecc71', label='Output Variables'),
        plt.Line2D([0], [0], color='#27ae60', linewidth=3, label='Positive Partial Corr'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=3, label='Negative Partial Corr')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('Direct Effect Network\n(Edge width ∝ |Partial Correlation|, Only significant edges shown)',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direct_effect_network.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created network diagram")


def create_confounding_analysis_plot(
    partial_df: pd.DataFrame,
    output_dir: str
):
    """
    Create plot showing which relationships are confounded.
    
    Args:
        partial_df: DataFrame with partial correlation results
        output_dir: Directory to save plots
    """
    logger.info("Creating confounding analysis plot...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Scatter plot of Pearson vs Partial
    ax1 = axes[0]
    
    colors = ['#e74c3c' if row['is_direct_effect'] else '#95a5a6' 
              for _, row in partial_df.iterrows()]
    
    ax1.scatter(
        partial_df['pearson_corr'].abs(),
        partial_df['partial_corr'].abs(),
        c=colors,
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Diagonal line
    max_val = max(partial_df['pearson_corr'].abs().max(), 
                  partial_df['partial_corr'].abs().max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No confounding')
    
    ax1.set_xlabel('|Pearson Correlation|', fontsize=12)
    ax1.set_ylabel('|Partial Correlation|', fontsize=12)
    ax1.set_title('Confounding Detection\n(Below diagonal = confounded)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                  markersize=10, label='Direct Effect'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
                  markersize=10, label='Not Direct/Significant')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 2. Distribution of confounding
    ax2 = axes[1]
    
    confounding = partial_df['corr_difference'].dropna()
    ax2.hist(confounding, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.axvline(x=confounding.mean(), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean = {confounding.mean():.3f}')
    
    ax2.set_xlabel('Confounding Effect (|Pearson| - |Partial|)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Confounding Effects\n(Positive = correlation inflated by confounders)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Confounding Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confounding_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created confounding analysis plot")
