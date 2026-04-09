import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

logger = logging.getLogger(__name__)


def _create_single_heatmap(args):
    """Create a single heatmap for a given metric. Used for parallel execution."""
    metric, pivot_data, output_dir = args
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0 if metric in ['pearson_r', 'spearman_r'] else None,
        vmin=-1 if metric in ['pearson_r', 'spearman_r'] else 0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    ax.set_title(f'{metric.replace("_", " ").title()} - Inputs vs Outputs', fontsize=14)
    ax.set_xlabel('Output Variables', fontsize=12)
    ax.set_ylabel('Input Variables', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'heatmap_{metric}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    return metric


def create_correlation_heatmap(metrics_df: pd.DataFrame, output_dir: str, n_workers: int = 8):
    """Create correlation heatmap for all input-output pairs using parallel processing."""
    logger.info("Creating correlation heatmaps in parallel...")
    
    metrics = ['pearson_r', 'spearman_r', 'mutual_info', 'r2_score']
    
    # Prepare arguments for parallel processing
    args_list = []
    for metric in metrics:
        pivot_data = metrics_df.pivot(
            index='input',
            columns='output',
            values=metric
        )
        args_list.append((metric, pivot_data, output_dir))
    
    # Use multiprocessing pool
    with mp.Pool(processes=min(n_workers, len(metrics))) as pool:
        results = pool.map(_create_single_heatmap, args_list)
    
    logger.info(f"Created {len(results)} heatmaps")