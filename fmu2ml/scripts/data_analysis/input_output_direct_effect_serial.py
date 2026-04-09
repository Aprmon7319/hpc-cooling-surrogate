"""
Script for performing direct effect analysis on FMU input-output relationships.

This script:
1. Generates simulation data suitable for analysis
2. Performs univariate sensitivity analysis
3. Computes response surfaces
4. Analyzes interaction effects
5. Creates comprehensive visualizations
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fmu2ml.analysis.input_output_relations.direct_effect_analysis_serial import (
    DirectEffectAnalyzer,
    DataGenerator
)
from fmu2ml.simulation.fmu_simulator import FMUSimulator
from fmu2ml.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform direct effect analysis on FMU inputs and outputs'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='fmu2ml/config/defaults/fmu_simulation.yaml',
        help='Path to simulation config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/direct_effect_analysis',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for sensitivity analysis'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate new simulation data'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to existing data file'
    )
    
    parser.add_argument(
        '--response-surfaces',
        action='store_true',
        help='Compute response surfaces'
    )

    parser.add_argument(
        '--input-ranges-qflow',
        nargs=2,
        type=float,
        default=[12.0, 40.0],
        help='Q_flow input variable range as two floats: min max'
    )
    parser.add_argument(
        '--input-ranges-tair',
        nargs=2,
        type=float,
        default=[288.15, 308.15],
        help='T_Air input variable range as two floats: min max'
    )
    parser.add_argument(
        '--input-ranges-text',
        nargs=2,
        type=float,
        default=[283.15, 313.15],
        help='T_ext input variable range as two floats: min max'
    )
    
    return parser.parse_args()


def create_correlation_heatmap(metrics_df, output_dir):
    """Create correlation heatmap for all input-output pairs."""
    logger = logging.getLogger(__name__)
    logger.info("Creating correlation heatmap...")
    
    # Pivot data for heatmap
    for metric in ['pearson_r', 'spearman_r', 'mutual_info', 'r2_score']:
        pivot_data = metrics_df.pivot(
            index='input',
            columns='output',
            values=metric
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Create heatmap
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


def create_scatter_plots(prepared_data, metrics_df, output_dir):
    """Create scatter plots with regression lines for all pairs."""
    logger = logging.getLogger(__name__)
    logger.info("Creating scatter plots...")
    
    input_vars = ['Q_flow', 'T_Air', 'T_ext']
    output_vars = metrics_df['output'].unique()
    
    # Create subplots for each input
    for input_name in input_vars:
        n_outputs = len(output_vars)
        n_cols = 4
        n_rows = (n_outputs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        input_data = prepared_data['inputs'][input_name]
        if input_data.ndim > 1:
            input_flat = input_data.flatten()
        else:
            # Will be repeated for each output
            input_flat = input_data
        
        for idx, output_name in enumerate(output_vars):
            ax = axes[idx]
            
            # Get output data
            output_data = prepared_data['outputs'][output_name].flatten()
            
            # Repeat input if needed
            if input_flat.ndim == 1 and len(input_flat) != len(output_data):
                input_plot = np.repeat(input_flat, output_data.size // input_flat.size)
            else:
                input_plot = input_flat
            
            # Remove NaN
            mask = ~(np.isnan(input_plot) | np.isnan(output_data))
            input_clean = input_plot[mask]
            output_clean = output_data[mask]
            
            # Sample for plotting if too many points
            if len(input_clean) > 1000:
                sample_idx = np.random.choice(len(input_clean), 1000, replace=False)
                input_clean = input_clean[sample_idx]
                output_clean = output_clean[sample_idx]
            
            # Scatter plot
            ax.scatter(input_clean, output_clean, alpha=0.5, s=10)
            
            # Get metrics for this pair
            pair_metrics = metrics_df[
                (metrics_df['input'] == input_name) &
                (metrics_df['output'] == output_name)
            ].iloc[0]
            
            # Add regression line
            x_line = np.linspace(input_clean.min(), input_clean.max(), 100)
            y_line = pair_metrics['linear_coef'] * x_line + pair_metrics['linear_intercept']
            ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')
            
            # Add metrics to plot
            ax.text(
                0.05, 0.95,
                f"R² = {pair_metrics['r2_score']:.3f}\n"
                f"Pearson = {pair_metrics['pearson_r']:.3f}\n"
                f"MI = {pair_metrics['mutual_info']:.3f}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8
            )
            
            ax.set_xlabel(input_name, fontsize=10)
            ax.set_ylabel(output_name, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(output_vars), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Direct Effects: {input_name} on All Outputs',
            fontsize=16,
            y=1.00
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'scatter_plots_{input_name}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def create_importance_ranking_plots(rankings_df, output_dir):
    """Create bar charts showing input importance for each output."""
    logger = logging.getLogger(__name__)
    logger.info("Creating importance ranking plots...")
    
    output_vars = rankings_df['output'].unique()
    n_outputs = len(output_vars)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, output_name in enumerate(output_vars):
        ax = axes[idx]
        
        output_rankings = rankings_df[rankings_df['output'] == output_name].copy()
        output_rankings = output_rankings.sort_values('rank')
        
        bars = ax.barh(
            output_rankings['input'],
            output_rankings['value'],
            color=['#2ecc71', '#3498db', '#e74c3c'][:len(output_rankings)]
        )
        
        ax.set_xlabel('Mutual Information', fontsize=10)
        ax.set_ylabel('Input Variables', fontsize=10)
        ax.set_title(output_name, fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=8
            )
    
    # Hide unused subplots
    for idx in range(len(output_vars), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Input Importance Rankings by Output', fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'importance_rankings.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def create_response_surface_plots(analyzer, prepared_data, output_dir):
    """Create 3D surface and contour plots for key relationships."""
    logger = logging.getLogger(__name__)
    logger.info("Creating response surface plots...")
    
    # Key output variables to analyze
    key_outputs = ['T_prim_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM']
    
    # Input pairs to analyze
    input_pairs = [
        ('Q_flow', 'T_Air'),
        ('Q_flow', 'T_ext'),
        ('T_Air', 'T_ext')
    ]
    
    for output_name in key_outputs:
        for input1, input2 in input_pairs:
            try:
                # Compute response surface
                X, Y, Z = analyzer.compute_response_surface(
                    prepared_data,
                    input1,
                    input2,
                    output_name,
                    n_points=50
                )
                
                # Create figure with 3D surface and contour plot
                fig = plt.figure(figsize=(16, 6))
                
                # 3D Surface plot
                ax1 = fig.add_subplot(121, projection='3d')
                surf = ax1.plot_surface(
                    X, Y, Z,
                    cmap=cm.viridis,
                    linewidth=0,
                    antialiased=True,
                    alpha=0.8
                )
                ax1.set_xlabel(input1, fontsize=11)
                ax1.set_ylabel(input2, fontsize=11)
                ax1.set_zlabel(output_name, fontsize=11)
                ax1.set_title(f'3D Surface: {output_name}', fontsize=12)
                fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
                
                # Contour plot
                ax2 = fig.add_subplot(122)
                contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
                ax2.set_xlabel(input1, fontsize=11)
                ax2.set_ylabel(input2, fontsize=11)
                ax2.set_title(f'Contour: {output_name}', fontsize=12)
                fig.colorbar(contour, ax=ax2)
                
                plt.suptitle(
                    f'Response Surface: {output_name} vs {input1} × {input2}',
                    fontsize=14
                )
                plt.tight_layout()
                
                filename = f'response_surface_{output_name}_{input1}_{input2}.png'
                plt.savefig(
                    os.path.join(output_dir, filename),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
            except Exception as e:
                logger.warning(f"Failed to create surface plot for {output_name}: {e}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_dir=args.output_dir)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_cdus = config.get('NUM_CDUS', 49)
    input_ranges = {
        'Q_flow': args.input_ranges_qflow,
        'T_Air': args.input_ranges_tair,
        'T_ext': args.input_ranges_text
    }
    
    logger.info("="*80)
    logger.info("FMU Direct Effect Analysis")
    logger.info("="*80)
    logger.info(f"Number of CDUs: {num_cdus}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = DirectEffectAnalyzer(num_cdus=num_cdus)
    
    # Generate or load data
    if args.generate_data:
        logger.info("Generating simulation data...")
        
        # Initialize simulator
        simulator = FMUSimulator(config=config)
        data_generator = DataGenerator(simulator, num_cdus, config)
        
        # Generate data
        data = data_generator.generate_sensitivity_data(n_samples=args.n_samples, input_ranges=input_ranges)
        
        # Save data
        data_file = os.path.join(args.output_dir, 'sensitivity_data.parquet')
        data.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")
        
    elif args.data_file:
        logger.info(f"Loading data from {args.data_file}")
        data = pd.read_parquet(args.data_file)
    else:
        logger.error("Must specify --generate-data or --data-file")
        return
    
    logger.info(f"Data shape: {data.shape}")
    
    # Prepare data
    prepared_data = analyzer.prepare_data(data)
    
    # Perform univariate analysis
    logger.info("Performing univariate sensitivity analysis...")
    metrics_df = analyzer.analyze_all_pairs(prepared_data)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'sensitivity_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Rank inputs by importance
    rankings_df = analyzer.rank_input_importance(metrics_df, metric='mutual_info')
    rankings_file = os.path.join(args.output_dir, 'input_rankings.csv')
    rankings_df.to_csv(rankings_file, index=False)
    logger.info(f"Saved rankings to {rankings_file}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Correlation heatmaps
    create_correlation_heatmap(metrics_df, args.output_dir)
    
    # Scatter plots
    create_scatter_plots(prepared_data, metrics_df, args.output_dir)
    
    # Importance rankings
    create_importance_ranking_plots(rankings_df, args.output_dir)
    
    # Response surfaces (if requested)
    if args.response_surfaces:
        create_response_surface_plots(analyzer, prepared_data, args.output_dir)
    
    # Analyze interaction effects
    logger.info("Analyzing interaction effects...")
    interaction_results = []
    
    for output in analyzer.output_vars:
        interaction_metrics = analyzer.analyze_interaction_effects(
            prepared_data,
            output
        )
        interaction_results.append(interaction_metrics)
    
    interaction_df = pd.DataFrame(interaction_results)
    interaction_file = os.path.join(args.output_dir, 'interaction_effects.csv')
    interaction_df.to_csv(interaction_file, index=False)
    logger.info(f"Saved interaction effects to {interaction_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Analysis Summary")
    logger.info("="*80)
    
    logger.info("\nTop 3 Most Important Inputs per Output:")
    for output in analyzer.output_vars:
        output_rankings = rankings_df[rankings_df['output'] == output].head(3)
        logger.info(f"\n{output}:")
        for _, row in output_rankings.iterrows():
            logger.info(f"  {row['rank']}. {row['input']}: {row['value']:.3f}")
    
    logger.info("\nInteraction Effects (R² improvement):")
    for _, row in interaction_df.iterrows():
        if row['interaction_effect'] > 0.05:  # Significant interaction
            logger.info(
                f"  {row['output']}: {row['interaction_effect']:.3f} "
                f"(Linear: {row['r2_linear']:.3f}, Poly: {row['r2_polynomial']:.3f})"
            )
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()