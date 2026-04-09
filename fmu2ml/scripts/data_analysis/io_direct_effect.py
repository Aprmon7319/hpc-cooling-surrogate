"""
Script for performing direct effect analysis on FMU input-output relationships.
Parallelized version using Dask and multiprocessing with modular components.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fmu2ml.analysis.input_output_relations.analyzers import (
    DirectEffectAnalyzer,
    DataGenerator
)
from fmu2ml.analysis.input_output_relations.visualizers import (
    create_correlation_heatmap,
    create_scatter_plots,
    create_importance_ranking_plots,
    create_response_surface_plots,
    create_slice_plots
)
from fmu2ml.utils.logging_utils import setup_logging
from raps.config import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform direct effect analysis on FMU inputs and outputs'
    )
    
    parser.add_argument(
        '--system-name',
        type=str,
        default='marconi100',
        help='System configuration name (e.g., marconi100, leonardo)'
    )
    
    parser.add_argument(
        '--analysis-config',
        type=str,
        default='fmu2ml/config/defaults/direct_effect.yaml',
        help='Path to analysis config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results (overrides config)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of samples for sensitivity analysis (overrides config)'
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
        help='Path to existing data file (overrides config)'
    )
    
    parser.add_argument(
        '--response-surfaces',
        action='store_true',
        help='Compute and visualize response surfaces'
    )
    
    parser.add_argument(
        '--slice-plots',
        action='store_true',
        help='Create slice plots for key outputs'
    )
    
    parser.add_argument(
        '--isolated-effects',
        action='store_true',
        help='Analyze isolated input effects (others held at mean)'
    )

    parser.add_argument(
        '--input-ranges-qflow',
        nargs=2,
        type=float,
        default=None,
        help='Q_flow input variable range as two floats: min max'
    )
    
    parser.add_argument(
        '--input-ranges-tair',
        nargs=2,
        type=float,
        default=None,
        help='T_Air input variable range as two floats: min max'
    )
    
    parser.add_argument(
        '--input-ranges-text',
        nargs=2,
        type=float,
        default=None,
        help='T_ext input variable range as two floats: min max'
    )
    
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )
    
    parser.add_argument(
        '--threads-per-worker',
        type=int,
        default=None,
        help='Threads per Dask worker (overrides config)'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=str,
        default=None,
        help='Memory limit per Dask worker (overrides config)'
    )
    
    parser.add_argument(
        '--surface-points',
        type=int,
        default=20,
        help='Number of points per dimension for response surfaces'
    )
    
    parser.add_argument(
        '--isolation-tolerance',
        type=float,
        default=0.1,
        help='Tolerance for isolated effects (fraction of std dev)'
    )
    
    return parser.parse_args()


def load_configs(args):
    """Load and merge configuration files."""
    # Load system configuration from ConfigManager
    system_config = ConfigManager(system_name=args.system_name).get_config()
    
    # Load analysis config if provided
    if os.path.exists(args.analysis_config):
        with open(args.analysis_config, 'r') as f:
            analysis_config = yaml.safe_load(f)
    else:
        analysis_config = {}
    
    # Merge configs (analysis config takes precedence)
    config = {**system_config, **analysis_config}
    
    # Override with command line arguments
    if args.output_dir:
        config['output_path'] = args.output_dir
    
    if args.data_file:
        config['data_path'] = args.data_file
    
    if args.n_samples:
        config['n_samples'] = args.n_samples
    else:
        config['n_samples'] = config.get('n_samples', 1000)
    
    # Parallel settings
    if 'parallel' not in config:
        config['parallel'] = {}
    
    if args.n_workers:
        config['parallel']['n_workers'] = args.n_workers
    if args.threads_per_worker:
        config['parallel']['threads_per_worker'] = args.threads_per_worker
    if args.memory_limit:
        config['parallel']['memory_limit'] = args.memory_limit
    
    # Set defaults for parallel settings
    config['parallel'].setdefault('n_workers', 8)
    config['parallel'].setdefault('threads_per_worker', 1)
    config['parallel'].setdefault('memory_limit', '5GB')
    
    # Input ranges
    input_ranges = {}
    if args.input_ranges_qflow:
        input_ranges['Q_flow'] = tuple(args.input_ranges_qflow)
    else:
        input_ranges['Q_flow'] = tuple(config.get('input_ranges', {}).get('Q_flow', [50.0, 200.0]))
    
    if args.input_ranges_tair:
        input_ranges['T_Air'] = tuple(args.input_ranges_tair)
    else:
        input_ranges['T_Air'] = tuple(config.get('input_ranges', {}).get('T_Air', [288.15, 308.15]))
    
    if args.input_ranges_text:
        input_ranges['T_ext'] = tuple(args.input_ranges_text)
    else:
        input_ranges['T_ext'] = tuple(config.get('input_ranges', {}).get('T_ext', [283.15, 313.15]))
    
    config['input_ranges'] = input_ranges
    config['surface_points'] = args.surface_points
    config['isolation_tolerance'] = args.isolation_tolerance
    config['system_name'] = args.system_name
    
    return config


def create_isolated_effects_comparison_plot(
    all_effects_df: pd.DataFrame,
    isolated_effects_df: pd.DataFrame,
    output_dir: str
):
    """Create comparison plots between all-data and isolated effects analysis."""
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger(__name__)
    logger.info("Creating isolated effects comparison plots...")
    
    # Merge dataframes for comparison
    all_effects_df = all_effects_df.copy()
    all_effects_df['analysis_type'] = 'full_data'
    
    comparison_df = pd.concat([
        all_effects_df[['input', 'output', 'pearson_r', 'mutual_info', 'r2_score', 'analysis_type']],
        isolated_effects_df[['input', 'output', 'pearson_r', 'mutual_info', 'r2_score', 'analysis_type']]
    ])
    
    metrics = ['pearson_r', 'mutual_info', 'r2_score']
    metric_names = ['Pearson Correlation', 'Mutual Information', 'R² Score']
    
    for metric, metric_name in zip(metrics, metric_names):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        inputs = ['Q_flow', 'T_Air', 'T_ext']
        
        for idx, input_var in enumerate(inputs):
            ax = axes[idx]
            input_data = comparison_df[comparison_df['input'] == input_var].copy()
            pivot = input_data.pivot(index='output', columns='analysis_type', values=metric)
            
            x = range(len(pivot))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], pivot['full_data'], 
                   width, label='Full Data', alpha=0.8, color='#3498db')
            ax.bar([i + width/2 for i in x], pivot['isolated'], 
                   width, label='Isolated (others at mean)', alpha=0.8, color='#e74c3c')
            
            ax.set_xlabel('Output Variables', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f'{input_var}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add difference annotations
            for i, output_name in enumerate(pivot.index):
                full_val = pivot.loc[output_name, 'full_data']
                isolated_val = pivot.loc[output_name, 'isolated']
                diff = isolated_val - full_val
                
                if abs(diff) > 0.05:  # Only show significant differences
                    y_pos = max(full_val, isolated_val) + 0.02
                    color = 'green' if diff > 0 else 'red'
                    ax.text(i, y_pos, f'{diff:+.3f}', 
                           ha='center', va='bottom', fontsize=7, color=color)
        
        plt.suptitle(
            f'Comparison: Full Data vs Isolated Effects - {metric_name}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'isolated_comparison_{metric}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Created {len(metrics)} isolated effects comparison plots")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config = load_configs(args)
    
    # Setup logging
    output_dir = config.get('output_path', f'results/direct_effect_analysis_{args.system_name}')
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(log_dir=output_dir)
    logger = logging.getLogger(__name__)
    
    num_cdus = config.get('NUM_CDUS', config.get('num_cdus', 49))
    n_samples = config['n_samples']
    input_ranges = config['input_ranges']
    
    logger.info("="*80)
    logger.info("FMU Direct Effect Analysis (Parallelized)")
    logger.info("="*80)
    logger.info(f"System: {args.system_name}")
    logger.info(f"Number of CDUs: {num_cdus}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Input ranges:")
    logger.info(f"  Q_flow: {input_ranges['Q_flow']} kW")
    logger.info(f"  T_Air: {input_ranges['T_Air']} K")
    logger.info(f"  T_ext: {input_ranges['T_ext']} K")
    logger.info(f"Parallel settings:")
    logger.info(f"  Workers: {config['parallel']['n_workers']}")
    logger.info(f"  Threads/worker: {config['parallel']['threads_per_worker']}")
    logger.info(f"  Memory limit: {config['parallel']['memory_limit']}")
    
    # Initialize data generator if needed
    data_generator = None
    if args.generate_data or args.response_surfaces:
        # Remove system_name from config to avoid duplicate keyword argument
        config_for_generator = {k: v for k, v in config.items() if k != 'system_name'}
        
        data_generator = DataGenerator(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            **config_for_generator
        )
    
    # Initialize analyzer with system_name
    # Remove system_name from config to avoid duplicate keyword argument
    config_for_analyzer = {k: v for k, v in config.items() if k != 'system_name'}
    
    analyzer = DirectEffectAnalyzer(
        system_name=args.system_name,
        n_workers=config['parallel']['n_workers'],
        threads_per_worker=config['parallel']['threads_per_worker'],
        memory_limit=config['parallel']['memory_limit'],
        **config_for_analyzer
    )
    
    # Generate or load data
    if args.generate_data:
        logger.info("="*80)
        logger.info("STEP 1: Generating Simulation Data")
        logger.info("="*80)
        
        data = data_generator.generate_sensitivity_data(
            n_samples=n_samples,
            input_ranges=input_ranges
        )
        
        data_file = os.path.join(output_dir, 'sensitivity_data.parquet')
        data.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")
        
    elif config.get('data_path') and os.path.exists(config['data_path']):
        logger.info(f"Loading data from {config['data_path']}")
        data = pd.read_parquet(config['data_path'])
    else:
        logger.error("Must specify --generate-data or provide valid data_path in config")
        return
    
    logger.info(f"Data shape: {data.shape}")
    
    # Prepare data
    logger.info("="*80)
    logger.info("STEP 2: Preparing Data")
    logger.info("="*80)
    prepared_data = analyzer.prepare_data(data)
    
    # Perform univariate analysis (parallelized)
    logger.info("="*80)
    logger.info("STEP 3: Univariate Sensitivity Analysis")
    logger.info("="*80)
    metrics_df = analyzer.analyze_all_pairs(prepared_data)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'sensitivity_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Rank inputs by importance
    logger.info("="*80)
    logger.info("STEP 4: Ranking Input Importance")
    logger.info("="*80)
    rankings_df = analyzer.rank_input_importance(metrics_df, metric='mutual_info')
    rankings_file = os.path.join(output_dir, 'input_rankings.csv')
    rankings_df.to_csv(rankings_file, index=False)
    logger.info(f"Saved rankings to {rankings_file}")
    
    # Analyze isolated effects (if requested)
    isolated_effects_df = None
    if args.isolated_effects:
        logger.info("="*80)
        logger.info("STEP 5: Analyzing Isolated Input Effects")
        logger.info("="*80)
        
        isolated_effects_df = analyzer.analyze_isolated_effects(
            prepared_data,
            tolerance=config['isolation_tolerance']
        )
        
        if not isolated_effects_df.empty:
            # Save isolated effects
            isolated_file = os.path.join(output_dir, 'isolated_effects.csv')
            isolated_effects_df.to_csv(isolated_file, index=False)
            logger.info(f"Saved isolated effects to {isolated_file}")
            
            # Create comparison plots
            create_isolated_effects_comparison_plot(
                metrics_df,
                isolated_effects_df,
                output_dir
            )
        else:
            logger.warning("No isolated effects could be computed")
    
    # Create visualizations (parallelized)
    step_num = 6 if args.isolated_effects else 5
    logger.info("="*80)
    logger.info(f"STEP {step_num}: Creating Visualizations")
    logger.info("="*80)
    
    # Correlation heatmaps (parallelized)
    create_correlation_heatmap(
        metrics_df,
        output_dir,
        n_workers=config['parallel']['n_workers']
    )
    
    # Scatter plots (parallelized)
    create_scatter_plots(
        prepared_data,
        metrics_df,
        output_dir,
        n_workers=config['parallel']['n_workers']
    )
    
    # Importance rankings
    create_importance_ranking_plots(rankings_df, output_dir)
    
    # Slice plots (if requested)
    if args.slice_plots:
        logger.info("Creating slice plots...")
        key_outputs = ['T_prim_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM']
        
        # Create slices for Q_flow vs T_Air
        create_slice_plots(
            prepared_data,
            output_vars=key_outputs,
            input1='Q_flow',
            input2='T_Air',
            n_slices=5,
            output_dir=output_dir
        )
        
        # Create slices for Q_flow vs T_ext
        create_slice_plots(
            prepared_data,
            output_vars=key_outputs,
            input1='Q_flow',
            input2='T_ext',
            n_slices=5,
            output_dir=output_dir
        )
        
        logger.info("Slice plots created")
    
    # Response surfaces (if requested)
    if args.response_surfaces:
        step_num += 1
        logger.info("="*80)
        logger.info(f"STEP {step_num}: Generating and Analyzing Response Surfaces")
        logger.info("="*80)
        
        key_outputs = ['T_prim_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM']
        input_pairs = [
            ('Q_flow', 'T_Air'),
            ('Q_flow', 'T_ext'),
            ('T_Air', 'T_ext')
        ]
        
        # Fixed inputs for response surface generation (use middle of ranges)
        fixed_inputs = {
            'Q_flow': (input_ranges['Q_flow'][0] + input_ranges['Q_flow'][1]) / 2 * 1000,  # Convert to W
            'T_Air': (input_ranges['T_Air'][0] + input_ranges['T_Air'][1]) / 2,
            'T_ext': (input_ranges['T_ext'][0] + input_ranges['T_ext'][1]) / 2
        }
        
        surfaces = {}
        
        for output_name in key_outputs:
            for input1, input2 in input_pairs:
                logger.info(f"Generating response surface for {output_name} vs {input1} x {input2}")
                
                # Prepare input ranges
                input1_range = input_ranges[input1]
                input2_range = input_ranges[input2]
                
                # Convert Q_flow to Watts if needed
                if input1 == 'Q_flow':
                    input1_range = (input1_range[0] * 1000, input1_range[1] * 1000)
                if input2 == 'Q_flow':
                    input2_range = (input2_range[0] * 1000, input2_range[1] * 1000)
                
                # Generate response surface data
                surface_data = data_generator.generate_response_surface_data(
                    input1=input1,
                    input2=input2,
                    input1_range=input1_range,
                    input2_range=input2_range,
                    n_points_per_dim=config['surface_points'],
                    fixed_inputs=fixed_inputs
                )
                
                if not surface_data.empty:
                    # Prepare surface data for analysis
                    surface_prepared = analyzer.prepare_data(surface_data)
                    
                    # Compute response surface
                    X, Y, Z = analyzer.compute_response_surface(
                        surface_prepared,
                        input1,
                        input2,
                        output_name,
                        n_points=config['surface_points']
                    )
                    
                    surfaces[(output_name, input1, input2)] = (X, Y, Z)
                    logger.info(f"  Successfully computed surface for {output_name}")
                else:
                    logger.warning(f"  Failed to generate surface data for {output_name}")
        
        # Create surface plots in parallel
        if surfaces:
            logger.info("Creating response surface visualizations...")
            create_response_surface_plots(
                surfaces,
                output_dir,
                n_workers=config['parallel']['n_workers']
            )
            logger.info(f"Created {len(surfaces)} response surface plots")
        else:
            logger.warning("No response surfaces were generated")
    
    # Analyze interaction effects (parallelized)
    step_num += 1
    logger.info("="*80)
    logger.info(f"STEP {step_num}: Analyzing Interaction Effects")
    logger.info("="*80)
    interaction_df = analyzer.analyze_all_interaction_effects(prepared_data)
    
    interaction_file = os.path.join(output_dir, 'interaction_effects.csv')
    interaction_df.to_csv(interaction_file, index=False)
    logger.info(f"Saved interaction effects to {interaction_file}")
    
    # Close Dask client
    analyzer._close_dask_client()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*80)
    
    logger.info("\nTop 3 Most Important Inputs per Output (by Mutual Information):")
    for output in analyzer.output_vars:
        output_rankings = rankings_df[rankings_df['output'] == output].head(3)
        if not output_rankings.empty:
            logger.info(f"\n{output}:")
            for _, row in output_rankings.iterrows():
                logger.info(f"  {row['rank']}. {row['input']}: {row['value']:.4f}")
    
    if args.isolated_effects and isolated_effects_df is not None and not isolated_effects_df.empty:
        logger.info("\n" + "-"*80)
        logger.info("Isolated Effects Analysis:")
        logger.info("-"*80)
        
        for input_var in ['Q_flow', 'T_Air', 'T_ext']:
            input_data = isolated_effects_df[isolated_effects_df['input'] == input_var]
            if not input_data.empty:
                logger.info(f"\n{input_var} (with others at mean):")
                # Show top 3 outputs by mutual information
                top_outputs = input_data.nlargest(3, 'mutual_info')
                for _, row in top_outputs.iterrows():
                    logger.info(
                        f"  → {row['output']}: MI={row['mutual_info']:.4f}, "
                        f"R²={row['r2_score']:.4f} (n={int(row['n_samples'])})"
                    )
    
    logger.info("\nSignificant Interaction Effects (R² improvement > 0.05):")
    significant_interactions = interaction_df[interaction_df['interaction_effect'] > 0.05]
    if not significant_interactions.empty:
        for _, row in significant_interactions.iterrows():
            logger.info(
                f"  {row['output']}: +{row['interaction_effect']:.4f} "
                f"(Linear R²: {row['r2_linear']:.4f} → Polynomial R²: {row['r2_polynomial']:.4f})"
            )
    else:
        logger.info("  No significant interaction effects detected")
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()