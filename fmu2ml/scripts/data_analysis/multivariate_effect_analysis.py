"""
Script for performing multivariate effect analysis on FMU input-output relationships.

Analyzes:
- Partial correlation analysis (direct vs confounded effects)
- Multiple regression analysis (beta coefficients, significance, VIF)
- Lagged effect analysis (distributed lag models)
- Autocorrelation analysis (ACF, PACF, cross-correlation)

Parallelized version using Dask and multiprocessing.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fmu2ml.analysis.multivariate_effect_analysis.analyzers import (
    PartialCorrelationAnalyzer,
    MultipleRegressionAnalyzer,
    LaggedEffectAnalyzer,
    AutocorrelationAnalyzer
)
from fmu2ml.analysis.multivariate_effect_analysis.visualizers import (
    create_correlation_comparison_chart,
    create_correlation_heatmaps,
    create_network_diagram,
    create_confounding_analysis_plot,
    create_beta_coefficient_plot,
    create_significance_heatmap,
    create_variance_explained_chart,
    create_vif_plot,
    create_regression_summary_plot,
    create_lag_coefficient_plots,
    create_lag_heatmap,
    create_memory_length_plot,
    create_cumulative_effect_summary,
    create_lag_analysis_summary,
    create_acf_plots,
    create_ccf_plots,
    create_persistence_summary,
    create_ccf_heatmap,
    create_temporal_recommendations_plot
)
from fmu2ml.analysis.input_output_relations.analyzers import DataGenerator
from fmu2ml.utils.logging_utils import setup_logging
from raps.config import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform multivariate effect analysis on FMU inputs and outputs'
    )
    
    parser.add_argument(
        '--system-name',
        type=str,
        default='marconi100',
        help='System configuration name (e.g., marconi100, leonardo, summit)'
    )
    
    parser.add_argument(
        '--analysis-config',
        type=str,
        default='fmu2ml/config/defaults/multivariate_effect_analysis.yaml',
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
        help='Number of samples for analysis (overrides config)'
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
        '--max-lag',
        type=int,
        default=None,
        help='Maximum lag for lagged effect and autocorrelation analysis'
    )
    
    parser.add_argument(
        '--skip-partial-corr',
        action='store_true',
        help='Skip partial correlation analysis'
    )
    
    parser.add_argument(
        '--skip-regression',
        action='store_true',
        help='Skip multiple regression analysis'
    )
    
    parser.add_argument(
        '--skip-lagged',
        action='store_true',
        help='Skip lagged effect analysis'
    )
    
    parser.add_argument(
        '--skip-autocorr',
        action='store_true',
        help='Skip autocorrelation analysis'
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
        '--key-outputs',
        nargs='+',
        type=str,
        default=None,
        help='Key output variables to focus on (for detailed visualization)'
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
    
    # Merge configs
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
    
    if args.max_lag:
        config['max_lag'] = args.max_lag
    else:
        config['max_lag'] = config.get('lagged_effect', {}).get('max_lag', 30)
    
    # Parallel settings
    if 'parallel' not in config:
        config['parallel'] = {}
    
    if args.n_workers:
        config['parallel']['n_workers'] = args.n_workers
    if args.threads_per_worker:
        config['parallel']['threads_per_worker'] = args.threads_per_worker
    if args.memory_limit:
        config['parallel']['memory_limit'] = args.memory_limit
    
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
    config['system_name'] = args.system_name
    
    # Key outputs
    if args.key_outputs:
        config['key_outputs'] = args.key_outputs
    else:
        config['key_outputs'] = config.get('key_outputs', [
            'T_prim_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM'
        ])
    
    return config


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config = load_configs(args)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.get('output_path'):
        output_dir = config['output_path']
    else:
        output_dir = f'analysis_results/multivariate_effect_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir=output_dir)
    logger = logging.getLogger(__name__)
    
    num_cdus = config.get('NUM_CDUS', config.get('num_cdus', 49))
    n_samples = config['n_samples']
    input_ranges = config['input_ranges']
    
    logger.info("=" * 80)
    logger.info("FMU Multivariate Effect Analysis")
    logger.info("=" * 80)
    logger.info(f"System: {args.system_name}")
    logger.info(f"Number of CDUs: {num_cdus}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Max lag: {config['max_lag']}")
    logger.info(f"Input ranges:")
    logger.info(f"  Q_flow: {input_ranges['Q_flow']} kW")
    logger.info(f"  T_Air: {input_ranges['T_Air']} K")
    logger.info(f"  T_ext: {input_ranges['T_ext']} K")
    logger.info(f"Parallel settings:")
    logger.info(f"  Workers: {config['parallel']['n_workers']}")
    logger.info(f"  Threads/worker: {config['parallel']['threads_per_worker']}")
    logger.info(f"  Memory limit: {config['parallel']['memory_limit']}")
    
    # Save config
    with open(os.path.join(output_dir, 'analysis_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize data generator if needed
    data_generator = None
    if args.generate_data:
        config_for_generator = {k: v for k, v in config.items() if k != 'system_name'}
        
        data_generator = DataGenerator(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            **config_for_generator
        )
    
    # Generate or load data
    if args.generate_data:
        logger.info("=" * 80)
        logger.info("STEP 1: Generating Simulation Data")
        logger.info("=" * 80)
        
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
    
    step_num = 1 if not args.generate_data else 1
    
    # =====================================================================
    # PARTIAL CORRELATION ANALYSIS
    # =====================================================================
    if not args.skip_partial_corr:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Partial Correlation Analysis")
        logger.info("=" * 80)
        
        # Initialize analyzer
        partial_analyzer = PartialCorrelationAnalyzer(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            threads_per_worker=config['parallel']['threads_per_worker'],
            memory_limit=config['parallel']['memory_limit'],
            significance_threshold=config.get('partial_correlation', {}).get('significance_threshold', 0.05),
            edge_threshold=config.get('partial_correlation', {}).get('edge_threshold', 0.1),
            method=config.get('partial_correlation', {}).get('method', 'pearson')
        )
        
        # Prepare data
        prepared_data = partial_analyzer.prepare_data(data)
        
        # Analyze
        logger.info("Computing partial correlations...")
        partial_df = partial_analyzer.analyze_partial_correlations(prepared_data)
        
        partial_file = os.path.join(output_dir, 'partial_correlations.csv')
        partial_df.to_csv(partial_file, index=False)
        logger.info(f"Saved partial correlations to {partial_file}")
        
        # Get comparison matrices
        comparison_data = partial_analyzer.compute_correlation_comparison(prepared_data)
        
        # Build network data
        network_data = partial_analyzer.build_network_data(comparison_data['full_results'])
        
        # Create visualizations
        logger.info("Creating partial correlation visualizations...")
        
        viz_dir = os.path.join(output_dir, 'visualizations', 'partial_correlation')
        os.makedirs(viz_dir, exist_ok=True)
        
        create_correlation_comparison_chart(partial_df, viz_dir)
        create_correlation_heatmaps(comparison_data, viz_dir)
        create_network_diagram(network_data, viz_dir, 
                              layout=config.get('visualization', {}).get('network_layout', 'spring'))
        create_confounding_analysis_plot(partial_df, viz_dir)
        
        partial_analyzer._close_dask_client()
        
        logger.info("Partial correlation analysis complete")
    
    # =====================================================================
    # MULTIPLE REGRESSION ANALYSIS
    # =====================================================================
    if not args.skip_regression:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Multiple Regression Analysis")
        logger.info("=" * 80)
        
        # Initialize analyzer
        regression_analyzer = MultipleRegressionAnalyzer(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            threads_per_worker=config['parallel']['threads_per_worker'],
            memory_limit=config['parallel']['memory_limit'],
            vif_threshold=config.get('multiple_regression', {}).get('vif_threshold', 10.0),
            significance_threshold=config.get('multiple_regression', {}).get('significance_threshold', 0.05),
            standardize=config.get('multiple_regression', {}).get('standardize', True),
            bootstrap_samples=config.get('multiple_regression', {}).get('bootstrap_samples', 1000)
        )
        
        # Prepare data
        prepared_data = regression_analyzer.prepare_data(data)
        
        # Analyze
        logger.info("Performing multiple regression analysis...")
        regression_df = regression_analyzer.analyze_multiple_regression(prepared_data)
        
        regression_file = os.path.join(output_dir, 'multiple_regression.csv')
        regression_df.to_csv(regression_file, index=False)
        logger.info(f"Saved regression results to {regression_file}")
        
        # Extract matrices for visualization
        beta_matrix = regression_analyzer.compute_beta_coefficient_matrix(regression_df)
        pvalue_matrix = regression_analyzer.compute_significance_matrix(regression_df)
        vif_summary = regression_analyzer.compute_vif_summary(regression_df)
        variance_df = regression_analyzer.compute_variance_explained(regression_df)
        
        # Create visualizations
        logger.info("Creating regression visualizations...")
        
        viz_dir = os.path.join(output_dir, 'visualizations', 'regression')
        os.makedirs(viz_dir, exist_ok=True)
        
        create_beta_coefficient_plot(regression_df, viz_dir)
        create_significance_heatmap(pvalue_matrix, viz_dir)
        create_variance_explained_chart(variance_df, viz_dir)
        create_vif_plot(vif_summary, viz_dir, threshold=regression_analyzer.vif_threshold)
        create_regression_summary_plot(regression_df, viz_dir)
        
        regression_analyzer._close_dask_client()
        
        logger.info("Multiple regression analysis complete")
    
    # =====================================================================
    # LAGGED EFFECT ANALYSIS
    # =====================================================================
    if not args.skip_lagged:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Lagged Effect Analysis (Distributed Lag Models)")
        logger.info("=" * 80)
        
        # Initialize analyzer
        lagged_analyzer = LaggedEffectAnalyzer(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            threads_per_worker=config['parallel']['threads_per_worker'],
            memory_limit=config['parallel']['memory_limit'],
            max_lag=config['max_lag'],
            min_lag=config.get('lagged_effect', {}).get('min_lag', 0),
            lag_step=config.get('lagged_effect', {}).get('lag_step', 1),
            cv_folds=config.get('lagged_effect', {}).get('cv_folds', 5),
            significance_threshold=config.get('lagged_effect', {}).get('significance_threshold', 0.05)
        )
        
        # Prepare data
        prepared_data = lagged_analyzer.prepare_data(data)
        
        # Analyze
        logger.info("Analyzing lagged effects...")
        lag_df = lagged_analyzer.analyze_lagged_effects(prepared_data)
        
        lag_file = os.path.join(output_dir, 'lagged_effects.csv')
        lag_df.to_csv(lag_file, index=False)
        logger.info(f"Saved lag analysis to {lag_file}")
        
        # Compute summaries
        cumulative_df = lagged_analyzer.compute_cumulative_effects(lag_df)
        memory_summary = lagged_analyzer.get_memory_length_summary(lag_df)
        
        # Create visualizations
        logger.info("Creating lagged effect visualizations...")
        
        viz_dir = os.path.join(output_dir, 'visualizations', 'lagged_effects')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get full lag results for detailed plots
        if hasattr(lagged_analyzer, '_full_lag_results'):
            create_lag_coefficient_plots(
                lagged_analyzer._full_lag_results,
                viz_dir,
                n_workers=config['parallel']['n_workers']
            )
        
        create_lag_heatmap(lag_df, viz_dir)
        create_memory_length_plot(memory_summary, viz_dir)
        create_cumulative_effect_summary(cumulative_df, viz_dir)
        create_lag_analysis_summary(lag_df, viz_dir)
        
        lagged_analyzer._close_dask_client()
        
        logger.info("Lagged effect analysis complete")
    
    # =====================================================================
    # AUTOCORRELATION ANALYSIS
    # =====================================================================
    if not args.skip_autocorr:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Autocorrelation Analysis")
        logger.info("=" * 80)
        
        # Initialize analyzer
        autocorr_analyzer = AutocorrelationAnalyzer(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            threads_per_worker=config['parallel']['threads_per_worker'],
            memory_limit=config['parallel']['memory_limit'],
            max_lag=config.get('autocorrelation', {}).get('max_lag', 50),
            confidence_level=config.get('autocorrelation', {}).get('confidence_level', 0.95),
            compute_pacf=config.get('autocorrelation', {}).get('partial_autocorrelation', True),
            compute_ccf=config.get('autocorrelation', {}).get('cross_correlation', True)
        )
        
        # Prepare data
        prepared_data = autocorr_analyzer.prepare_data(data)
        
        # Analyze autocorrelation
        logger.info("Computing ACF and PACF...")
        input_acf_df, output_acf_df = autocorr_analyzer.analyze_autocorrelation(prepared_data)
        
        input_acf_file = os.path.join(output_dir, 'input_acf.csv')
        output_acf_file = os.path.join(output_dir, 'output_acf.csv')
        input_acf_df.to_csv(input_acf_file, index=False)
        output_acf_df.to_csv(output_acf_file, index=False)
        logger.info(f"Saved ACF analysis to {input_acf_file} and {output_acf_file}")
        
        # Analyze cross-correlation
        logger.info("Computing cross-correlations...")
        ccf_df = autocorr_analyzer.analyze_cross_correlation(prepared_data)
        
        ccf_file = os.path.join(output_dir, 'cross_correlations.csv')
        ccf_df.to_csv(ccf_file, index=False)
        logger.info(f"Saved CCF analysis to {ccf_file}")
        
        # Compute temporal recommendations
        recommendations_df = autocorr_analyzer.compute_temporal_recommendations(output_acf_df, ccf_df)
        
        rec_file = os.path.join(output_dir, 'temporal_recommendations.csv')
        recommendations_df.to_csv(rec_file, index=False)
        logger.info(f"Saved temporal recommendations to {rec_file}")
        
        # Create visualizations
        logger.info("Creating autocorrelation visualizations...")
        
        viz_dir = os.path.join(output_dir, 'visualizations', 'autocorrelation')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get full ACF results for detailed plots
        if hasattr(autocorr_analyzer, '_acf_results'):
            create_acf_plots(
                autocorr_analyzer._acf_results,
                viz_dir,
                n_workers=config['parallel']['n_workers']
            )
        
        # Get full CCF results for detailed plots
        if hasattr(autocorr_analyzer, '_ccf_results'):
            create_ccf_plots(
                autocorr_analyzer._ccf_results,
                viz_dir,
                n_workers=config['parallel']['n_workers']
            )
        
        create_persistence_summary(input_acf_df, output_acf_df, viz_dir)
        create_ccf_heatmap(ccf_df, viz_dir)
        create_temporal_recommendations_plot(recommendations_df, viz_dir)
        
        autocorr_analyzer._close_dask_client()
        
        logger.info("Autocorrelation analysis complete")
    
    # =====================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =====================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    if not args.skip_partial_corr and 'partial_df' in locals():
        logger.info("\n--- Partial Correlation Analysis ---")
        
        direct_effects = partial_df[partial_df['is_direct_effect']]
        logger.info(f"Direct effects identified: {len(direct_effects)}/{len(partial_df)}")
        
        if not direct_effects.empty:
            logger.info("\nTop 5 strongest direct effects:")
            # Sort by absolute value of partial_corr instead of using key argument
            direct_effects_sorted = direct_effects.copy()
            direct_effects_sorted['abs_partial_corr'] = direct_effects_sorted['partial_corr'].abs()
            top_direct = direct_effects_sorted.nlargest(5, 'abs_partial_corr')
            for _, row in top_direct.iterrows():
                logger.info(
                    f"  {row['input']} → {row['output']}: "
                    f"partial r = {row['partial_corr']:.3f} "
                    f"(Pearson r = {row['pearson_corr']:.3f})"
                )
    
    if not args.skip_regression and 'regression_df' in locals():
        logger.info("\n--- Multiple Regression Analysis ---")
        
        mean_r2 = regression_df['r2'].mean()
        logger.info(f"Mean R² across outputs: {mean_r2:.3f}")
        
        well_explained = regression_df[regression_df['r2'] > 0.7]
        logger.info(f"Outputs well explained (R² > 0.7): {len(well_explained)}/{len(regression_df)}")
        
        # Check multicollinearity
        if 'vif_summary' in locals() and not vif_summary.empty:
            high_vif = vif_summary[vif_summary['multicollinearity_warning']]
            if not high_vif.empty:
                logger.warning(f"⚠ Multicollinearity warning for: {list(high_vif['input'])}")
    
    if not args.skip_lagged and 'lag_df' in locals():
        logger.info("\n--- Lagged Effect Analysis ---")
        
        needs_sequence = lag_df['needs_sequence_input'].sum()
        logger.info(f"Relationships needing sequence inputs: {needs_sequence}/{len(lag_df)}")
        
        mean_memory = lag_df['memory_length'].mean()
        logger.info(f"Average memory length: {mean_memory:.1f} lags")
        
        if needs_sequence > len(lag_df) / 2:
            logger.info("RECOMMENDATION: Use LSTM or sequence model architecture")
        else:
            logger.info("RECOMMENDATION: Feedforward with lagged features may suffice")
    
    if not args.skip_autocorr and 'recommendations_df' in locals():
        logger.info("\n--- Temporal Modeling Recommendations ---")
        
        needs_temporal = recommendations_df['needs_temporal_model'].sum()
        logger.info(f"Outputs needing temporal modeling: {needs_temporal}/{len(recommendations_df)}")
        
        if needs_temporal > 0:
            logger.info("\nOutputs requiring temporal models:")
            temporal_outputs = recommendations_df[recommendations_df['needs_temporal_model']]
            for _, row in temporal_outputs.iterrows():
                rec_text = str(row['recommendation'])[:80] if pd.notna(row['recommendation']) else 'N/A'
                logger.info(f"  {row['output']}: {rec_text}...")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
