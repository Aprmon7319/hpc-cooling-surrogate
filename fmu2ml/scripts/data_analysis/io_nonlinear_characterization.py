"""
Script for performing non-linear relationship characterization analysis 
on FMU input-output relationships.

Analyzes:
- Polynomial/spline fitting for non-linear relationship detection
- Threshold and saturation detection using segmented regression
- Operating regime classification
- Model comparison (linear vs. non-linear)

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

from fmu2ml.analysis.nonlinear_characterization.analyzers import (
    NonlinearAnalyzer,
    ThresholdDetector
)
from fmu2ml.analysis.nonlinear_characterization.visualizers import (
    create_nonlinearity_plots,
    create_model_comparison_plots,
    create_residual_plots,
    create_threshold_plots,
    create_regime_scatter_plots,
    create_nonlinearity_strength_chart
)
from fmu2ml.analysis.nonlinear_characterization.visualizers.residual_visualizer import (
    create_residual_summary_plot
)
from fmu2ml.analysis.nonlinear_characterization.visualizers.threshold_visualizer import (
    create_threshold_summary_plot
)
from fmu2ml.analysis.nonlinear_characterization.visualizers.regime_visualizer import (
    create_regime_summary_plot
)
from fmu2ml.analysis.nonlinear_characterization.visualizers.strength_chart_visualizer import (
    create_neural_network_recommendations
)
from fmu2ml.analysis.input_output_relations.analyzers import DataGenerator
from fmu2ml.utils.logging_utils import setup_logging
from raps.config import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform non-linear relationship characterization on FMU inputs and outputs'
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
        default='fmu2ml/config/defaults/nonlinear_characterization.yaml',
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
        '--max-polynomial-degree',
        type=int,
        default=None,
        help='Maximum polynomial degree to test (default: 5)'
    )
    
    parser.add_argument(
        '--max-breakpoints',
        type=int,
        default=None,
        help='Maximum breakpoints for segmented regression (default: 3)'
    )
    
    parser.add_argument(
        '--skip-polynomial',
        action='store_true',
        help='Skip polynomial fitting analysis'
    )
    
    parser.add_argument(
        '--skip-threshold',
        action='store_true',
        help='Skip threshold/saturation detection'
    )
    
    parser.add_argument(
        '--skip-regime',
        action='store_true',
        help='Skip operating regime classification'
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
    
    if args.max_polynomial_degree:
        config['max_polynomial_degree'] = args.max_polynomial_degree
    else:
        config['max_polynomial_degree'] = config.get('max_polynomial_degree', 5)
    
    if args.max_breakpoints:
        config['max_breakpoints'] = args.max_breakpoints
    else:
        config['max_breakpoints'] = config.get('max_breakpoints', 3)
    
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
        output_dir = f'analysis_results/nonlinear_characterization_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir=output_dir)
    logger = logging.getLogger(__name__)
    
    num_cdus = config.get('NUM_CDUS', config.get('num_cdus', 49))
    n_samples = config['n_samples']
    input_ranges = config['input_ranges']
    
    logger.info("=" * 80)
    logger.info("FMU Non-Linear Relationship Characterization Analysis")
    logger.info("=" * 80)
    logger.info(f"System: {args.system_name}")
    logger.info(f"Number of CDUs: {num_cdus}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Max polynomial degree: {config['max_polynomial_degree']}")
    logger.info(f"Max breakpoints: {config['max_breakpoints']}")
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
    
    # Initialize analyzers
    config_for_analyzer = {k: v for k, v in config.items() 
                          if k not in ['system_name', 'max_polynomial_degree', 'max_breakpoints']}
    
    nonlinear_analyzer = NonlinearAnalyzer(
        system_name=args.system_name,
        n_workers=config['parallel']['n_workers'],
        threads_per_worker=config['parallel']['threads_per_worker'],
        memory_limit=config['parallel']['memory_limit'],
        max_polynomial_degree=config['max_polynomial_degree'],
        **config_for_analyzer
    )
    
    threshold_detector = ThresholdDetector(
        system_name=args.system_name,
        n_workers=config['parallel']['n_workers'],
        threads_per_worker=config['parallel']['threads_per_worker'],
        memory_limit=config['parallel']['memory_limit'],
        max_breakpoints=config['max_breakpoints'],
        **config_for_analyzer
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
    
    # Prepare data
    logger.info("=" * 80)
    logger.info("STEP 2: Preparing Data")
    logger.info("=" * 80)
    prepared_data = nonlinear_analyzer.prepare_data(data)
    
    step_num = 2
    
    # =====================================================================
    # POLYNOMIAL/SPLINE FITTING ANALYSIS
    # =====================================================================
    if not args.skip_polynomial:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Polynomial/Spline Fitting Analysis")
        logger.info("=" * 80)
        
        # Polynomial fits
        logger.info("Computing polynomial fits...")
        polynomial_df, polynomial_full_results = nonlinear_analyzer.analyze_polynomial_fits(prepared_data)
        
        polynomial_file = os.path.join(output_dir, 'polynomial_fits.csv')
        polynomial_df.to_csv(polynomial_file, index=False)
        logger.info(f"Saved polynomial fits to {polynomial_file}")
        
        # Model comparison
        logger.info("Computing model comparisons...")
        comparison_df = nonlinear_analyzer.compute_model_comparison(prepared_data)
        
        comparison_file = os.path.join(output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Saved model comparison to {comparison_file}")
        
        # Get prediction data for visualizations
        logger.info("Generating prediction data for visualizations...")
        prediction_data_dict = {}
        
        all_outputs = list(prepared_data['outputs'].keys())
        all_outputs.extend(prepared_data.get('datacenter_outputs', {}).keys())
        
        for input_name in nonlinear_analyzer.input_vars:
            for output_name in all_outputs:
                pred_data = nonlinear_analyzer.get_prediction_data(
                    prepared_data, input_name, output_name,
                    degrees=[1, 2, 3]
                )
                if pred_data:
                    prediction_data_dict[(input_name, output_name)] = pred_data
        
        # Create visualizations
        logger.info("Creating polynomial fit visualizations...")
        
        viz_dir = os.path.join(output_dir, 'visualizations', 'polynomial')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Scatter plots with multiple fits
        create_nonlinearity_plots(
            prediction_data_dict,
            viz_dir,
            n_workers=config['parallel']['n_workers']
        )
        
        # Model comparison plots
        create_model_comparison_plots(
            comparison_df,
            polynomial_df,
            viz_dir
        )
        
        # Residual plots
        residual_dir = os.path.join(output_dir, 'visualizations', 'residuals')
        os.makedirs(residual_dir, exist_ok=True)
        
        create_residual_plots(
            prediction_data_dict,
            residual_dir,
            n_workers=config['parallel']['n_workers']
        )
        
        create_residual_summary_plot(prediction_data_dict, residual_dir)
        
        # Non-linearity strength chart
        strength_dir = os.path.join(output_dir, 'visualizations', 'nonlinearity_strength')
        os.makedirs(strength_dir, exist_ok=True)
        
        create_nonlinearity_strength_chart(comparison_df, strength_dir)
        create_neural_network_recommendations(comparison_df, strength_dir)
        
        logger.info("Polynomial fitting analysis complete")
    
    # =====================================================================
    # THRESHOLD/SATURATION DETECTION
    # =====================================================================
    if not args.skip_threshold:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Threshold/Saturation Detection")
        logger.info("=" * 80)
        
        # Detect thresholds
        logger.info("Detecting thresholds and breakpoints...")
        threshold_df, threshold_full_results = threshold_detector.detect_thresholds_all_pairs(prepared_data)
        
        threshold_file = os.path.join(output_dir, 'threshold_detection.csv')
        threshold_df.to_csv(threshold_file, index=False)
        logger.info(f"Saved threshold detection to {threshold_file}")
        
        # Get segmented fit data for visualizations
        logger.info("Generating segmented fit data for visualizations...")
        fit_data_dict = {}
        
        all_outputs = list(prepared_data['outputs'].keys())
        all_outputs.extend(prepared_data.get('datacenter_outputs', {}).keys())
        
        for input_name in threshold_detector.input_vars:
            for output_name in all_outputs:
                fit_data = threshold_detector.get_segmented_fit_data(
                    prepared_data, input_name, output_name
                )
                if fit_data:
                    fit_data_dict[(input_name, output_name)] = fit_data
        
        # Create visualizations
        logger.info("Creating threshold visualizations...")
        
        threshold_viz_dir = os.path.join(output_dir, 'visualizations', 'thresholds')
        os.makedirs(threshold_viz_dir, exist_ok=True)
        
        create_threshold_plots(
            fit_data_dict,
            threshold_viz_dir,
            n_workers=config['parallel']['n_workers']
        )
        
        create_threshold_summary_plot(threshold_df, threshold_viz_dir)
        
        logger.info("Threshold detection complete")
    
    # =====================================================================
    # OPERATING REGIME CLASSIFICATION
    # =====================================================================
    if not args.skip_regime:
        step_num += 1
        logger.info("=" * 80)
        logger.info(f"STEP {step_num}: Operating Regime Classification")
        logger.info("=" * 80)
        
        # Classify regimes
        logger.info("Classifying operating regimes...")
        regime_df, regime_full_results = threshold_detector.classify_regimes_all_pairs(prepared_data)
        
        regime_file = os.path.join(output_dir, 'regime_classification.csv')
        regime_df.to_csv(regime_file, index=False)
        logger.info(f"Saved regime classification to {regime_file}")
        
        # Create visualizations
        logger.info("Creating regime visualizations...")
        
        regime_viz_dir = os.path.join(output_dir, 'visualizations', 'regimes')
        os.makedirs(regime_viz_dir, exist_ok=True)
        
        create_regime_scatter_plots(
            regime_full_results,
            regime_viz_dir,
            n_workers=config['parallel']['n_workers']
        )
        
        create_regime_summary_plot(regime_df, regime_viz_dir)
        
        logger.info("Regime classification complete")
    
    # =====================================================================
    # CLEANUP AND SUMMARY
    # =====================================================================
    nonlinear_analyzer._close_dask_client()
    threshold_detector._close_dask_client()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    if not args.skip_polynomial and 'comparison_df' in locals():
        logger.info("\n--- Non-linearity Analysis ---")
        
        strongly_nonlinear = comparison_df[comparison_df['r2_improvement_pct'] > 5]
        logger.info(f"Strongly non-linear relationships (>5% R² improvement): {len(strongly_nonlinear)}")
        
        if not strongly_nonlinear.empty:
            logger.info("\nTop 5 most non-linear relationships:")
            top_nonlinear = comparison_df.nlargest(5, 'r2_improvement_pct')
            for _, row in top_nonlinear.iterrows():
                logger.info(
                    f"  {row['input']} → {row['output']}: "
                    f"{row['r2_improvement_pct']:.1f}% improvement, "
                    f"optimal degree: {row['best_degree']}"
                )
        
        # Recommendations
        mean_improvement = comparison_df['r2_improvement_pct'].mean()
        logger.info(f"\nAverage R² improvement from non-linear models: {mean_improvement:.1f}%")
        
        if mean_improvement > 5:
            logger.info("RECOMMENDATION: Non-linear activations are strongly recommended for neural operators")
        elif mean_improvement > 2:
            logger.info("RECOMMENDATION: Consider non-linear activations for key relationships")
        else:
            logger.info("RECOMMENDATION: Linear models may be sufficient for most relationships")
    
    if not args.skip_threshold and 'threshold_df' in locals():
        logger.info("\n--- Threshold/Saturation Detection ---")
        
        has_threshold = threshold_df['has_threshold'].sum()
        has_saturation = threshold_df['has_saturation'].sum()
        
        logger.info(f"Relationships with detected thresholds: {has_threshold}")
        logger.info(f"Relationships with saturation behavior: {has_saturation}")
        
        if has_threshold > 0:
            logger.info("\nPairs with threshold behavior:")
            threshold_pairs = threshold_df[threshold_df['has_threshold']]
            for _, row in threshold_pairs.iterrows():
                logger.info(f"  {row['input']} → {row['output']}")
    
    if not args.skip_regime and 'regime_df' in locals():
        logger.info("\n--- Operating Regime Analysis ---")
        
        mean_regimes = regime_df['n_regimes'].mean()
        logger.info(f"Average number of operating regimes: {mean_regimes:.1f}")
        
        multi_regime = regime_df[regime_df['n_regimes'] > 2]
        if not multi_regime.empty:
            logger.info(f"\nRelationships with 3+ regimes: {len(multi_regime)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
