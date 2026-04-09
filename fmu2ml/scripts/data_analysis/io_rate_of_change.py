"""
Script for performing Rate of Change (Dynamic Effects) Analysis.
Analyzes temporal derivatives and impulse responses of FMU inputs/outputs.

This script performs:
1. Temporal Derivative Analysis - comparing level vs rate effects
2. Impulse Response Analysis - characterizing system dynamics
3. Lag Correlation Analysis - determining response time delays
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
from fmu2ml.utils.logging_utils import setup_logging

from fmu2ml.analysis.rate_of_change.analyzers import (
    RateOfChangeAnalyzer,
    ImpulseResponseAnalyzer,
    DynamicDataGenerator
)
from fmu2ml.analysis.rate_of_change.visualizers import (
    create_rate_comparison_plots,
    create_impulse_response_plots,
    create_time_series_overlay_plots,
    create_response_characteristics_heatmap,
    create_level_rate_correlation_bars
)
from fmu2ml.analysis.rate_of_change.visualizers.rate_comparison_visualizer import create_rate_summary_plot
from fmu2ml.analysis.rate_of_change.visualizers.impulse_response_visualizer import create_averaged_impulse_plots
from fmu2ml.analysis.rate_of_change.visualizers.time_series_visualizer import create_lag_correlation_plot
from fmu2ml.analysis.rate_of_change.visualizers.response_heatmap_visualizer import create_dynamics_summary_heatmap
from raps.config import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform rate of change (dynamic effects) analysis on FMU inputs and outputs'
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
        default='fmu2ml/config/defaults/rate_of_change.yaml',
        help='Path to analysis config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results (overrides config)'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate new simulation data with dynamic patterns'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to existing data file'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['step', 'ramp', 'varying', 'combined'],
        default='combined',
        help='Type of dynamic data to generate'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of samples for data generation (overrides config)'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=None,
        help='Number of step changes for impulse data'
    )
    
    # Analysis flags
    parser.add_argument(
        '--level-rate-only',
        action='store_true',
        help='Only perform level vs rate analysis'
    )
    
    parser.add_argument(
        '--impulse-only',
        action='store_true',
        help='Only perform impulse response analysis'
    )
    
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip creating visualizations'
    )
    
    # Input ranges
    parser.add_argument(
        '--input-ranges-qflow',
        nargs=2,
        type=float,
        default=None,
        help='Q_flow input range: min max'
    )
    
    parser.add_argument(
        '--input-ranges-tair',
        nargs=2,
        type=float,
        default=None,
        help='T_Air input range: min max'
    )
    
    parser.add_argument(
        '--input-ranges-text',
        nargs=2,
        type=float,
        default=None,
        help='T_ext input range: min max'
    )
    
    # Parallel settings
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--threads-per-worker',
        type=int,
        default=None,
        help='Threads per Dask worker'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=str,
        default=None,
        help='Memory limit per worker'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--max-lag',
        type=int,
        default=30,
        help='Maximum lag for correlation analysis'
    )
    
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=5,
        help='Smoothing window for derivative computation'
    )
    
    return parser.parse_args()


def load_configs(args):
    """Load and merge configuration files."""
    # Load system configuration
    system_config = ConfigManager(system_name=args.system_name).get_config()
    
    # Load analysis config
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
        config['n_samples'] = config.get('n_samples', 2000)
    
    if args.n_steps:
        config['n_steps'] = args.n_steps
    else:
        config['n_steps'] = config.get('n_steps', 20)
    
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
    config['max_lag'] = args.max_lag
    config['smooth_window'] = args.smooth_window
    config['data_type'] = args.data_type
    
    # Derivative settings
    if 'derivative' not in config:
        config['derivative'] = {}
    config['derivative'].setdefault('dt', 1.0)
    config['derivative']['smooth_window'] = args.smooth_window
    
    # Impulse detection settings
    if 'impulse_detection' not in config:
        config['impulse_detection'] = {}
    config['impulse_detection'].setdefault('threshold_percentile', 90)
    config['impulse_detection'].setdefault('min_change_fraction', 0.1)
    config['impulse_detection'].setdefault('min_duration', 10)
    
    # Response analysis settings
    if 'response_analysis' not in config:
        config['response_analysis'] = {}
    config['response_analysis'].setdefault('pre_window', 30)
    config['response_analysis'].setdefault('post_window', 60)
    
    return config


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config = load_configs(args)
    
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.get('output_path', 'results/rate_of_change')
    output_dir = os.path.join(output_dir, f'rate_of_change_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir=output_dir)
    logger = logging.getLogger(__name__)
    
    num_cdus = config.get('NUM_CDUS', config.get('num_cdus', 49))
    n_samples = config['n_samples']
    input_ranges = config['input_ranges']
    
    logger.info("=" * 80)
    logger.info("FMU Rate of Change (Dynamic Effects) Analysis")
    logger.info("=" * 80)
    logger.info(f"System: {args.system_name}")
    logger.info(f"Number of CDUs: {num_cdus}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Data type: {config['data_type']}")
    logger.info(f"Input ranges:")
    logger.info(f"  Q_flow: {input_ranges['Q_flow']} kW")
    logger.info(f"  T_Air: {input_ranges['T_Air']} K")
    logger.info(f"  T_ext: {input_ranges['T_ext']} K")
    logger.info(f"Parallel settings:")
    logger.info(f"  Workers: {config['parallel']['n_workers']}")
    logger.info(f"  Threads/worker: {config['parallel']['threads_per_worker']}")
    logger.info(f"  Memory limit: {config['parallel']['memory_limit']}")
    
    # Remove system_name from config to avoid duplicate
    config_for_components = {k: v for k, v in config.items() if k != 'system_name'}
    
    # Initialize analyzers
    rate_analyzer = RateOfChangeAnalyzer(
        system_name=args.system_name,
        n_workers=config['parallel']['n_workers'],
        threads_per_worker=config['parallel']['threads_per_worker'],
        memory_limit=config['parallel']['memory_limit'],
        **config_for_components
    )
    
    impulse_analyzer = ImpulseResponseAnalyzer(
        system_name=args.system_name,
        n_workers=config['parallel']['n_workers'],
        threads_per_worker=config['parallel']['threads_per_worker'],
        memory_limit=config['parallel']['memory_limit'],
        **config_for_components
    )
    
    # Generate or load data
    if args.generate_data:
        logger.info("=" * 80)
        logger.info("STEP 1: Generating Dynamic Simulation Data")
        logger.info("=" * 80)
        
        data_generator = DynamicDataGenerator(
            system_name=args.system_name,
            n_workers=config['parallel']['n_workers'],
            **config_for_components
        )
        
        if config['data_type'] == 'step':
            data = data_generator.generate_step_response_data(
                n_steps=config['n_steps'],
                step_duration=config.get('step_duration', 120),
                input_ranges=input_ranges
            )
        elif config['data_type'] == 'ramp':
            data = data_generator.generate_ramp_data(
                n_ramps=config['n_steps'],
                ramp_duration=config.get('ramp_duration', 60),
                hold_duration=config.get('hold_duration', 60),
                input_ranges=input_ranges
            )
        elif config['data_type'] == 'varying':
            data = data_generator.generate_varying_rate_data(
                n_samples=n_samples,
                input_ranges=input_ranges
            )
        else:  # combined
            data = data_generator.generate_combined_dynamic_data(
                n_samples=n_samples,
                input_ranges=input_ranges
            )
        
        # Save data
        data_file = os.path.join(output_dir, 'dynamic_data.parquet')
        data.to_parquet(data_file)
        logger.info(f"Saved data to {data_file}")
        
    elif config.get('data_path') and os.path.exists(config['data_path']):
        logger.info(f"Loading data from {config['data_path']}")
        data = pd.read_parquet(config['data_path'])
    else:
        logger.error("Must specify --generate-data or provide valid data_path in config")
        return
    
    logger.info(f"Data shape: {data.shape}")
    
    # ========================================
    # STEP 2: Prepare Data
    # ========================================
    logger.info("=" * 80)
    logger.info("STEP 2: Preparing Data")
    logger.info("=" * 80)
    
    prepared_data = rate_analyzer.prepare_data(data)
    
    # ========================================
    # STEP 3: Compute Derivatives
    # ========================================
    logger.info("=" * 80)
    logger.info("STEP 3: Computing Temporal Derivatives")
    logger.info("=" * 80)
    
    derivatives_data = rate_analyzer.compute_derivatives(
        prepared_data,
        dt=config['derivative']['dt'],
        smooth_window=config['derivative']['smooth_window']
    )
    
    # ========================================
    # STEP 4: Level vs Rate Analysis
    # ========================================
    if not args.impulse_only:
        logger.info("=" * 80)
        logger.info("STEP 4: Level vs Rate Effect Analysis")
        logger.info("=" * 80)
        
        level_rate_df = rate_analyzer.analyze_all_level_rate_effects(derivatives_data)
        
        # Save results
        level_rate_file = os.path.join(output_dir, 'level_rate_metrics.csv')
        level_rate_df.to_csv(level_rate_file, index=False)
        logger.info(f"Saved level-rate metrics to {level_rate_file}")
        
        # Log summary
        n_level_dominant = (level_rate_df['dominant_effect'] == 'level').sum()
        n_rate_dominant = (level_rate_df['dominant_effect'] == 'rate').sum()
        n_mixed = (level_rate_df['dominant_effect'] == 'mixed').sum()
        
        logger.info(f"Effect dominance: Level={n_level_dominant}, Rate={n_rate_dominant}, Mixed={n_mixed}")
    else:
        level_rate_df = pd.DataFrame()
    
    # ========================================
    # STEP 5: Lag Correlation Analysis
    # ========================================
    if not args.impulse_only:
        logger.info("=" * 80)
        logger.info("STEP 5: Response Lag Analysis")
        logger.info("=" * 80)
        
        lag_df = rate_analyzer.analyze_response_lags(
            prepared_data,
            max_lag=config['max_lag']
        )
        
        # Save results
        lag_file = os.path.join(output_dir, 'lag_analysis.csv')
        lag_df.to_csv(lag_file, index=False)
        logger.info(f"Saved lag analysis to {lag_file}")
        
        # Get detailed lag correlation data
        lag_correlation_data = rate_analyzer.get_lag_correlation_data()
    else:
        lag_df = pd.DataFrame()
        lag_correlation_data = []
    
    # ========================================
    # STEP 6: Impulse Response Analysis
    # ========================================
    if not args.level_rate_only:
        logger.info("=" * 80)
        logger.info("STEP 6: Impulse Response Analysis")
        logger.info("=" * 80)
        
        # Prepare data for impulse analyzer
        impulse_prepared = impulse_analyzer.prepare_data(data)
        
        # Detect step changes
        step_events = impulse_analyzer.detect_step_changes(
            impulse_prepared,
            threshold_percentile=config['impulse_detection']['threshold_percentile'],
            min_change_fraction=config['impulse_detection']['min_change_fraction'],
            min_duration=config['impulse_detection']['min_duration']
        )
        
        # Analyze responses
        response_df = impulse_analyzer.analyze_all_responses(
            impulse_prepared,
            step_events,
            pre_window=config['response_analysis']['pre_window'],
            post_window=config['response_analysis']['post_window']
        )
        
        # Save results
        response_file = os.path.join(output_dir, 'impulse_responses.csv')
        response_df.to_csv(response_file, index=False)
        logger.info(f"Saved impulse responses to {response_file}")
        
        # Create summary
        response_summary_df = impulse_analyzer.summarize_response_characteristics(response_df)
        summary_file = os.path.join(output_dir, 'response_summary.csv')
        response_summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved response summary to {summary_file}")
        
        # Get response curves for visualization
        response_curves = impulse_analyzer.get_response_curves()
        
    else:
        step_events = {}
        response_df = pd.DataFrame()
        response_summary_df = pd.DataFrame()
        response_curves = []
    
    # ========================================
    # STEP 7: Create Combined Summary
    # ========================================
    if not level_rate_df.empty and not lag_df.empty:
        logger.info("=" * 80)
        logger.info("STEP 7: Creating Combined Dynamic Effects Summary")
        logger.info("=" * 80)
        
        dynamics_summary = rate_analyzer.summarize_dynamic_effects(level_rate_df, lag_df)
        
        summary_file = os.path.join(output_dir, 'dynamics_summary.csv')
        dynamics_summary.to_csv(summary_file, index=False)
        logger.info(f"Saved dynamics summary to {summary_file}")
    else:
        dynamics_summary = pd.DataFrame()
    
    # ========================================
    # STEP 8: Create Visualizations
    # ========================================
    if not args.skip_visualizations:
        logger.info("=" * 80)
        logger.info("STEP 8: Creating Visualizations")
        logger.info("=" * 80)
        
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Level vs Rate comparison plots
        if not level_rate_df.empty:
            logger.info("Creating level vs rate comparison plots...")
            create_rate_comparison_plots(
                derivatives_data,
                level_rate_df,
                viz_dir,
                n_workers=config['parallel']['n_workers']
            )
            
            create_rate_summary_plot(level_rate_df, viz_dir)
            create_level_rate_correlation_bars(level_rate_df, viz_dir)
        
        # Time series overlay plots
        if step_events:
            logger.info("Creating time series overlay plots...")
            create_time_series_overlay_plots(
                prepared_data,
                derivatives_data,
                step_events,
                viz_dir
            )
        
        # Lag correlation plots
        if lag_correlation_data:
            logger.info("Creating lag correlation plots...")
            create_lag_correlation_plot(lag_correlation_data, viz_dir)
        
        # Impulse response plots
        if not response_df.empty:
            logger.info("Creating impulse response plots...")
            create_impulse_response_plots(
                response_curves,
                response_df,
                viz_dir,
                n_workers=config['parallel']['n_workers'],
                max_plots=config.get('visualization', {}).get('max_impulse_plots', 50)
            )
            
            create_averaged_impulse_plots(response_curves, response_df, viz_dir)
        
        # Response characteristics heatmap
        if not response_summary_df.empty:
            logger.info("Creating response characteristics heatmap...")
            create_response_characteristics_heatmap(response_summary_df, viz_dir)
        
        # Dynamics summary heatmap
        if not dynamics_summary.empty:
            logger.info("Creating dynamics summary heatmap...")
            create_dynamics_summary_heatmap(dynamics_summary, viz_dir)
    
    # Close Dask clients
    rate_analyzer._close_dask_client()
    impulse_analyzer._close_dask_client()
    
    # ========================================
    # Print Summary
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    if not level_rate_df.empty:
        logger.info("\nLevel vs Rate Effect Analysis:")
        logger.info("-" * 40)
        
        for input_name in ['Q_flow', 'T_Air', 'T_ext']:
            input_data = level_rate_df[level_rate_df['input'] == input_name]
            if not input_data.empty:
                n_level = (input_data['dominant_effect'] == 'level').sum()
                n_rate = (input_data['dominant_effect'] == 'rate').sum()
                n_mixed = (input_data['dominant_effect'] == 'mixed').sum()
                avg_level_r = input_data['level_strength'].mean()
                avg_rate_r = input_data['rate_strength'].mean()
                
                logger.info(f"\n  {input_name}:")
                logger.info(f"    Dominant effects: Level={n_level}, Rate={n_rate}, Mixed={n_mixed}")
                logger.info(f"    Avg strength: Level |r|={avg_level_r:.3f}, Rate |r|={avg_rate_r:.3f}")
    
    if not lag_df.empty:
        logger.info("\nResponse Lag Analysis:")
        logger.info("-" * 40)
        
        for input_name in ['Q_flow', 'T_Air', 'T_ext']:
            input_lags = lag_df[lag_df['input'] == input_name]
            if not input_lags.empty:
                avg_lag = input_lags['optimal_lag'].mean()
                max_corr = input_lags['max_correlation'].mean()
                logger.info(f"  {input_name}: Avg optimal lag = {avg_lag:.1f} steps, Avg max corr = {max_corr:.3f}")
    
    if not response_summary_df.empty:
        logger.info("\nImpulse Response Characteristics:")
        logger.info("-" * 40)
        
        for input_name in response_summary_df['input'].unique():
            input_summary = response_summary_df[response_summary_df['input'] == input_name]
            avg_rise = input_summary['rise_time_mean'].mean()
            avg_gain = input_summary['gain_mean'].mean()
            avg_tau = input_summary['time_constant_mean'].mean()
            
            logger.info(f"\n  {input_name}:")
            logger.info(f"    Avg rise time: {avg_rise:.1f} steps")
            logger.info(f"    Avg gain: {avg_gain:.4f}")
            logger.info(f"    Avg time constant: {avg_tau:.1f} steps")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
