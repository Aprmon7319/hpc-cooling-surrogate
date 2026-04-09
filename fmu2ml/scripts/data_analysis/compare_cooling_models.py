#!/usr/bin/env python3
"""
Comparative Cooling Model Analysis Runner.

This script performs a comprehensive comparison of cooling system behaviors
across different HPC data center configurations. It supports:

1. Multi-system comparison (e.g., Summit, Marconi100, Frontier)
2. Data generation or loading from existing files
3. Comprehensive metrics computation
4. Visualization generation
5. Report creation

Usage:
    python compare_cooling_models.py --systems marconi100 summit frontier
    python compare_cooling_models.py --config path/to/config.yaml
    python compare_cooling_models.py --systems summit frontier --n-samples 1000 --output-dir results/

Examples:
    # Compare two systems with default settings
    python compare_cooling_models.py --systems marconi100 summit
    
    # Compare with custom data generation
    python compare_cooling_models.py --systems summit frontier \\
        --generate-data --n-samples 500 --output-dir my_results/
    
    # Use existing data files
    python compare_cooling_models.py --systems marconi100 summit \\
        --data-files marconi100:data/m100.parquet,summit:data/summit.parquet
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fmu2ml.utils.logging_utils import setup_logging
from fmu2ml.analysis.comparative import (
    CoolingModelComparator,
    SystemProfiler,
    create_comprehensive_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare cooling model behaviors across HPC data center systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # System selection
    parser.add_argument(
        '--systems',
        type=str,
        nargs='+',
        default=['marconi100', 'summit'],
        help='System names to compare (e.g., marconi100 summit frontier)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='fmu2ml/config/defaults/comparative_analysis.yaml',
        help='Path to analysis configuration file'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results (overrides config)'
    )
    
    # Data handling
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate new simulation data for comparison'
    )
    
    parser.add_argument(
        '--data-files',
        type=str,
        default=None,
        help='Comma-separated system:path pairs (e.g., marconi100:data/m100.parquet,summit:data/summit.parquet)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of samples for data generation (overrides config)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to use per system for analysis'
    )
    
    # Input ranges
    parser.add_argument(
        '--input-ranges-qflow',
        nargs=2,
        type=float,
        default=None,
        help='Q_flow input range: min max (kW)'
    )
    
    parser.add_argument(
        '--input-ranges-tair',
        nargs=2,
        type=float,
        default=None,
        help='T_Air input range: min max (K)'
    )
    
    parser.add_argument(
        '--input-ranges-text',
        nargs=2,
        type=float,
        default=None,
        help='T_ext input range: min max (K)'
    )
    
    # Parallelization
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )
    
    # Visualization options
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip creating visualizations'
    )
    
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip creating PDF report'
    )
    
    # Analysis options
    parser.add_argument(
        '--profile-only',
        action='store_true',
        help='Only show system profiles without running comparison'
    )
    
    parser.add_argument(
        '--quick-analysis',
        action='store_true',
        help='Run quick analysis with reduced samples'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def parse_data_files(data_files_str: str) -> dict:
    """Parse data files argument into dictionary."""
    if not data_files_str:
        return {}
    
    data_paths = {}
    for pair in data_files_str.split(','):
        if ':' in pair:
            system, path = pair.split(':', 1)
            data_paths[system.strip()] = path.strip()
    
    return data_paths


def show_system_profiles(systems: list) -> None:
    """Display system profiles for selected systems."""
    print("\n" + "=" * 80)
    print("SYSTEM PROFILES")
    print("=" * 80)
    
    for system_name in systems:
        try:
            profiler = SystemProfiler(system_name)
            print(profiler.summarize())
            print()
        except Exception as e:
            print(f"Error loading profile for {system_name}: {e}")
            print()


def main():
    """Main entry point for comparative analysis."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    systems = args.systems if args.systems else config.get('systems', ['marconi100', 'summit'])
    
    logger.info("=" * 80)
    logger.info("COMPARATIVE COOLING MODEL ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Systems to compare: {', '.join(systems)}")
    
    # Profile-only mode
    if args.profile_only:
        show_system_profiles(systems)
        return 0
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output_dir', 'analysis_results/comparative_analysis'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = output_dir / "comparative_analysis.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Determine number of workers
    n_workers = args.n_workers or config.get('parallel', {}).get('n_workers', 4)
    
    # Initialize comparator
    logger.info("Initializing cooling model comparator...")
    comparator = CoolingModelComparator(
        system_names=systems,
        n_workers=n_workers
    )
    
    # Determine data handling
    data_paths = parse_data_files(args.data_files) if args.data_files else config.get('data_paths', {})
    generate_data = args.generate_data or config.get('data_generation', {}).get('enabled', False)
    
    # Handle data
    if data_paths:
        logger.info("Loading data from provided files...")
        comparator.load_data(data_paths, sample_size=args.max_samples)
    elif generate_data:
        # Determine samples
        if args.quick_analysis:
            n_samples = 100
        elif args.n_samples:
            n_samples = args.n_samples
        else:
            n_samples = config.get('data_generation', {}).get('n_samples', 500)
        
        # Build input ranges
        input_ranges = config.get('data_generation', {}).get('input_ranges', {})
        if args.input_ranges_qflow:
            input_ranges['Q_flow'] = tuple(args.input_ranges_qflow)
        if args.input_ranges_tair:
            input_ranges['T_Air'] = tuple(args.input_ranges_tair)
        if args.input_ranges_text:
            input_ranges['T_ext'] = tuple(args.input_ranges_text)
        
        logger.info(f"Generating {n_samples} samples per system...")
        comparator.generate_data(
            n_samples=n_samples,
            input_ranges=input_ranges if input_ranges else None,
            transition_steps=config.get('data_generation', {}).get('transition_steps', 60),
            stabilization_hours=config.get('data_generation', {}).get('stabilization_hours', 2)
        )
    else:
        logger.error("No data source specified. Use --data-files or --generate-data")
        return 1
    
    # Compute metrics
    logger.info("Computing comparison metrics...")
    time_step = config.get('analysis', {}).get('time_step', 1.0)
    comparator.compute_metrics(time_step=time_step)
    
    # Run comparisons
    logger.info("Running comparative analysis...")
    comparison_results = {
        'efficiency': comparator.compare_efficiency(),
        'thermal': comparator.compare_thermal_performance(),
        'flow': comparator.compare_flow_characteristics(),
        'dynamic': comparator.compare_dynamic_response(),
        'sensitivity': comparator.compare_sensitivity(),
        'normalized': comparator.compute_normalized_comparison()
    }
    
    # Save results
    logger.info("Saving results...")
    comparator.save_results(output_dir, include_data=config.get('advanced', {}).get('save_raw_data', False))
    
    # Generate visualizations
    if not args.skip_visualizations:
        logger.info("Generating visualizations...")
        viz_config = config.get('visualization', {})
        
        report_paths = create_comprehensive_report(
            comparison_results=comparison_results,
            system_metrics=comparator.system_metrics,
            output_dir=output_dir,
            system_data=comparator.system_data if viz_config.get('include_time_series', False) else None,
            create_pdf=not args.no_pdf and viz_config.get('create_pdf_report', True),
            figsize=tuple(viz_config.get('figsize', [14, 10]))
        )
        
        logger.info(f"Created {len(report_paths)} visualization files")
    
    # Generate summary report
    summary = comparator.generate_summary_report()
    print("\n" + summary)
    
    # Save summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
