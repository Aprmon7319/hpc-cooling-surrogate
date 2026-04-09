#!/usr/bin/env python3
"""
Model Comparison CLI

Compares multiple trained models and FMU on test data.

Usage:
    python -m fmu2ml.scripts.compare_models --models fno.pt hybrid_fno.pt --fmu --data data/ --output comparison/
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from fmu2ml.config import get_system_config
from fmu2ml.inference import CoolingModelInference
from fmu2ml.evaluation import ModelComparator
from fmu2ml.simulation import FMUWrapper
from fmu2ml.data.processors import create_data_loaders
from fmu2ml.utils.logging_utils import setup_logger
from fmu2ml.visualization import create_comparison_plots


def compare_models(args):
    """Compare multiple models"""
    
    logger = setup_logger('compare')
    
    logger.info("=" * 80)
    logger.info("FMU2ML Model Comparison")
    logger.info("=" * 80)
    logger.info(f"Models: {args.models}")
    logger.info(f"Include FMU: {args.fmu}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    predictors = {}
    for model_path in args.models:
        model_name = Path(model_path).stem
        predictors[model_name] = CoolingModelInference(
            model_path=model_path,
            model_type='auto'  # Auto-detect from checkpoint
        )
        logger.info(f"  Loaded: {model_name}")
    
    # Load FMU if requested
    fmu = None
    if args.fmu:
        logger.info("Loading FMU...")
        config = get_system_config(args.system, use_raps=True)
        fmu = FMUWrapper(config)
    
    # Create data loader
    logger.info("Loading test data...")
    _, _, test_loader = create_data_loaders(
        data_path=args.data,
        test_chunks=args.test_chunks,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=False
    )
    
    # Initialize comparator
    comparator = ModelComparator(
        models=predictors,
        fmu=fmu,
        test_loader=test_loader,
        output_dir=output_dir
    )
    
    # Run comparison
    logger.info("Running comparison...")
    results = comparator.compare_all()
    
    # Save results
    logger.info("Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'comparison_results.csv', index=False)
    
    # Generate comparison plots
    if args.visualize:
        logger.info("Generating comparison plots...")
        create_comparison_plots(
            results=results,
            output_dir=output_dir / 'plots'
        )
    
    # Print summary
    logger.info("\nComparison Summary:")
    logger.info("-" * 80)
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
    
    logger.info("=" * 80)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple trained models and FMU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two ML models
  python -m fmu2ml.scripts.compare_models --models fno.pt hybrid_fno.pt --data data/ --output comparison/

  # Compare ML models with FMU
  python -m fmu2ml.scripts.compare_models --models fno.pt hybrid_fno.pt \\
      --fmu --system marconi100 --data data/ --output comparison/
        """
    )
    
    # Required arguments
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory containing parquet files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--fmu', action='store_true',
                        help='Include FMU in comparison')
    parser.add_argument('--system', type=str, default='marconi100',
                        choices=['marconi100', 'summit', 'frontier', 'fugaku', 'lassen'],
                        help='HPC system name (for FMU, default: marconi100)')
    parser.add_argument('--test-chunks', type=int, nargs='+', default=[10],
                        help='Test chunk IDs (default: [10])')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate comparison plots')
    
    args = parser.parse_args()
    
    # Run comparison
    compare_models(args)


if __name__ == '__main__':
    main()