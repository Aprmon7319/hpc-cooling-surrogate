"""
FMU Simulation CLI

Runs FMU simulations to generate training data.

Usage:
    python -m fmu2ml.scripts.run_simulation --system marconi100 --input data/input.parquet --output data/fmu_output/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from fmu2ml.config import get_system_config
from fmu2ml.simulation import FMUSimulator, ParallelFMURunner
from fmu2ml.utils.logging_utils import setup_logging


def run_simulation(args):
    """Run FMU simulation"""
    
    logger = setup_logging('simulate')
    
    logger.info("=" * 80)
    logger.info("FMU2ML FMU Simulation")
    logger.info("=" * 80)
    logger.info(f"System: {args.system}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Parallel chunks: {args.chunks}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load system config
    logger.info("Loading system configuration...")
    config = get_system_config(args.system, use_raps=True)
    
    # Load input data
    logger.info("Loading input data...")
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single input file
        if input_path.suffix == '.parquet':
            input_data = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            input_data = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"  Loaded {len(input_data)} samples")
        input_files = [input_data]
    else:
        # Directory with multiple files
        input_files = []
        for f in sorted(input_path.glob('*.parquet')):
            df = pd.read_parquet(f)
            input_files.append(df)
            logger.info(f"  Loaded {f.name}: {len(df)} samples")
    
    # Run simulation
    if args.chunks == 1 or len(input_files) == 1:
        # Single process
        logger.info("Running simulation (single process)...")
        simulator = FMUSimulator(config)
        
        for idx, input_df in enumerate(input_files):
            logger.info(f"Processing chunk {idx+1}/{len(input_files)}...")
            output_df = simulator.run(input_df)
            
            # Save output
            output_path = output_dir / f'fmu_output_chunk_{idx}.parquet'
            output_df.to_parquet(output_path)
            logger.info(f"  Saved: {output_path}")
    
    else:
        # Parallel processing
        logger.info(f"Running simulation ({args.chunks} parallel processes)...")
        runner = ParallelFMURunner(
            config=config,
            num_processes=args.chunks
        )
        
        outputs = runner.run_parallel(input_files)
        
        # Save outputs
        for idx, output_df in enumerate(outputs):
            output_path = output_dir / f'fmu_output_chunk_{idx}.parquet'
            output_df.to_parquet(output_path)
            logger.info(f"  Saved: {output_path}")
    
    logger.info("=" * 80)
    logger.info("Simulation complete!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run FMU simulations to generate training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file simulation
  python -m fmu2ml.scripts.run_simulation --system marconi100 \\
      --input data/input.parquet --output data/fmu_output/

  # Parallel simulation
  python -m fmu2ml.scripts.run_simulation --system summit \\
      --input data/input_dir/ --output data/fmu_output/ --chunks 4
        """
    )
    
    # Required arguments
    parser.add_argument('--system', type=str, required=True,
                        choices=['marconi100', 'summit', 'frontier', 'fugaku', 'lassen'],
                        help='HPC system name')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    
    # Optional arguments
    parser.add_argument('--chunks', type=int, default=1,
                        help='Number of parallel processes (default: 1)')
    parser.add_argument('--use-raps', action='store_true', default=True,
                        help='Use RAPS configuration (default: True)')
    
    args = parser.parse_args()
    
    # Run simulation
    run_simulation(args)


if __name__ == '__main__':
    main()