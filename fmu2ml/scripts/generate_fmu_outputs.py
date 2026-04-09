import argparse
import logging
from pathlib import Path
from fmu2ml.simulation import BatchFMUOutputGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate FMU outputs from input files"
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_input_*.parquet',
        help='File pattern to match (default: *_input_*.parquet)'
    )
    
    parser.add_argument(
        '--system',
        type=str,
        default='marconi100',
        help='System configuration name'
    )
    
    parser.add_argument(
        '--stabilization-hours',
        type=int,
        default=2,
        help='Maximum stabilization hours'
    )
    
    parser.add_argument(
        '--step-size',
        type=int,
        default=1,
        help='Simulation step size in seconds'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Find input files
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob(args.pattern))
    
    if not input_files:
        logger.error(f"No input files found matching pattern: {args.pattern}")
        return 1
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Create batch generator
    generator = BatchFMUOutputGenerator(
        system_name=args.system,
        num_workers=args.workers,
        stabilization_hours=args.stabilization_hours,
        step_size=args.step_size
    )
    
    # Generate outputs
    results = generator.generate_batch(
        [str(f) for f in input_files],
        args.output_dir,
        parallel=args.parallel
    )
    
    # Report results
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"Batch complete: {successful} successful, {failed} failed")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    exit(main())