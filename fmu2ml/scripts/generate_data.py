"""
Data Generation CLI

Generates FMU input data (power scenarios and temperatures) for training.

Usage:
    python -m fmu2ml.scripts.generate_data --system marconi100 --duration 24 --output data/
    
    # Parallel generation
    python -m fmu2ml.scripts.generate_data --system summit --duration 168 --chunks 4 --output data/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

from fmu2ml.config import get_system_config
from fmu2ml.data.generators import ScenarioGenerator, PowerGenerator
from fmu2ml.data.generators.temperature_generator import generate_temperature_dataset
from fmu2ml.utils.logging_utils import setup_logging


def generate_chunk(args):
    """Generate a single data chunk"""
    chunk_id, duration_per_chunk, start_offset_hours, base_date, system_name, output_dir, seed = args
    
    logger = setup_logging(f'generate_chunk_{chunk_id}')
    logger.info(f"Starting chunk {chunk_id}")
    
    try:
        # Get system config
        config = get_system_config(system_name, use_raps=True)
        
        # Calculate start date for this chunk
        from datetime import timedelta
        start_dt = datetime.fromisoformat(base_date.replace('Z', '+00:00'))
        start_dt += timedelta(hours=start_offset_hours)
        start_date_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Generate power scenarios
        scenario_gen = ScenarioGenerator(
            num_cdus=config.num_cdus,
            min_power=config.min_power,
            max_power=config.max_power,
            seed=seed + chunk_id
        )
        
        scenarios = scenario_gen.generate_scenarios()
        
        # Generate continuous power data
        power_gen = PowerGenerator(
            num_cdus=config.num_cdus,
            min_power=config.min_power,
            max_power=config.max_power,
            seed=seed + chunk_id
        )
        
        power_df = power_gen.generate_continuous_power(
            scenarios=scenarios,
            duration_hours=duration_per_chunk,
            timestep_seconds=60
        )
        
        # Generate temperatures
        temp_df = generate_temperature_dataset(
            power_df=power_df,
            start_date=start_date_str,
            timestep_seconds=60,
            seed=seed + chunk_id,
            output_dir=output_dir,
            output_format='parquet',
            save_output=True
        )
        
        logger.info(f"Chunk {chunk_id} completed: {len(temp_df)} samples")
        return chunk_id, len(temp_df)
        
    except Exception as e:
        logger.error(f"Chunk {chunk_id} failed: {e}")
        return chunk_id, 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate FMU input data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 24 hours of data for Marconi100
  python -m fmu2ml.scripts.generate_data --system marconi100 --duration 24 --output data/

  # Generate 1 week of data in 4 parallel chunks
  python -m fmu2ml.scripts.generate_data --system summit --duration 168 --chunks 4 --output data/
  
  # Custom seed and output format
  python -m fmu2ml.scripts.generate_data --system frontier --duration 48 --seed 123 --format csv
        """
    )
    
    # Required arguments
    parser.add_argument('--system', type=str, required=True,
                        choices=['marconi100', 'summit', 'frontier', 'fugaku', 'lassen'],
                        help='HPC system name')
    parser.add_argument('--duration', type=int, required=True,
                        help='Total duration in hours')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    
    # Optional arguments
    parser.add_argument('--chunks', type=int, default=1,
                        help='Number of parallel chunks (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--start-date', type=str, default='2023-01-01T00:00:00Z',
                        help='Start date (ISO format, default: 2023-01-01T00:00:00Z)')
    parser.add_argument('--format', type=str, default='parquet',
                        choices=['parquet', 'csv', 'hdf5'],
                        help='Output format (default: parquet)')
    parser.add_argument('--timestep', type=int, default=60,
                        help='Timestep in seconds (default: 60)')
    parser.add_argument('--use-raps', action='store_true', default=True,
                        help='Use RAPS configuration (default: True)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('generate_data')
    
    logger.info("=" * 80)
    logger.info("FMU2ML Data Generation")
    logger.info("=" * 80)
    logger.info(f"System: {args.system}")
    logger.info(f"Duration: {args.duration} hours")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunks: {args.chunks}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Format: {args.format}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate duration per chunk
    duration_per_chunk = args.duration // args.chunks
    
    if args.chunks == 1:
        # Single chunk generation
        result = generate_chunk((
            0, duration_per_chunk, 0, args.start_date, 
            args.system, str(output_dir), args.seed
        ))
        logger.info(f"Generated {result[1]} samples")
    else:
        # Parallel generation
        logger.info(f"Generating {args.chunks} chunks in parallel...")
        
        # Prepare arguments for each chunk
        chunk_args = [
            (
                i, 
                duration_per_chunk, 
                i * duration_per_chunk,
                args.start_date,
                args.system,
                str(output_dir),
                args.seed
            )
            for i in range(args.chunks)
        ]
        
        # Run in parallel
        with mp.Pool(processes=min(args.chunks, mp.cpu_count())) as pool:
            results = pool.map(generate_chunk, chunk_args)
        
        total_samples = sum(r[1] for r in results)
        logger.info(f"Generated {total_samples} total samples across {args.chunks} chunks")
    
    logger.info("=" * 80)
    logger.info("Data generation complete!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()