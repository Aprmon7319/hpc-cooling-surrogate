import multiprocessing as mp
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .full_generator import generate_complete_fmu_dataset
from raps.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('parallel_generator')

def run_chunk(args):
    """Generate a single chunk of data in its own directory"""
    chunk_id, duration_per_chunk, start_offset_hours, base_date, config, output_dir = args
    
    try:
        # Calculate start date for this chunk
        start_dt = datetime.fromisoformat(base_date.replace('Z', '+00:00'))
        start_dt += timedelta(hours=start_offset_hours)
        start_date_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Create chunk-specific output directory
        chunk_output_dir = f'{output_dir}/chunk_{chunk_id}'
        Path(chunk_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Chunk {chunk_id}: Starting from {start_date_str} ({duration_per_chunk}h)")
        logger.info(f"Chunk {chunk_id}: Output dir: {chunk_output_dir}")
        
        # Generate dataset for this chunk
        result = generate_complete_fmu_dataset(
            n_cdus=config.get('n_cdus', 0),
            duration_hours=duration_per_chunk,
            timestep_seconds=config.get('timestep_seconds', 1),
            start_date=start_date_str,
            scenario_distribution=config.get('scenario_distribution', [0.5, 0.4, 0.1]),
            seed=config.get('seed', 42) + chunk_id,  # Different seed per chunk
            output_dir=chunk_output_dir,
            save_output=True,
            config=config.get('system_config', None)
        )
        
        logger.info(f"Chunk {chunk_id}: ✓ Complete - Shape: {result['fmu_input'].shape}, Saved to: {chunk_output_dir}")
        return chunk_id, True, chunk_output_dir, result['filename']
    
    except Exception as e:
        logger.error(f"Chunk {chunk_id}: ✗ Failed - {str(e)}")
        return chunk_id, False, None, str(e)

def generate_parallel_dataset(
    total_duration: int = 576,
    num_chunks: int = 24,
    n_cdus: int = 0,
    timestep_seconds: int = 1,
    scenario_distribution: List[float] = None,
    base_date: str = None,
    num_processes: int = None,
    output_dir: str = "data",
    system_name: str = "marconi100",
    seed: int = 42
) -> Dict:
    """
    Generate FMU dataset in parallel, saving each chunk to a separate directory.
    
    Parameters:
    -----------
    total_duration : int
        Total duration in hours
    num_chunks : int
        Number of chunks to split the generation into
    n_cdus : int
        Number of CDUs to simulate
    timestep_seconds : int
        Time step in seconds
    scenario_distribution : List[float]
        Distribution of [normal, edge, fault] scenarios
    base_date : str
        Start date in ISO format
    num_processes : int
        Number of parallel processes (default: CPU count)
    output_dir : str
        Base output directory (chunks will be in output_dir/chunk_0, chunk_1, etc.)
    
    Returns:
    --------
    Dict with generation results and statistics
    """
    if scenario_distribution is None:
        scenario_distribution = [0.5, 0.4, 0.1]
    
    if base_date is None:
        base_date = "2024-01-01T00:00:00Z"
    
    if num_processes is None:
        num_processes = min(num_chunks, mp.cpu_count())
    
    duration_per_chunk = total_duration // num_chunks
    
    
    # Get n_cdus from system config if not explicitly provided (n_cdus <= 0)
    if n_cdus <= 0:       
        system_config = ConfigManager(system_name=system_name).get_config()
        n_cdus = system_config.get('NUM_CDUS', 49)  # Default to 49 if not in config
        logger.info(f"Using system config for {system_name}: {n_cdus} CDUs")
    else:
        system_config = ConfigManager(system_name=system_name).get_config()
        logger.info(f"Using explicit n_cdus: {n_cdus} CDUs (overriding system config)")
    
    # Configuration
    config = {
        'n_cdus': n_cdus,
        'timestep_seconds': timestep_seconds,
        'scenario_distribution': scenario_distribution,
        'system_config': ConfigManager(system_name=system_name).get_config(),
        'seed': seed
    }
    
    logger.info("="*60)
    logger.info(f"Parallel FMU Dataset Generation")
    logger.info(f"Total: {total_duration}h in {num_chunks} chunks ({duration_per_chunk}h each)")
    logger.info(f"Config: {n_cdus} CDUs, {timestep_seconds}s timestep")
    logger.info(f"Distribution: Normal={scenario_distribution[0]:.0%}, Edge={scenario_distribution[1]:.0%}, Fault={scenario_distribution[2]:.0%}")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"Output structure: {output_dir}/chunk_0/, chunk_1/, ..., chunk_{num_chunks-1}/")
    logger.info("="*60)
    
    # Create arguments for each chunk
    chunk_args = [
        (i, duration_per_chunk, i * duration_per_chunk, base_date, config, output_dir)
        for i in range(num_chunks)
    ]
    
    # Run chunks in parallel
    start_time = datetime.now()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_chunk, chunk_args)
    
    elapsed = datetime.now() - start_time
    
    # Summary
    successful = sum(1 for _, success, _, _ in results if success)
    failed = num_chunks - successful
    
    logger.info("="*60)
    logger.info(f"✓ Complete: {successful}/{num_chunks} successful in {elapsed}")
    logger.info(f"Average time per chunk: {elapsed / num_chunks}")
    
    # List all chunk directories
    logger.info("\nGenerated chunk directories:")
    for chunk_id, success, chunk_dir, filename in results:
        if success:
            logger.info(f"  ✓ {chunk_dir}/ -> {filename}")
        else:
            logger.error(f"  ✗ Chunk {chunk_id} failed: {filename}")
    
    if failed > 0:
        logger.warning(f"\n⚠ Failed chunks: {failed}/{num_chunks}")
    
    return {
        'successful_chunks': successful,
        'failed_chunks': failed,
        'total_time': elapsed,
        'avg_time_per_chunk': elapsed / num_chunks,
        'chunk_results': results,
        'output_directories': [r[2] for r in results if r[1]],
        'base_output_dir': output_dir
    }

# Example command-line usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate FMU dataset in parallel chunks')
    parser.add_argument('--duration', type=int, default=576, help='Total duration in hours')
    parser.add_argument('--chunks', type=int, default=24, help='Number of chunks')
    parser.add_argument('--cdus', type=int, default=0, help='Number of CDUs')
    parser.add_argument('--timestep', type=int, default=1, help='Timestep in seconds')
    parser.add_argument('--processes', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--output', type=str, default='data', help='Base output directory')
    parser.add_argument('--start-date', type=str, default='2024-01-01T00:00:00Z', help='Start date (ISO format)')
    parser.add_argument('--system-name', type=str, default='marconi100', help='System configuration name')
    parser.add_argument('--scenario-distribution', type=float, nargs=3, default=[0.5, 0.4, 0.1],
                        help='Scenario distribution as three floats (normal, edge, fault)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (not used in parallel generation)')
    args = parser.parse_args()
    
    result = generate_parallel_dataset(
        total_duration=args.duration,
        num_chunks=args.chunks,
        n_cdus=args.cdus,
        timestep_seconds=args.timestep,
        base_date=args.start_date,
        num_processes=args.processes,
        output_dir=args.output,
        system_name=args.system_name,
        scenario_distribution=args.scenario_distribution,
        seed=args.seed
    )
    
    print(f"\n✓ Generation complete!")
    print(f"  Successful: {result['successful_chunks']}/{args.chunks} chunks")
    print(f"  Total time: {result['total_time']}")
    print(f"  Data saved in: {result['base_output_dir']}/chunk_*/ directories")