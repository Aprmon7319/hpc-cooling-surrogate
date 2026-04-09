import os
import sys
import gc
import time
import glob
import logging
import argparse
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import concurrent.futures

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

# Import from fmu2ml modules
from fmu2ml.simulation import FMUOutputGenerator
from raps.config import ConfigManager

# Constants
PROGRESS_INTERVAL = 1800  # 30 minutes
MAX_MEMORY_MB = 8192  # Maximum memory usage before forcing garbage collection

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"fmu_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_chunk(
    chunk_id: int,
    input_dir: str,
    output_dir: Optional[str] = None,
    system_name: str = 'marconi100',
    stabilization_hours: int = 2,
    step_size: int = 1,
    stabilization_threshold: float = 0.1
) -> Dict:
    """Process chunk using FMUOutputGenerator"""
    
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        
        # Set default output directory
        if output_dir is None:
            output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Find input file
        input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
        if not input_files:
            raise FileNotFoundError(f"No parquet files found in {input_dir}")
        
        input_file = sorted(input_files)[0]
        logger.info(f"[Chunk {chunk_id}] Processing: {input_file}")
        
        # Load input data
        df = pd.read_parquet(input_file)
        logger.info(f"[Chunk {chunk_id}] Input shape: {df.shape}")
        
        # Load configuration
        config = ConfigManager(system_name=system_name).get_config()
        
        # Remove system_name from config to avoid duplicate keyword argument
        config_overrides = {k: v for k, v in config.items() if k != 'system_name'}
        
        # Initialize FMU output generator
        with FMUOutputGenerator(
            system_name=system_name,
            stabilization_hours=stabilization_hours,
            stabilization_threshold=stabilization_threshold,
            step_size=step_size,
            output_dir=output_dir,
            **config_overrides  # Now safe to unpack without system_name
        ) as generator:
            
            # Generate output
            logger.info(f"[Chunk {chunk_id}] Starting FMU simulation...")
            
            # Determine output filename
            actual_hours = len(df) * step_size / 3600
            output_filename = f"fmu_output_{actual_hours:.1f}hrs_operational.parquet"
            output_file = os.path.join(output_dir, output_filename)
            
            # Generate FMU output
            output_df = generator.generate_from_input(
                input_data=df,
                output_file=output_file,
                save_stabilization=False
            )
            
            logger.info(f"[Chunk {chunk_id}] Output shape: {output_df.shape}")
        
        # Clean up
        del df, output_df
        gc.collect()
        
        # Calculate statistics
        elapsed_total = time.time() - start_time
        
        # Read row count without loading full data
        parquet_file_info = pq.read_metadata(output_file)
        total_steps = parquet_file_info.num_rows
        
        actual_hours = total_steps / 3600
        
        stats = {
            "chunk_id": chunk_id,
            "input_file": input_file,
            "output_file": output_file,
            "processed_steps": total_steps,
            "processed_hours": actual_hours,
            "elapsed_seconds": elapsed_total,
            "steps_per_second": total_steps / elapsed_total if elapsed_total > 0 else 0,
            "max_memory_mb": get_memory_usage(),
            "success": True
        }

        logger.info(
            f"[Chunk {chunk_id}] Completed in {elapsed_total:.1f}s "
            f"({stats['steps_per_second']:.1f} steps/s)"
        )
        
        # Save chunk statistics
        stats_file = os.path.join(output_dir, f"fmu_stats_chunk_{chunk_id}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        return stats
        
    except Exception as e:
        logger.error(f"[Chunk {chunk_id}] ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "chunk_id": chunk_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }


def main():
    """Main function for parallel FMU output generation"""
    
    # Setup logging for main process
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Generate FMU output from input data")
    
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Base directory containing chunk folders")
    parser.add_argument("--chunks", type=str, default="all",
                       help="Chunks to process (comma-separated, or 'all')")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel processes to use")
    parser.add_argument("--step-size", type=int, default=1,
                       help="Step size for processing")
    parser.add_argument("--stabilization-hours", type=int, default=3,
                       help="Hours for stabilization phase")
    parser.add_argument("--stabilization-threshold", type=float, default=0.1,
                       help="Threshold for stabilization detection")
    parser.add_argument("--system-name", type=str, default="marconi100",
                       help="System configuration name")
    parser.add_argument("--resume", action="store_true",
                       help="Skip chunks that already have output files")
    parser.add_argument("--memory-limit", type=int, default=32768,
                       help="Memory limit in MB per process (default: 32GB)")
    
    args = parser.parse_args()
    
    # Set memory limit for child processes
    global MAX_MEMORY_MB
    MAX_MEMORY_MB = args.memory_limit
    
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Memory limit per process: {MAX_MEMORY_MB} MB")
    logger.info(f"System configuration: {args.system_name}")
    
    # Find chunk directories
    if args.chunks == "all":
        chunk_dirs = sorted(glob.glob(os.path.join(args.data_dir, "chunk_*")))
        chunk_ids = [int(os.path.basename(d).split("_")[1]) for d in chunk_dirs]
    else:
        chunk_ids = [int(c.strip()) for c in args.chunks.split(",")]
        chunk_dirs = [os.path.join(args.data_dir, f"chunk_{c}") for c in chunk_ids]
    
    # Filter out already processed chunks if resuming
    if args.resume:
        filtered_chunks = []
        filtered_dirs = []
        for chunk_id, chunk_dir in zip(chunk_ids, chunk_dirs):
            output_files = glob.glob(os.path.join(chunk_dir, "fmu_output_*_operational.parquet"))
            if not output_files:
                filtered_chunks.append(chunk_id)
                filtered_dirs.append(chunk_dir)
            else:
                logger.info(f"Skipping chunk {chunk_id} (already processed)")
        
        chunk_ids = filtered_chunks
        chunk_dirs = filtered_dirs
    
    if not chunk_ids:
        logger.info("No chunks to process")
        return
    
    logger.info(f"Processing {len(chunk_ids)} chunks: {chunk_ids}")
    
    # Create tasks
    tasks = []
    for chunk_id, chunk_dir in zip(chunk_ids, chunk_dirs):
        tasks.append((
            chunk_id,
            chunk_dir,
            chunk_dir,
            args.system_name,
            args.stabilization_hours,
            args.stabilization_threshold,
            args.step_size
        ))
    
    # Run in parallel or sequentially
    all_stats = []
    start_time = time.time()
    
    if args.parallel > 1:
        logger.info(f"Starting parallel processing with {args.parallel} workers")
        
        # Calculate max workers based on available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        max_workers_by_memory = int(available_memory / MAX_MEMORY_MB)
        actual_workers = min(args.parallel, max_workers_by_memory)
        
        if actual_workers < args.parallel:
            logger.warning(
                f"Reducing workers from {args.parallel} to {actual_workers} "
                f"based on available memory"
            )
        
        # Use process pool with memory limits
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [executor.submit(process_chunk, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                all_stats.append(future.result())
    else:
        logger.info("Starting sequential processing")
        for task in tasks:
            stat = process_chunk(*task)
            all_stats.append(stat)
    
    # Write summary
    success_count = sum(1 for s in all_stats if s.get("success", False))
    failed_count = len(all_stats) - success_count
    total_time = time.time() - start_time
    
    logger.info("="*60)
    logger.info(f"Processing complete:")
    logger.info(f"  Total chunks: {len(chunk_ids)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info("="*60)
    
    # Write detailed stats to CSV and JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV for successful runs
    stats_df = pd.DataFrame([s for s in all_stats if s.get("success", False)])
    if not stats_df.empty:
        csv_file = os.path.join(args.data_dir, f"fmu_processing_stats_{timestamp}.csv")
        stats_df.to_csv(csv_file, index=False)
        logger.info(f"Processing statistics saved to {csv_file}")
        
        # Calculate aggregate statistics
        total_steps = stats_df['processed_steps'].sum()
        total_hours = stats_df['processed_hours'].sum()
        avg_speed = stats_df['steps_per_second'].mean()
        
        logger.info(f"Aggregate statistics:")
        logger.info(f"  Total steps processed: {total_steps:,}")
        logger.info(f"  Total hours simulated: {total_hours:.1f}")
        logger.info(f"  Average speed: {avg_speed:.1f} steps/second")
    
    # JSON for all runs (including failures)
    json_file = os.path.join(args.data_dir, f"fmu_processing_report_{timestamp}.json")
    report = {
        "timestamp": timestamp,
        "summary": {
            "total_chunks": len(chunk_ids),
            "successful": success_count,
            "failed": failed_count,
            "total_time_seconds": total_time,
            "parallel_workers": args.parallel,
            "system_name": args.system_name
        },
        "chunks": all_stats
    }
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Full report saved to {json_file}")
    
    # List failed chunks for easy retry
    if failed_count > 0:
        failed_chunks = [s['chunk_id'] for s in all_stats if not s.get('success', False)]
        logger.error(f"Failed chunks: {','.join(map(str, failed_chunks))}")
        logger.error(f"Retry with: --chunks {','.join(map(str, failed_chunks))}")

if __name__ == "__main__":
    main()