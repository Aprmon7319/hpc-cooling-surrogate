import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Callable
import pandas as pd
import logging
from datetime import datetime
import gc

from .fmu_simulator import FMUSimulator


class ParallelRunner:
    """Run multiple FMU simulations in parallel"""
    
    __slots__ = ['system_name', 'config_overrides', 'n_workers', 'logger']
    
    def __init__(
        self,
        system_name: str = "marconi100",
        n_workers: int = None,
        **config_overrides
    ):
        """Initialize parallel runner"""
        self.system_name = system_name
        self.config_overrides = config_overrides
        self.n_workers = n_workers or mp.cpu_count()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Parallel runner initialized with {self.n_workers} workers"
        )

    @staticmethod
    def _run_single_simulation(args: tuple) -> Dict:
        """Run a single simulation (static method for multiprocessing)"""
        (chunk_id, input_data, system_name, stabilization_hours, 
         step_size, config_overrides) = args
        
        logger = logging.getLogger(__name__)
        logger.info(f"[Chunk {chunk_id}] Starting simulation...")
        
        simulator = None
        try:
            # Create simulator instance
            simulator = FMUSimulator(
                system_name=system_name,
                **config_overrides
            )
            
            # Run simulation
            results = simulator.run_simulation(
                input_data,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                save_history=False
            )
            
            return {
                'chunk_id': chunk_id,
                'success': True,
                'results': results,
                'num_steps': len(results),
                'message': 'Simulation completed successfully'
            }
            
        except Exception as e:
            logger.error(f"[Chunk {chunk_id}] Error: {str(e)}")
            return {
                'chunk_id': chunk_id,
                'success': False,
                'error': str(e),
                'message': f'Simulation failed: {str(e)}'
            }
        finally:
            # Ensure cleanup
            if simulator:
                simulator.cleanup()
            gc.collect()

    def run_batch(
        self,
        input_list: List[pd.DataFrame],
        stabilization_hours: int = 2,
        step_size: int = 1,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """Run batch of simulations in parallel"""
        # Prepare arguments for each simulation
        args_list = [
            (
                i,
                input_data,
                self.system_name,
                stabilization_hours,
                step_size,
                self.config_overrides
            )
            for i, input_data in enumerate(input_list)
        ]
        
        # Run simulations in parallel with memory-aware processing
        self.logger.info(
            f"Starting {len(input_list)} simulations "
            f"with {self.n_workers} workers..."
        )
        
        # Use spawn context for better memory isolation
        ctx = mp.get_context('spawn')
        
        # Configure pool with memory limits
        with ctx.Pool(
            processes=self.n_workers,
            maxtasksperchild=2  # Restart workers after 2 tasks to prevent memory bloat
        ) as pool:
            results = pool.map(
                self._run_single_simulation, 
                args_list,
                chunksize=1  # Process one at a time for better memory control
            )
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for result in results:
                if result['success']:
                    chunk_id = result['chunk_id']
                    output_file = output_path / f"chunk_{chunk_id}_results.parquet"
                    result['results'].to_parquet(
                        output_file, 
                        index=False,
                        engine='pyarrow',
                        compression='snappy'
                    )
                    self.logger.info(f"Saved chunk {chunk_id} to {output_file}")
        
        # Log summary
        success_count = sum(1 for r in results if r['success'])
        self.logger.info(
            f"Batch complete: {success_count}/{len(results)} successful"
        )
        
        return results

    def run_from_files(
        self,
        input_files: List[str],
        output_dir: str,
        stabilization_hours: int = 2,
        step_size: int = 1
    ) -> List[Dict]:
        """Run simulations from input files"""
        # Load all input files efficiently
        self.logger.info(f"Loading {len(input_files)} input files...")
        input_list = []
        
        for file_path in input_files:
            path = Path(file_path)
            try:
                if path.suffix == '.parquet':
                    data = pd.read_parquet(file_path, engine='pyarrow')
                else:
                    data = pd.read_csv(file_path, engine='c')
                input_list.append(data)
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not input_list:
            self.logger.error("No valid input files loaded")
            return []
        
        # Run batch
        return self.run_batch(
            input_list,
            stabilization_hours=stabilization_hours,
            step_size=step_size,
            output_dir=output_dir
        )