import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from raps.config import ConfigManager
from raps.cooling import ThermoFluidsModel

class FMUOutputGenerator:
    """
    Generate FMU simulation outputs from input data
    
    This class handles:
    - Loading input data
    - Running FMU simulation with stabilization
    - Streaming output data to disk
    - Memory management
    """
    
    __slots__ = ['logger', 'system_name', 'stabilization_hours', 'stabilization_threshold',
                 'step_size', 'output_dir', 'config', 'model', 'StabilityDetector', 'current_time']
    
    def __init__(
        self,
        system_name: str = "marconi100",
        stabilization_hours: int = 2,
        stabilization_threshold: float = 0.1,
        step_size: int = 1,
        current_time: int = 0,
        output_dir: Optional[str] = None,
        **config_overrides
    ):
        self.logger = logging.getLogger(__name__)
        self.current_time = current_time
        
        self.system_name = system_name
        self.stabilization_hours = stabilization_hours
        self.stabilization_threshold = stabilization_threshold
        self.step_size = step_size
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration once
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        # Initialize cooling model
        self.model = None
        self._initialize_model()
        
        self.logger.info(f"FMU output generator initialized for {system_name}")
        
        # Import stability detector
        from .stability_detector import StabilityDetector
        self.StabilityDetector = StabilityDetector

    def _initialize_model(self):
        """Initialize the cooling model"""
        self.model = ThermoFluidsModel(**self.config)
        self.model.initialize()
        self.logger.info("Cooling model initialized")
    
    def reset_model(self):
        """Reset model to initial state"""
        if self.model:
            try:
                self.model.terminate()
            except:
                pass
        self._initialize_model()
        self.logger.info("Model reset to initial state")
    
    def run_stabilization(
        self,
        first_row_data: Dict,
        max_hours: Optional[int] = None,
        min_hours: float = 0.5
    ) -> Dict:
        """Run stabilization phase until steady state"""
        if max_hours is None:
            max_hours = self.stabilization_hours
        
        max_steps = int(max_hours * 3600 / self.step_size)
        min_steps = int(min_hours * 3600 / self.step_size)
        
        # Initialize stability detector
        detector = self.StabilityDetector(
            window_size=300,
            threshold=self.stabilization_threshold,
            min_steps=min_steps,
            max_steps=max_steps
        )
        
        self.logger.info(f"Starting stabilization (max {max_hours} hours)...")
        
        # Pre-calculate time steps
        time_steps = np.arange(0, max_steps) * self.step_size
        check_points = np.arange(min_steps, max_steps, 60)  # Check every minute
        
        for i, current_time in enumerate(time_steps):
            # Generate FMU inputs
            fmu_inputs = self.model.generate_fmu_inputs(
                first_row_data,
                uncertainties=False
            )
            
            # Step the model
            _, fmu_output = self.model.step(current_time, fmu_inputs, self.step_size)
            
            # Check stability
            detector.add_step(fmu_output)
            
            if i in check_points:
                is_stable, metrics = detector.is_stable()
                
                if is_stable:
                    elapsed_hours = current_time / 3600
                    self.current_time = current_time
                    self.logger.info(f"Steady state reached at {elapsed_hours:.2f} hours")
                    return detector.get_summary()
            
            # Progress logging
            if i % 1800 == 0 and i > 0:
                self.logger.debug(f"Stabilization progress: {i}/{max_steps} steps")
        self.current_time = max_steps * self.step_size
        self.logger.warning(f"Max stabilization time ({max_hours} hours) reached")
        return detector.get_summary()

    def run_operational_simulation(
        self,
        input_data: pd.DataFrame,
        start_time: Optional[int] = None
    ) -> pd.DataFrame:
        """Run operational simulation after stabilization"""
        n_steps = len(input_data)
        
        self.logger.info(f"Running operational simulation for {n_steps} steps...")
        
        current_time = self.current_time
        
        for idx in range(n_steps):
            row_data = input_data.iloc[idx].to_dict()
            
            # Generate FMU inputs
            fmu_inputs = self.model.generate_fmu_inputs(row_data, uncertainties=False)
            
            # Step the model
            _, fmu_output = self.model.step(current_time, fmu_inputs, self.step_size)
            
            
            current_time += self.step_size
            
            # Progress logging
            if (idx + 1) % 1800 == 0:
                progress_pct = ((idx + 1) / n_steps) * 100
                self.logger.info(f"Progress: {idx + 1}/{n_steps} ({progress_pct:.1f}%)")
        
        output_df = pd.DataFrame.from_records(self.model.fmu_history)
        self.logger.info("Operational simulation complete")
        return output_df
    
    def generate_from_input(
        self,
        input_data: pd.DataFrame,
        output_file: Optional[str] = None,
        save_stabilization: bool = False
    ) -> pd.DataFrame:
        """Generate FMU output from input data"""
        self.reset_model()
        
        # Extract first row for stabilization
        first_row = input_data.iloc[0].to_dict()
        
        # Run stabilization
        stabilization_summary = self.run_stabilization(first_row)
        
        # Clear history if not saving stabilization data
        if not save_stabilization and hasattr(self.model, 'fmu_history'):
            self.model.fmu_history.clear()
        
        # Pre-allocate results list with estimated size
        n_steps = len(input_data)
        
        # Run operational simulation
        self.logger.info(f"Running operational simulation for {n_steps} steps...")
        current_time = stabilization_summary.get('total_steps', 0) * self.step_size
        
        # Process in chunks for better memory management
        chunk_size = 1000
        for chunk_start in range(0, n_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_steps)
            chunk_data = input_data.iloc[chunk_start:chunk_end]
            
            for idx in range(len(chunk_data)):
                row_data = chunk_data.iloc[idx].to_dict()
                
                # Generate FMU inputs
                fmu_inputs = self.model.generate_fmu_inputs(row_data, uncertainties=False)
                
                # Step the model
                _, fmu_output = self.model.step(current_time, fmu_inputs, self.step_size)
                
                
                current_time += self.step_size
            
            # Progress logging
            if chunk_end % 1800 == 0:
                progress_pct = (chunk_end / n_steps) * 100
                self.logger.info(f"Progress: {chunk_end}/{n_steps} ({progress_pct:.1f}%)")
        
        # Create output DataFrame efficiently
        output_df = pd.DataFrame.from_records(self.model.fmu_history)
        
        # Save to file if requested
        if output_file:
            self._save_output(output_df, output_file)
        
        self.logger.info("Simulation complete")
        return output_df

    def generate_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate FMU output from input file"""
        input_path = Path(input_file)
        self.logger.info(f"Loading input from: {input_file}")
        
        # Load data efficiently
        if input_path.suffix == '.parquet':
            input_data = pd.read_parquet(input_file, engine='pyarrow')
        else:
            input_data = pd.read_csv(input_file, engine='c')
        
        self.logger.info(f"Loaded input data: {input_data.shape}")
        
        # Generate output filename if not provided
        if output_file is None:
            duration_hours = len(input_data) * self.step_size / 3600
            output_file = self.output_dir / f"fmu_output_{duration_hours:.1f}hrs.parquet"
        
        return self.generate_from_input(input_data, str(output_file))

    def _save_output(self, output_df: pd.DataFrame, output_file: str):
        """Save output DataFrame to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            output_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        else:
            output_df.to_csv(output_path, index=False, chunksize=50000)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Output saved to: {output_path} ({file_size_mb:.1f} MB)")

    def cleanup(self):
        """Clean up resources"""
        if self.model:
            try:
                self.model.terminate()
                if hasattr(self.model, 'cleanup'):
                    self.model.cleanup()
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class BatchFMUOutputGenerator:
    """Generate FMU outputs for multiple input files in batch"""
    
    __slots__ = ['system_name', 'num_workers', 'memory_limit_mb', 'generator_kwargs', 'logger']
    
    def __init__(
        self,
        system_name: str = "marconi100",
        num_workers: int = 1,
        memory_limit_mb: int = 32768,
        **generator_kwargs
    ):
        self.system_name = system_name
        self.num_workers = num_workers
        self.memory_limit_mb = memory_limit_mb
        self.generator_kwargs = generator_kwargs
        self.logger = logging.getLogger(__name__)

    def generate_batch(
        self,
        input_files: List[str],
        output_dir: str,
        parallel: bool = False
    ) -> List[Dict]:
        """Generate outputs for multiple input files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing {len(input_files)} input files")
        
        if parallel and self.num_workers > 1:
            # Use spawn method for better memory isolation
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            
            # Process in batches to control memory usage
            batch_size = max(1, len(input_files) // (self.num_workers * 2))
            args = [
                (f, output_dir, i, self.system_name, self.generator_kwargs)
                for i, f in enumerate(input_files)
            ]
            
            with ctx.Pool(processes=self.num_workers, maxtasksperchild=2) as pool:
                results = pool.map(self._process_single_file, args, chunksize=1)
        else:
            # Sequential processing
            results = []
            for i, input_file in enumerate(input_files):
                result = self._process_single_file(
                    (input_file, output_dir, i, self.system_name, self.generator_kwargs)
                )
                results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        self.logger.info(f"Batch complete: {successful}/{len(results)} successful")
        
        return results

    @staticmethod
    def _process_single_file(args: Tuple) -> Dict:
        """Process a single file (static method for multiprocessing)"""
        input_file, output_dir, file_id, system_name, generator_kwargs = args
        
        logger = logging.getLogger(__name__)
        logger.info(f"[File {file_id}] Processing {Path(input_file).name}")
        
        try:
            # Create output filename
            input_path = Path(input_file)
            output_file = output_dir / f"output_{input_path.stem}.parquet"
            
            # Generate output
            with FMUOutputGenerator(
                system_name=system_name,
                output_dir=str(output_dir),
                **generator_kwargs
            ) as generator:
                output_df = generator.generate_from_file(
                    str(input_file),
                    str(output_file)
                )
            
            return {
                'file_id': file_id,
                'input_file': str(input_file),
                'output_file': str(output_file),
                'success': True,
                'num_samples': len(output_df),
                'message': 'Success'
            }
            
        except Exception as e:
            logger.error(f"[File {file_id}] Error: {str(e)}")
            return {
                'file_id': file_id,
                'input_file': str(input_file),
                'success': False,
                'error': str(e),
                'message': f'Failed: {str(e)}'
            }


# Convenience function
def generate_fmu_output(
    input_file: str,
    output_file: Optional[str] = None,
    system_name: str = "marconi100",
    stabilization_hours: int = 2,
    step_size: int = 1,
    **config_overrides
) -> pd.DataFrame:
    """Convenience function to generate FMU output from input file"""
    with FMUOutputGenerator(
        system_name=system_name,
        stabilization_hours=stabilization_hours,
        step_size=step_size,
        **config_overrides
    ) as generator:
        return generator.generate_from_file(input_file, output_file)