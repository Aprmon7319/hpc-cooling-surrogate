import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging

from raps.config import ConfigManager
from raps.cooling import ThermoFluidsModel


class FMUSimulator:
    """Main FMU simulation runner"""
    
    __slots__ = ['config', 'system_name', 'model', 'logger', '_stability_detector']
    
    def __init__(self, system_name: str = "marconi100", **config_overrides):
        """Initialize FMU simulator"""
        # Load configuration once
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        self.system_name = system_name
        
        # Initialize cooling model
        self.model = ThermoFluidsModel(**self.config)
        self.model.initialize()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"✓ FMU simulator initialized for {system_name}")
        self.logger.info(f"NUM_CDUS: {self.config['NUM_CDUS']}")
        
        # Lazy load stability detector
        self._stability_detector = None

    @property
    def stability_detector(self):
        """Lazy load stability detector"""
        if self._stability_detector is None:
            from .stability_detector import StabilityDetector
            self._stability_detector = StabilityDetector
        return self._stability_detector

    def reset(self):
        """Reset FMU to initial state"""
        if self.model:
            try:
                self.model.terminate()
            except:
                pass
        
        self.model = ThermoFluidsModel(**self.config)
        self.model.initialize()
        self.logger.info("FMU reset to initial state")
    
    def run_stabilization(
        self,
        first_row_data: Dict,
        max_hours: int = 4,
        min_hours: float = 0.5,
        step_size: int = 1
    ) -> Dict:
        """Run stabilization phase until steady state"""
        max_steps = int(max_hours * 3600 / step_size)
        min_steps = int(min_hours * 3600 / step_size)
        
        detector = self.stability_detector(
            window_size=300,
            threshold=0.1,
            min_steps=min_steps,
            max_steps=max_steps
        )
        
        self.logger.info(f"Starting stabilization (max {max_hours} hours)...")
        
        # Pre-calculate check intervals
        check_interval = 60  # Check every minute
        check_steps = set(range(min_steps, max_steps, check_interval))
        
        for i in range(max_steps):
            current_time = i * step_size
            
            # Generate FMU inputs
            fmu_inputs = self.model.generate_fmu_inputs(
                first_row_data,
                uncertainties=False
            )
            
            # Step the model
            _, fmu_output = self.model.step(current_time, fmu_inputs, step_size)
            
            # Check stability
            detector.add_step(fmu_output)
            
            if i in check_steps:
                is_stable, metrics = detector.is_stable()
                
                if is_stable:
                    self.logger.info(
                        f"Steady state reached at {current_time / 3600:.2f} hours"
                    )
                    return detector.get_summary()
        
        self.logger.warning(
            f"Max stabilization time ({max_hours} hours) reached"
        )
        return detector.get_summary()

    def run_simulation(
        self,
        input_data: pd.DataFrame,
        stabilization_hours: int = 2,
        step_size: int = 1,
        save_history: bool = True,
        reset_state: bool = False  
    ) -> pd.DataFrame:
        """Run complete FMU simulation"""
        
        # Reset if requested
        if reset_state:
            self.reset()
        
        # Stabilization phase
        if stabilization_hours > 0:
            self.reset()
            first_row = input_data.iloc[0].to_dict()
            stabilization_summary = self.run_stabilization(
                first_row,
                max_hours=stabilization_hours,
                step_size=step_size
            )
            current_time = stabilization_summary.get('total_steps', 0) * step_size
        else:
            current_time = 0
        
        # Clear history to save memory
        if not save_history and hasattr(self.model, 'fmu_history'):
            self.model.fmu_history.clear()
        
        # Operational simulation
        n_steps = len(input_data)
        self.logger.info(f"Running operational simulation for {n_steps} steps...")
        
        # Pre-allocate results list
        results = []
        
        # Process in chunks for better cache efficiency
        chunk_size = min(1000, n_steps)
        log_interval = 1800
        
        for chunk_idx in range(0, n_steps, chunk_size):
            chunk_end = min(chunk_idx + chunk_size, n_steps)
            chunk = input_data.iloc[chunk_idx:chunk_end]
            
            # Convert chunk to list of dicts once
            chunk_dicts = chunk.to_dict('records')
            
            for row_data in chunk_dicts:
                # Generate FMU inputs
                fmu_inputs = self.model.generate_fmu_inputs(
                    row_data,
                    uncertainties=False
                )
                
                # Step the model
                fmu_input, fmu_output = self.model.step(current_time, fmu_inputs, step_size)

                if fmu_output and fmu_input:
                    fmu_output.update(fmu_input)
                    results.append(fmu_output)
                
                current_time += step_size
            
            # Progress logging
            if chunk_end % log_interval == 0:
                self.logger.info(
                    f"Progress: {chunk_end}/{n_steps} steps "
                    f"({chunk_end/n_steps*100:.1f}%)"
                )
        
        self.logger.info("Simulation complete")
        return pd.DataFrame.from_records(results)

    def run_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        stabilization_hours: int = 2,
        step_size: int = 1
    ) -> pd.DataFrame:
        """Run simulation from input file"""
        # Load input data efficiently
        input_path = Path(input_file)
        if input_path.suffix == '.parquet':
            input_data = pd.read_parquet(input_file, engine='pyarrow')
        else:
            input_data = pd.read_csv(input_file, engine='c')
        
        self.logger.info(f"Loaded input data: {input_data.shape}")
        
        # Run simulation
        results = self.run_simulation(
            input_data,
            stabilization_hours=stabilization_hours,
            step_size=step_size
        )
        
        # Save results efficiently
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.parquet':
                results.to_parquet(output_file, index=False, 
                                 engine='pyarrow', compression='snappy')
            else:
                results.to_csv(output_file, index=False, chunksize=50000)
            
            self.logger.info(f"Results saved to: {output_file}")
        
        return results

    def cleanup(self):
        """Clean up FMU resources"""
        try:
            self.model.terminate()
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
            self.logger.info("FMU resources cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()