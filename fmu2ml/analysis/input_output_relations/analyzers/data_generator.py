import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional
import logging
import multiprocessing as mp
from functools import partial

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


def _simulate_grid_point(
    args: Tuple,
    system_name: str,
    config_overrides: Dict,
    num_cdus: int,
    stabilization_hours: int,
    step_size: int,
    steady_state_steps: int
) -> Optional[pd.DataFrame]:
    """
    Worker function to simulate a single grid point.
    Each worker creates its own FMU instance for independent simulation.
    
    Args:
        args: Tuple of (grid_idx, val1, val2, input1, input2, fixed_inputs)
        system_name: System configuration name
        config_overrides: Configuration overrides
        num_cdus: Number of CDUs
        stabilization_hours: Hours for stabilization
        step_size: Simulation step size
        steady_state_steps: Steps to run after stabilization
        
    Returns:
        DataFrame with averaged steady-state results, or None if failed
    """
    grid_idx, val1, val2, input1, input2, fixed_inputs = args
    
    # Import here to avoid issues with multiprocessing
    from fmu2ml.simulation.fmu_simulator import FMUSimulator
    
    try:
        # Create independent FMU instance for this grid point
        simulator = FMUSimulator(
            system_name=system_name,
            **config_overrides
        )
        
        # Set up inputs for this grid point
        inputs_dict = fixed_inputs.copy()
        inputs_dict[input1] = val1
        inputs_dict[input2] = val2
        
        # Helper to create input row with current grid point values
        def create_input_row():
            row = {}
            row['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = inputs_dict['T_ext']
            for cdu_idx in range(1, num_cdus + 1):
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = inputs_dict['Q_flow']
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = inputs_dict['T_Air']
            return row
        
        # Calculate total steps needed for stabilization + measurement
        stabilization_steps = int(stabilization_hours * 3600 / step_size)
        total_steps = stabilization_steps + steady_state_steps
        
        # Create input DataFrame with the SAME inputs for entire simulation
        # This ensures the system stabilizes AT the operating point we want to measure
        input_rows = [create_input_row() for _ in range(total_steps)]
        input_df = pd.DataFrame(input_rows)
        
        # Run simulation WITHOUT additional stabilization phase
        # (we're already including stabilization in our input data)
        result = simulator.run_simulation(
            input_data=input_df,
            stabilization_hours=0,  # No separate stabilization - we handle it in input_df
            step_size=step_size,
            save_history=False,
            reset_state=True
        )
        
        # Clean up FMU resources
        simulator.cleanup()
        
        if not result.empty and len(result) >= steady_state_steps:
            # Take only the last steady_state_steps rows (after stabilization)
            # and average them for the final measurement
            last_n = min(steady_state_steps, len(result))
            avg_result = result.iloc[-last_n:].mean().to_frame().T
            
            # Add grid point metadata
            avg_result['_grid_idx'] = grid_idx
            avg_result[f'_input_{input1}'] = val1
            avg_result[f'_input_{input2}'] = val2
            
            return avg_result
        
        return None
        
    except Exception as e:
        logger.warning(f"Grid point {grid_idx} ({val1:.2f}, {val2:.2f}) failed: {e}")
        return None


class DataGenerator:
    """
    Generates simulation data suitable for direct effect analysis.
    Uses a single FMU instance to maintain simulation state continuity.
    """
    
    def __init__(
        self, 
        system_name: str = 'marconi100',
        **config_overrides
    ):
        """
        Initialize data generator.
        
        Args:
            system_name: System configuration name (e.g., 'marconi100', 'leonardo')
            **config_overrides: Additional configuration overrides
        """
        self.system_name = system_name
        
        # Load system configuration
        self.config = ConfigManager(system_name=system_name).get_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        # Get number of CDUs from config
        if system_name == 'lassen':
            self.num_cdus = self.config.get('NUM_CDUS', 45)
        elif system_name == 'marconi100':
            self.num_cdus = self.config.get('NUM_CDUS', 44)
        else:
            self.num_cdus = self.config.get('NUM_CDUS', 257) 
        
        # Store config overrides for FMU simulator
        self.config_overrides = {k: v for k, v in self.config.items() if k != 'system_name'}
        
        # FMU simulator instance (lazy initialization)
        self._simulator = None
        
        logger.info(f"DataGenerator initialized for system: {system_name}")
        logger.info(f"Number of CDUs: {self.num_cdus}")
    
    @property
    def simulator(self):
        """Lazy initialization of FMU simulator."""
        if self._simulator is None:
            from fmu2ml.simulation.fmu_simulator import FMUSimulator
            logger.info(f"Initializing FMU simulator for {self.system_name}...")
            self._simulator = FMUSimulator(
                system_name=self.system_name,
                **self.config_overrides
            )
        return self._simulator
    
    def cleanup(self):
        """Clean up FMU simulator resources."""
        if self._simulator is not None:
            try:
                self._simulator.cleanup()
            except Exception as e:
                logger.warning(f"Error during simulator cleanup: {e}")
            finally:
                self._simulator = None
                logger.info("FMU simulator cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
        

    def generate_sensitivity_data(
        self,
        total_timesteps: int = 3600,
        samples_per_hour: int = 60,
        input_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1,
        distribute_across_cdus: bool = True,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate data for sensitivity analysis with varied inputs.
        Creates continuous time-series with smooth transitions between conditions.
        
        NOTE: FMU simulation is inherently sequential (each state depends on previous).
        This method runs simulation in a single thread to maintain state continuity.
        For large datasets, use batch_size to save intermediate results and reduce memory.
        
        Args:
            total_timesteps: Total number of simulation timesteps
            samples_per_hour: Number of unique operating conditions to sample per hour
            input_ranges: Custom ranges for inputs (min, max)
            stabilization_hours: Hours for stabilization phase
            step_size: Simulation step size in seconds
            distribute_across_cdus: If True, each CDU gets different input samples
            seed: Random seed for reproducibility
            batch_size: If set, save results every batch_size timesteps to reduce memory
            save_intermediate: If True, save intermediate batch results to disk
            intermediate_dir: Directory for intermediate results (required if save_intermediate=True)
            
        Returns:
            DataFrame with simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate derived parameters
        steps_per_hour = 3600 // step_size
        total_hours = total_timesteps / steps_per_hour
        n_samples = int(total_hours * samples_per_hour)
        transition_steps = total_timesteps // n_samples if n_samples > 0 else total_timesteps
        
        logger.info(f"Generating sensitivity data (sequential FMU simulation):")
        logger.info(f"  Total timesteps: {total_timesteps}")
        logger.info(f"  Total simulation hours: {total_hours:.2f}")
        logger.info(f"  Samples per hour: {samples_per_hour}")
        logger.info(f"  Total unique conditions: {n_samples}")
        logger.info(f"  Transition steps between conditions: {transition_steps}")
        logger.info(f"  Distribute across CDUs: {distribute_across_cdus}")
        
        if input_ranges is None:
            input_ranges = {
                'Q_flow': (50.0, 200.0),  # kW
                'T_Air': (288.15, 308.15),     # K
                'T_ext': (283.15, 313.15)      # K
            }
        
        # Generate input data
        if distribute_across_cdus:
            input_df = self._generate_distributed_inputs(
                total_timesteps=total_timesteps,
                n_samples=n_samples,
                transition_steps=transition_steps,
                input_ranges=input_ranges
            )
        else:
            input_df = self._generate_uniform_inputs(
                total_timesteps=total_timesteps,
                n_samples=n_samples,
                transition_steps=transition_steps,
                input_ranges=input_ranges
            )
        
        logger.info(f"Generated input DataFrame with {len(input_df)} timesteps")
        
        # Run simulation - MUST be sequential to maintain state continuity
        if batch_size is not None and batch_size < total_timesteps:
            results_df = self._run_batched_simulation(
                input_df=input_df,
                batch_size=batch_size,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                save_intermediate=save_intermediate,
                intermediate_dir=intermediate_dir
            )
        else:
            results_df = self._run_single_simulation(
                input_df=input_df,
                stabilization_hours=stabilization_hours,
                step_size=step_size
            )
        
        return results_df
    
    def _run_single_simulation(
        self,
        input_df: pd.DataFrame,
        stabilization_hours: int,
        step_size: int
    ) -> pd.DataFrame:
        """
        Run a single continuous FMU simulation.
        
        Args:
            input_df: Input data DataFrame
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            
        Returns:
            DataFrame with simulation results
        """
        logger.info("Running single continuous FMU simulation...")
        
        try:
            results_df = self.simulator.run_simulation(
                input_data=input_df,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                save_history=False
            )
            
            logger.info(f"Successfully generated {len(results_df)} samples")
            return results_df
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _run_batched_simulation(
        self,
        input_df: pd.DataFrame,
        batch_size: int,
        stabilization_hours: int,
        step_size: int,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run FMU simulation in batches to manage memory for large datasets.
        Maintains state continuity between batches by keeping FMU instance alive.
        
        Args:
            input_df: Input data DataFrame
            batch_size: Number of timesteps per batch
            stabilization_hours: Hours for stabilization (only for first batch)
            step_size: Simulation step size
            save_intermediate: Whether to save intermediate results
            intermediate_dir: Directory for intermediate results
            
        Returns:
            DataFrame with combined simulation results
        """
        import os
        
        if save_intermediate and intermediate_dir is None:
            raise ValueError("intermediate_dir required when save_intermediate=True")
        
        if save_intermediate:
            os.makedirs(intermediate_dir, exist_ok=True)
        
        total_timesteps = len(input_df)
        n_batches = (total_timesteps + batch_size - 1) // batch_size
        
        logger.info(f"Running batched simulation: {n_batches} batches of {batch_size} timesteps")
        
        all_results = []
        batch_files = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_timesteps)
            
            batch_input = input_df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"Processing batch {batch_idx + 1}/{n_batches} "
                       f"(timesteps {start_idx}-{end_idx})")
            
            try:
                # Only apply stabilization for the first batch
                stab_hours = stabilization_hours if batch_idx == 0 else 0
                
                # Run simulation for this batch
                # The FMU simulator maintains internal state between calls
                batch_results = self.simulator.run_simulation(
                    input_data=batch_input,
                    stabilization_hours=stab_hours,
                    step_size=step_size,
                    save_history=False,
                    reset_state=False  # Keep state from previous batch
                )
                
                if save_intermediate:
                    batch_file = os.path.join(
                        intermediate_dir, 
                        f'batch_{batch_idx:04d}.parquet'
                    )
                    batch_results.to_parquet(batch_file)
                    batch_files.append(batch_file)
                    logger.info(f"Saved batch {batch_idx + 1} to {batch_file}")
                else:
                    all_results.append(batch_results)
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Try to continue with next batch after re-initialization
                logger.warning("Attempting to continue after error...")
                self.cleanup()
                continue
        
        # Combine results
        if save_intermediate:
            logger.info(f"Loading and combining {len(batch_files)} batch files...")
            all_results = [pd.read_parquet(f) for f in batch_files]
        
        if not all_results:
            logger.error("No successful batches!")
            return pd.DataFrame()
        
        results_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {len(results_df)} total samples from {len(all_results)} batches")
        
        return results_df
    
    def _generate_distributed_inputs(
        self,
        total_timesteps: int,
        n_samples: int,
        transition_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Generate input data with different samples distributed across CDUs.
        """
        logger.info("Generating distributed inputs across CDUs...")
        
        # T_ext is shared across all CDUs
        sampler_shared = qmc.LatinHypercube(d=1)
        t_ext_samples = sampler_shared.random(n=n_samples)
        t_ext_targets = qmc.scale(
            t_ext_samples,
            input_ranges['T_ext'][0],
            input_ranges['T_ext'][1]
        ).flatten()
        
        # Interpolate T_ext for all timesteps
        target_indices = np.linspace(0, n_samples - 1, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, total_timesteps)
        t_ext_series = np.interp(sample_indices, target_indices, t_ext_targets)
        
        # Generate CDU-specific inputs
        q_flow_per_cdu = {}
        t_air_per_cdu = {}
        
        for cdu_idx in range(1, self.num_cdus + 1):
            sampler_cdu = qmc.LatinHypercube(d=2, seed=cdu_idx * 1000)
            cdu_samples = sampler_cdu.random(n=n_samples)
            
            q_flow_targets = qmc.scale(
                cdu_samples[:, 0:1],
                input_ranges['Q_flow'][0] * 1000,
                input_ranges['Q_flow'][1] * 1000
            ).flatten()
            
            t_air_targets = qmc.scale(
                cdu_samples[:, 1:2],
                input_ranges['T_Air'][0],
                input_ranges['T_Air'][1]
            ).flatten()
            
            # Phase shift for temporal diversity
            phase_shift = (cdu_idx - 1) * n_samples // self.num_cdus
            q_flow_targets = np.roll(q_flow_targets, phase_shift)
            t_air_targets = np.roll(t_air_targets, phase_shift)
            
            q_flow_per_cdu[cdu_idx] = np.interp(sample_indices, target_indices, q_flow_targets)
            t_air_per_cdu[cdu_idx] = np.interp(sample_indices, target_indices, t_air_targets)
        
        # Build input DataFrame
        input_data_rows = []
        for i in range(total_timesteps):
            row = {
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext': t_ext_series[i]
            }
            
            for cdu_idx in range(1, self.num_cdus + 1):
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = q_flow_per_cdu[cdu_idx][i]
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = t_air_per_cdu[cdu_idx][i]
            
            input_data_rows.append(row)
        
        return pd.DataFrame(input_data_rows)
    
    def _generate_uniform_inputs(
        self,
        total_timesteps: int,
        n_samples: int,
        transition_steps: int,
        input_ranges: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Generate input data with same samples for all CDUs.
        """
        logger.info("Generating uniform inputs (same for all CDUs)...")
        
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_samples)
        
        q_flow_targets = qmc.scale(
            samples[:, 0:1],
            input_ranges['Q_flow'][0] * 1000,
            input_ranges['Q_flow'][1] * 1000
        ).flatten()
        
        t_air_targets = qmc.scale(
            samples[:, 1:2],
            input_ranges['T_Air'][0],
            input_ranges['T_Air'][1]
        ).flatten()
        
        t_ext_targets = qmc.scale(
            samples[:, 2:3],
            input_ranges['T_ext'][0],
            input_ranges['T_ext'][1]
        ).flatten()
        
        target_indices = np.linspace(0, n_samples - 1, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, total_timesteps)
        
        q_flow_series = np.interp(sample_indices, target_indices, q_flow_targets)
        t_air_series = np.interp(sample_indices, target_indices, t_air_targets)
        t_ext_series = np.interp(sample_indices, target_indices, t_ext_targets)
        
        input_data_rows = []
        for i in range(total_timesteps):
            row = {
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext': t_ext_series[i]
            }
            
            for cdu_idx in range(1, self.num_cdus + 1):
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = q_flow_series[i]
                row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = t_air_series[i]
            
            input_data_rows.append(row)
        
        return pd.DataFrame(input_data_rows)
    
    def generate_response_surface_data(
        self,
        input1: str,
        input2: str,
        input1_range: Tuple[float, float],
        input2_range: Tuple[float, float],
        n_points_per_dim: int = 20,
        fixed_inputs: Optional[Dict[str, float]] = None,
        stabilization_hours: int = 1,  # Reduced default - 1 hour is usually sufficient
        step_size: int = 1,
        steady_state_steps: int = 300,
        n_workers: int = 8,
        use_parallel: bool = True
    ) -> pd.DataFrame:
        """
        Generate data for response surface analysis.
        
        For response surfaces, we need independent simulations at each grid point.
        Each grid point runs its own stabilization WITH the specific input values,
        then measures the steady-state output.
        
        Args:
            input1: First input variable name ('Q_flow', 'T_Air', or 'T_ext')
            input2: Second input variable name ('Q_flow', 'T_Air', or 'T_ext')
            input1_range: (min, max) range for first input
            input2_range: (min, max) range for second input
            n_points_per_dim: Number of grid points per dimension
            fixed_inputs: Fixed values for other inputs (in proper units: Q_flow in W, T in K)
            stabilization_hours: Hours for stabilization at each grid point
            step_size: Simulation step size in seconds
            steady_state_steps: Steps to run after stabilization for averaging
            n_workers: Number of parallel workers (FMU instances)
            use_parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with response surface data
        """
        logger.info(f"Generating response surface data: {input1} x {input2}")
        logger.info(f"  Grid: {n_points_per_dim} x {n_points_per_dim} = {n_points_per_dim**2} points")
        logger.info(f"  Stabilization: {stabilization_hours} hours per point")
        logger.info(f"  Parallel workers: {n_workers if use_parallel else 1}")
        
        if fixed_inputs is None:
            fixed_inputs = {
                'Q_flow': 100.0 * 1000,  # 100 kW in W
                'T_Air': 298.15,          # 25°C in K
                'T_ext': 298.15           # 25°C in K
            }
        
        # Scale input ranges appropriately
        # Q_flow comes in kW but FMU expects W
        if input1 == 'Q_flow':
            input1_range_scaled = (input1_range[0] * 1000, input1_range[1] * 1000)
        else:
            input1_range_scaled = input1_range
            
        if input2 == 'Q_flow':
            input2_range_scaled = (input2_range[0] * 1000, input2_range[1] * 1000)
        else:
            input2_range_scaled = input2_range
        
        # Generate grid points
        input1_vals = np.linspace(input1_range_scaled[0], input1_range_scaled[1], n_points_per_dim)
        input2_vals = np.linspace(input2_range_scaled[0], input2_range_scaled[1], n_points_per_dim)
        
        # Create list of all grid points with their arguments
        grid_args = []
        for idx, (val1, val2) in enumerate([(v1, v2) for v1 in input1_vals for v2 in input2_vals]):
            grid_args.append((idx, val1, val2, input1, input2, fixed_inputs))
        
        total_points = len(grid_args)
        logger.info(f"  Total grid points to simulate: {total_points}")
        
        if use_parallel and n_workers > 1:
            results_df = self._run_parallel_response_surface(
                grid_args=grid_args,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                steady_state_steps=steady_state_steps,
                n_workers=n_workers
            )
        else:
            results_df = self._run_sequential_response_surface(
                grid_args=grid_args,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                steady_state_steps=steady_state_steps
            )
        
        logger.info(f"Successfully generated {len(results_df)} response surface samples")
        return results_df
    
    def _run_parallel_response_surface(
        self,
        grid_args: List[Tuple],
        stabilization_hours: int,
        step_size: int,
        steady_state_steps: int,
        n_workers: int
    ) -> pd.DataFrame:
        """
        Run response surface simulations in parallel.
        Each worker creates its own FMU instance.
        
        Args:
            grid_args: List of (idx, val1, val2, input1, input2, fixed_inputs) tuples
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            steady_state_steps: Steps after stabilization
            n_workers: Number of parallel workers
            
        Returns:
            DataFrame with combined results
        """
        total_points = len(grid_args)
        logger.info(f"Running {total_points} grid points in parallel with {n_workers} workers...")
        
        # Create partial function with fixed arguments
        worker_fn = partial(
            _simulate_grid_point,
            system_name=self.system_name,
            config_overrides=self.config_overrides,
            num_cdus=self.num_cdus,
            stabilization_hours=stabilization_hours,
            step_size=step_size,
            steady_state_steps=steady_state_steps
        )
        
        # Use spawn context for clean process creation (important for FMU)
        ctx = mp.get_context('spawn')
        
        all_results = []
        completed = 0
        
        # Process in batches to manage memory and provide progress updates
        batch_size = min(n_workers * 4, total_points)
        
        with ctx.Pool(processes=n_workers) as pool:
            for batch_start in range(0, total_points, batch_size):
                batch_end = min(batch_start + batch_size, total_points)
                batch_args = grid_args[batch_start:batch_end]
                
                # Use imap for ordered results with progress tracking
                batch_results = list(pool.imap(worker_fn, batch_args))
                
                # Collect successful results
                for result in batch_results:
                    if result is not None:
                        all_results.append(result)
                
                completed += len(batch_args)
                successful = len(all_results)
                logger.info(
                    f"Progress: {completed}/{total_points} points processed, "
                    f"{successful} successful ({successful/completed*100:.1f}%)"
                )
        
        if not all_results:
            logger.error("All grid points failed!")
            return pd.DataFrame()
        
        # Combine results
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by grid index to maintain grid order
        if '_grid_idx' in results_df.columns:
            results_df = results_df.sort_values('_grid_idx').reset_index(drop=True)
        
        return results_df
    
    def _run_sequential_response_surface(
        self,
        grid_args: List[Tuple],
        stabilization_hours: int,
        step_size: int,
        steady_state_steps: int
    ) -> pd.DataFrame:
        """
        Run response surface simulations sequentially.
        
        Args:
            grid_args: List of (idx, val1, val2, input1, input2, fixed_inputs) tuples
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            steady_state_steps: Steps after stabilization
            
        Returns:
            DataFrame with combined results
        """
        total_points = len(grid_args)
        logger.info(f"Running {total_points} grid points sequentially...")
        
        all_results = []
        stabilization_steps = int(stabilization_hours * 3600 / step_size)
        total_steps = stabilization_steps + steady_state_steps
        
        for args in grid_args:
            idx, val1, val2, input1, input2, fixed_inputs = args
            
            if idx % 10 == 0:
                logger.info(f"Processing grid point {idx + 1}/{total_points}")
            
            inputs_dict = fixed_inputs.copy()
            inputs_dict[input1] = val1
            inputs_dict[input2] = val2
            
            # Create input rows for ENTIRE simulation (stabilization + measurement)
            # All rows have the SAME input values - we want steady state at THIS operating point
            input_rows = []
            for _ in range(total_steps):
                row = {}
                row['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = inputs_dict['T_ext']
                
                for cdu_idx in range(1, self.num_cdus + 1):
                    row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'] = inputs_dict['Q_flow']
                    row[f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'] = inputs_dict['T_Air']
                
                input_rows.append(row)
            
            # Create DataFrame for this grid point
            input_df = pd.DataFrame(input_rows)
            
            try:
                # Reset state and run without separate stabilization
                # (stabilization is included in our input data)
                result = self.simulator.run_simulation(
                    input_data=input_df,
                    stabilization_hours=0,  # Already included in input_df
                    step_size=step_size,
                    save_history=False,
                    reset_state=True
                )
                
                if not result.empty and len(result) >= steady_state_steps:
                    # Take the last steady_state_steps rows and average
                    last_n = min(steady_state_steps, len(result))
                    avg_result = result.iloc[-last_n:].mean().to_frame().T
                    
                    # Add metadata
                    avg_result['_grid_idx'] = idx
                    avg_result[f'_input_{input1}'] = val1
                    avg_result[f'_input_{input2}'] = val2
                    
                    all_results.append(avg_result)
                    
            except Exception as e:
                logger.warning(f"Grid point ({val1}, {val2}) failed: {e}")
                continue
        
        if not all_results:
            logger.error("All grid points failed!")
            return pd.DataFrame()
        
        results_df = pd.concat(all_results, ignore_index=True)
        return results_df
    
    def generate_multi_surface_data(
        self,
        input_pairs: List[Tuple[str, str]],
        input_ranges: Dict[str, Tuple[float, float]],
        n_points_per_dim: int = 15,
        fixed_inputs: Optional[Dict[str, float]] = None,
        stabilization_hours: int = 3,
        step_size: int = 1,
        steady_state_steps: int = 1800,
        n_workers: int = 8
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Generate response surface data for multiple input pairs in parallel.
        
        Args:
            input_pairs: List of (input1, input2) tuples to analyze
            input_ranges: Dictionary of input ranges
            n_points_per_dim: Grid points per dimension
            fixed_inputs: Fixed values for non-varied inputs
            stabilization_hours: Hours for stabilization
            step_size: Simulation step size
            steady_state_steps: Steps after stabilization
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping input pairs to their response surface DataFrames
        """
        logger.info(f"Generating {len(input_pairs)} response surfaces in parallel...")
        
        results = {}
        
        # Distribute workers across input pairs
        workers_per_surface = max(1, n_workers // len(input_pairs))
        
        for input1, input2 in input_pairs:
            logger.info(f"Generating surface for {input1} x {input2}...")
            
            surface_data = self.generate_response_surface_data(
                input1=input1,
                input2=input2,
                input1_range=input_ranges[input1],
                input2_range=input_ranges[input2],
                n_points_per_dim=n_points_per_dim,
                fixed_inputs=fixed_inputs,
                stabilization_hours=stabilization_hours,
                step_size=step_size,
                steady_state_steps=steady_state_steps,
                n_workers=workers_per_surface,
                use_parallel=True
            )
            
            results[(input1, input2)] = surface_data
        
        return results