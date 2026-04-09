"""
CDU Comparative Analysis - Main Runner Script.

Orchestrates the complete CDU-level comparative analysis workflow:
- Phase 1: Data Generation (standardized inputs)
- Phase 2: Static Analysis (response surfaces, sensitivity, correlations)
- Phase 3: Dynamic Analysis (step/impulse response, time constants)
- Phase 4: Transfer Function Analysis (gains, coupling, stability)
- Phase 5: Operating Regime Analysis (efficiency, constraints)
- Phase 6: Visualization and Reporting
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for CDU comparative analysis."""
    
    # Models to compare
    models: List[str] = field(default_factory=lambda: ["marconi100", "summit", "lassen"])
    
    # CDU selection
    cdu_ids: Optional[List[int]] = None  # None = auto-select representative CDUs
    n_representative_cdus: int = 5
    
    # Data generation
    generate_data: bool = True
    n_grid_points: int = 10
    n_lhs_samples: int = 500
    step_duration: int = 600
    ramp_duration: int = 300
    stabilization_hours: int = 2
    run_fmu_simulation: bool = True  # Whether to run FMU simulation
    
    # Input ranges
    input_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Q_flow': (12.0, 40.0),
        'T_Air': (288.15, 308.15),
        'T_ext': (283.15, 313.15)
    })
    
    # Analysis settings
    run_static_analysis: bool = True
    run_dynamic_analysis: bool = True
    run_transfer_function: bool = True
    run_operating_regime: bool = True
    
    # Visualization
    create_visualizations: bool = True
    create_report: bool = True
    
    # Output
    output_dir: str = "analysis_results/cdu_comparison"
    random_seed: int = 42
    
    # Parallel processing
    n_workers: int = 4


# System configurations
SYSTEM_CONFIGS = {
    "marconi100": {
        "num_cdus": 49,
        "description": "Marconi100 - 49 CDUs"
    },
    "summit": {
        "num_cdus": 257,
        "description": "Summit - 257 CDUs"
    },
    "lassen": {
        "num_cdus": 257,
        "description": "Lassen - 257 CDUs"
    },
    "frontier": {
        "num_cdus": 128,
        "description": "Frontier - 128 CDUs"
    }
}
def convert_for_json(obj):
    """Convert non-serializable objects for JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (datetime, timedelta)):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # For objects with __dict__, convert to dict but avoid circular refs
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        # Recursively handle dicts
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively handle lists/tuples
        return [convert_for_json(item) for item in obj]
    else:
        try:
            return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"


class CDUComparativeAnalysis:
    """
    Main class for CDU-level comparative analysis.
    
    Implements the complete analysis workflow:
    1. Data Generation
    2. Static Analysis
    3. Dynamic Analysis
    4. Transfer Function Analysis
    5. Operating Regime Analysis
    6. Visualization and Reporting
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the analysis runner."""
        self.config = config
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Determine CDU IDs to analyze
        self.cdu_ids = config.cdu_ids or self._select_representative_cdus()
        
        # Initialize analyzers
        self._init_analyzers()
        
        # Initialize visualizers
        self._init_visualizers()
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {
            "config": self._config_to_dict(),
            "models": config.models,
            "cdu_ids": self.cdu_ids,
            "static": {},
            "dynamic": {},
            "transfer_function": {},
            "operating_regime": {},
            "comparison": {}
        }
        
        logger.info(f"CDUComparativeAnalysis initialized")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  CDUs: {self.cdu_ids}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for serialization."""
        return {
            "models": self.config.models,
            "cdu_ids": self.config.cdu_ids,
            "n_representative_cdus": self.config.n_representative_cdus,
            "n_grid_points": self.config.n_grid_points,
            "n_lhs_samples": self.config.n_lhs_samples,
            "input_ranges": {k: list(v) for k, v in self.config.input_ranges.items()},
            "random_seed": self.config.random_seed
        }
    
    def _select_representative_cdus(self) -> List[int]:
        """Select representative CDU IDs for analysis."""
        # Find max CDUs across models
        max_cdus = max(SYSTEM_CONFIGS.get(m, {}).get("num_cdus", 1) 
                       for m in self.config.models)
        
        n = min(max_cdus, self.config.n_representative_cdus)
        
        if max_cdus <= n:
            return list(range(1, max_cdus + 1))
        
        # Select evenly spaced CDUs
        indices = np.linspace(1, max_cdus, n, dtype=int)
        return list(indices)
    
    def _init_analyzers(self) -> None:
        """Initialize all analyzers."""
        # Import analyzers
        from ..analyzers import (
            CDUResponseAnalyzer,
            SensitivityAnalyzer, SensitivityConfig,
            DynamicResponseAnalyzer, DynamicConfig,
            TransferFunctionAnalyzer, TransferFunctionConfig,
            OperatingRegimeAnalyzer, OperatingRegimeConfig
        )
        
        # CDU Response Analyzers (one per model)
        self.cdu_analyzers: Dict[str, CDUResponseAnalyzer] = {}
        for model in self.config.models:
            try:
                self.cdu_analyzers[model] = CDUResponseAnalyzer(system_name=model)
            except Exception as e:
                logger.warning(f"Could not initialize CDUResponseAnalyzer for {model}: {e}")
        
        # Sensitivity Analyzer
        self.sensitivity_analyzer = SensitivityAnalyzer(SensitivityConfig())
        
        # Dynamic Response Analyzer
        self.dynamic_analyzer = DynamicResponseAnalyzer(DynamicConfig())
        
        # Transfer Function Analyzer
        self.tf_analyzer = TransferFunctionAnalyzer(TransferFunctionConfig())
        
        # Operating Regime Analyzer
        self.regime_analyzer = OperatingRegimeAnalyzer(OperatingRegimeConfig())
        
        # FMU Simulators (initialized lazily)
        self._simulators: Dict[str, Any] = {}
        
        logger.debug("Analyzers initialized")
    
    def _init_visualizers(self) -> None:
        """Initialize all visualizers."""
        from ..visualizers import CDUComparisonVisualizer
        from ..visualizers.response_surface_visualizer import ResponseSurfaceVisualizer
        from ..visualizers.model_comparison_dashboard import ModelComparisonDashboard
        from ..visualizers.simple_comparison_visualizer import SimpleComparisonVisualizer
        
        plots_dir = self.output_dir / "plots"
        
        self.cdu_visualizer = CDUComparisonVisualizer(output_dir=plots_dir / "cdu")
        self.surface_visualizer = ResponseSurfaceVisualizer(output_dir=plots_dir / "surfaces")
        self.dashboard = ModelComparisonDashboard(output_dir=plots_dir / "dashboards")
        self.simple_visualizer = SimpleComparisonVisualizer(output_dir=plots_dir / "comparison")
        
        # Store plots directory for StandardInputOutputVisualizer (initialized per-model)
        self.io_plots_dir = plots_dir / "io_plots"
        self.io_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize physics validator
        from ..analyzers import PhysicsConstraintValidator
        self.physics_validator = PhysicsConstraintValidator()
        
        logger.debug("Visualizers initialized")
    
    def _get_simulator(self, model: str):
        """
        Get or create FMU simulator for a model.
        
        Args:
            model: Model/system name (e.g., 'marconi100', 'summit')
            
        Returns:
            FMUSimulator instance
        """
        if model not in self._simulators:
            try:
                from fmu2ml.simulation.fmu_simulator import FMUSimulator
                self._simulators[model] = FMUSimulator(system_name=model)
                logger.info(f"Initialized FMU simulator for {model}")
            except Exception as e:
                logger.error(f"Failed to initialize FMU simulator for {model}: {e}")
                raise
        return self._simulators[model]
    
    def run_fmu_simulation(
        self,
        model: str,
        input_data: pd.DataFrame,
        stabilization_hours: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run FMU simulation for a model with given inputs.
        
        Args:
            model: Model/system name
            input_data: DataFrame with input columns
            stabilization_hours: Hours for stabilization (None uses config)
            
        Returns:
            DataFrame with both inputs and outputs from FMU
        """
        if stabilization_hours is None:
            stabilization_hours = self.config.stabilization_hours
            
        simulator = self._get_simulator(model)
        
        logger.info(f"Running FMU simulation for {model} ({len(input_data)} samples)...")
        
        try:
            # Reset simulator to initial state
            simulator.reset()
            
            # Run the simulation
            output_data = simulator.run_simulation(
                input_data=input_data,
                stabilization_hours=stabilization_hours,
                step_size=1,
                save_history=False
            )
            
            logger.info(f"FMU simulation complete for {model}: {output_data.shape}")
            return output_data
            
        except Exception as e:
            logger.error(f"FMU simulation failed for {model}: {e}")
            raise
    
    def cleanup_simulators(self) -> None:
        """Clean up all FMU simulator resources."""
        for model, simulator in self._simulators.items():
            try:
                simulator.cleanup()
                logger.debug(f"Cleaned up simulator for {model}")
            except Exception as e:
                logger.warning(f"Error cleaning up simulator for {model}: {e}")
        self._simulators.clear()

    # =========================================================================
    # Phase 1: Data Generation
    # =========================================================================
    
    def generate_standardized_data(
        self,
        simulator_func: Optional[callable] = None
    ) -> None:
        """
        Generate standardized input data for all models.
        
        Creates identical input sequences for fair comparison.
        
        Args:
            simulator_func: Optional function to run FMU simulation
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Generation")
        logger.info("=" * 60)
        
        from ..data_generators import (
            StandardizedInputGenerator,
            StepInputGenerator,
            RampInputGenerator,
            GridInputGenerator
        )
        
        for model in self.config.models:
            logger.info(f"Generating data for {model}")
            
            # Initialize generators
            try:
                input_gen = StandardizedInputGenerator(
                    system_name=model,
                    input_ranges=self.config.input_ranges,
                    random_seed=self.config.random_seed
                )
                step_gen = StepInputGenerator(
                    system_name=model,
                    input_ranges=self.config.input_ranges
                )
                grid_gen = GridInputGenerator(
                    system_name=model,
                    input_ranges=self.config.input_ranges
                )
            except Exception as e:
                logger.error(f"Failed to initialize generators for {model}: {e}")
                continue
            
            # Generate LHS samples for sensitivity analysis
            lhs_inputs = input_gen.generate_lhs_inputs(
                n_samples=self.config.n_lhs_samples,
                uniform_across_cdus=True
            )
            
            # Generate step inputs for dynamic analysis
            step_inputs_list = []
            for input_var in ['Q_flow', 'T_Air', 'T_ext']:
                lo, hi = self.config.input_ranges[input_var]
                step_inputs = step_gen.generate_step_sequence(
                    target_input=input_var,
                    step_from=lo,
                    step_to=hi,
                    pre_step_duration=self.config.step_duration,
                    post_step_duration=self.config.step_duration * 2
                )
                step_inputs_list.append(step_inputs)
            
            # Generate grid inputs for response surface
            grid_inputs = grid_gen.generate_2d_grid(
                var1='Q_flow',
                var2='T_Air',
                n_points=self.config.n_grid_points,
                transition_steps=60
            )
            
            # Combine all inputs
            all_inputs = pd.concat([lhs_inputs] + step_inputs_list + [grid_inputs], 
                                   ignore_index=True)
            
            # Run FMU simulation to get outputs
            if self.config.run_fmu_simulation:
                logger.info(f"Running FMU simulation for {model}...")
                try:
                    output = self.run_fmu_simulation(model, all_inputs)
                    self.data[model] = output
                    logger.info(f"Simulation complete for {model}: {output.shape}")
                except Exception as e:
                    logger.error(f"FMU simulation failed for {model}: {e}")
                    logger.warning(f"Storing input-only data for {model}")
                    self.data[model] = all_inputs
            elif simulator_func:
                # Use custom simulator function if provided
                logger.info(f"Running custom simulation for {model}...")
                try:
                    output = simulator_func(model, all_inputs)
                    self.data[model] = output
                except Exception as e:
                    logger.error(f"Custom simulation failed for {model}: {e}")
                    self.data[model] = all_inputs
            else:
                # Store input data only (for later simulation)
                self.data[model] = all_inputs
                logger.warning(f"Inputs only for {model}: {len(all_inputs)} samples (no simulation)")
        
        # Save generated data
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for model, df in self.data.items():
            # Save with appropriate suffix based on content
            has_outputs = any(col.startswith("simulator[") for col in df.columns)
            suffix = "_simulation.parquet" if has_outputs else "_inputs.parquet"
            df.to_parquet(data_dir / f"{model}{suffix}", index=False)
            logger.info(f"Saved {model} data: {df.shape}")
        
        # Cleanup simulators to free resources
        self.cleanup_simulators()
    
    def load_data(self, data_paths: Dict[str, str]) -> None:
        """
        Load existing simulation data.
        
        Args:
            data_paths: Dict mapping model name to data file path
        """
        logger.info("Loading simulation data")
        
        for model, path in data_paths.items():
            path = Path(path)
            if not path.exists():
                logger.warning(f"Data file not found: {path}")
                continue
            
            if path.suffix == '.parquet':
                self.data[model] = pd.read_parquet(path)
            elif path.suffix == '.csv':
                self.data[model] = pd.read_csv(path)
            else:
                logger.warning(f"Unsupported file format: {path}")
                continue
            
            logger.info(f"Loaded {model}: {self.data[model].shape}")
    
    # =========================================================================
    # Phase 2: Static Analysis
    # =========================================================================
    
    def run_static_analysis(self) -> Dict[str, Any]:
        """
        Run static analysis: response surfaces, sensitivity, correlations.
        
        Returns:
            Static analysis results
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Static Analysis")
        logger.info("=" * 60)
        
        results = {
            "response_statistics": {},
            "correlations": {},
            "static_gains": {},
            "nonlinearity": {},
            "sensitivity_matrices": {},
            "sensitivity_rankings": {},
            "sobol_indices": {},
            "response_surfaces": {}
        }
        
        for model in self.config.models:
            if model not in self.data:
                logger.warning(f"No data for {model}, skipping")
                continue
            
            data = self.data[model]
            results["response_statistics"][model] = {}
            results["correlations"][model] = {}
            results["static_gains"][model] = {}
            results["nonlinearity"][model] = {}
            results["sensitivity_matrices"][model] = {}
            results["sensitivity_rankings"][model] = {}
            results["sobol_indices"][model] = {}
            results["response_surfaces"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Static analysis: {model} CDU {cdu_id}")
                
                # CDU Response Analysis
                if model in self.cdu_analyzers:
                    analyzer = self.cdu_analyzers[model]
                    
                    # A1: Response statistics
                    stats = analyzer.compute_response_statistics(data, cdu_id)
                    results["response_statistics"][model][cdu_id] = stats
                    
                    # A3: I/O correlations
                    corr_df = analyzer.compute_io_correlations(data, cdu_id)
                    results["correlations"][model][cdu_id] = corr_df.to_dict('records')
                    
                    # Static gains (DC gains)
                    gains_df = analyzer.compute_static_gains(data, cdu_id)
                    results["static_gains"][model][cdu_id] = gains_df.to_dict('records')
                    
                    # A4: Nonlinearity characterization
                    nonlin_df = analyzer.compute_nonlinearity_index(data, cdu_id)
                    results["nonlinearity"][model][cdu_id] = nonlin_df.to_dict('records')
                    
                    # A1: Response surface (for visualization)
                    for output_var in ['T_sec_r_C', 'W_flow_CDUP_kW', 'V_flow_prim_GPM']:
                        surface = analyzer.compute_response_surface_samples(
                            data, cdu_id,
                            input_pair=('Q_flow', 'T_Air'),
                            output_var=output_var,
                            n_bins=self.config.n_grid_points
                        )
                        if surface:
                            key = f"{output_var}_Q_T"
                            results["response_surfaces"][model].setdefault(cdu_id, {})[key] = surface
                
                # A2: Sensitivity analysis
                try:
                    sens_matrix, ops = self.sensitivity_analyzer.compute_sensitivity_matrix(
                        data, cdu_id, model, n_sample_points=20
                    )
                    results["sensitivity_matrices"][model][cdu_id] = sens_matrix.tolist()
                    
                    rankings = self.sensitivity_analyzer.rank_sensitivities(sens_matrix)
                    results["sensitivity_rankings"][model][cdu_id] = rankings
                    
                    # Sobol indices
                    first_order, total_order = self.sensitivity_analyzer.compute_sobol_indices(
                        data, cdu_id, model
                    )
                    results["sobol_indices"][model][cdu_id] = {
                        "first_order": first_order,
                        "total_order": total_order
                    }
                except Exception as e:
                    logger.warning(f"Sensitivity analysis failed for {model} CDU {cdu_id}: {e}")
        
        self.results["static"] = results
        return results
    
    # =========================================================================
    # Phase 3: Dynamic Analysis
    # =========================================================================
    
    def run_dynamic_analysis(self) -> Dict[str, Any]:
        """
        Run dynamic analysis: step response, impulse response, time constants.
        
        Returns:
            Dynamic analysis results
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Dynamic Analysis")
        logger.info("=" * 60)
        
        results = {
            "step_response": {},
            "time_constants": {},
            "delays": {},
            "frequency_response": {}
        }
        
        for model in self.config.models:
            if model not in self.data:
                continue
            
            data = self.data[model]
            results["step_response"][model] = {}
            results["time_constants"][model] = {}
            results["delays"][model] = {}
            results["frequency_response"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Dynamic analysis: {model} CDU {cdu_id}")
                
                try:
                    # Analyze from data
                    dynamic_results = self.dynamic_analyzer.analyze_from_data(
                        data, cdu_id, model, time_col="time"
                    )
                    
                    # Extract time constants
                    tc_dict = {}
                    step_dict = {}
                    delay_dict = {}
                    
                    for input_var, outputs in dynamic_results.items():
                        for output_var, result in outputs.items():
                            key = f"{input_var}->{output_var}"
                            tc_dict[key] = result.dominant_time_constant
                            delay_dict[key] = result.delay
                            
                            if result.step_response is not None:
                                step_dict[output_var] = {
                                    "response": result.step_response.tolist(),
                                    "rise_time": result.rise_time,
                                    "settling_time": result.settling_time,
                                    "overshoot": result.overshoot,
                                    "steady_state_gain": result.steady_state_gain
                                }
                    
                    results["time_constants"][model][cdu_id] = tc_dict
                    results["step_response"][model][cdu_id] = step_dict
                    results["delays"][model][cdu_id] = delay_dict
                    
                except Exception as e:
                    logger.warning(f"Dynamic analysis failed for {model} CDU {cdu_id}: {e}")
        
        self.results["dynamic"] = results
        return results
    
    # =========================================================================
    # Phase 4: Transfer Function Analysis
    # =========================================================================
    
    def run_transfer_function_analysis(self) -> Dict[str, Any]:
        """
        Run transfer function analysis: gains, coupling, stability.
        
        Returns:
            Transfer function analysis results
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: Transfer Function Analysis")
        logger.info("=" * 60)
        
        results = {
            "dc_gain_matrices": {},
            "transfer_functions": {},
            "rga": {},
            "coupling": {},
            "stability": {}
        }
        
        for model in self.config.models:
            if model not in self.data:
                continue
            
            data = self.data[model]
            results["dc_gain_matrices"][model] = {}
            results["transfer_functions"][model] = {}
            results["rga"][model] = {}
            results["coupling"][model] = {}
            results["stability"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Transfer function analysis: {model} CDU {cdu_id}")
                
                try:
                    # Complete CDU analysis
                    tf_result = self.tf_analyzer.analyze_cdu(data, cdu_id, model)
                    
                    # DC gain matrix
                    if tf_result.dc_gain_matrix is not None:
                        results["dc_gain_matrices"][model][cdu_id] = tf_result.dc_gain_matrix.tolist()
                        
                        # RGA
                        rga = self.tf_analyzer.compute_relative_gain_array(tf_result.dc_gain_matrix)
                        results["rga"][model][cdu_id] = rga.tolist()
                        
                        # Coupling analysis
                        coupling = self.tf_analyzer.analyze_coupling(tf_result.dc_gain_matrix)
                        results["coupling"][model][cdu_id] = coupling
                    
                    # Transfer function parameters
                    results["transfer_functions"][model][cdu_id] = tf_result.transfer_functions
                    
                    # Stability analysis
                    results["stability"][model][cdu_id] = tf_result.stability_analysis
                    
                except Exception as e:
                    logger.warning(f"TF analysis failed for {model} CDU {cdu_id}: {e}")
        
        self.results["transfer_function"] = results
        return results
    
    # =========================================================================
    # Phase 5: Operating Regime Analysis
    # =========================================================================
    
    def run_operating_regime_analysis(self) -> Dict[str, Any]:
        """
        Run operating regime analysis: efficiency, envelope, constraints.
        
        Returns:
            Operating regime analysis results
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: Operating Regime Analysis")
        logger.info("=" * 60)
        
        results = {
            "thermal_efficiency": {},
            "pumping_efficiency": {},
            "operating_envelope": {},
            "constraints": {},
            "pue": {}
        }
        
        for model in self.config.models:
            if model not in self.data:
                continue
            
            data = self.data[model]
            results["thermal_efficiency"][model] = {}
            results["pumping_efficiency"][model] = {}
            results["operating_envelope"][model] = {}
            results["constraints"][model] = {}
            results["pue"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Operating regime analysis: {model} CDU {cdu_id}")
                
                try:
                    # Complete CDU analysis
                    regime_result = self.regime_analyzer.analyze_cdu(data, cdu_id, model)
                    
                    results["thermal_efficiency"][model][cdu_id] = regime_result.thermal_efficiency
                    results["pumping_efficiency"][model][cdu_id] = regime_result.pumping_efficiency
                    results["pue"][model][cdu_id] = regime_result.pue_stats
                    
                    # Operating envelope
                    envelope = self.regime_analyzer.compute_operating_envelope(data, cdu_id)
                    results["operating_envelope"][model][cdu_id] = envelope
                    
                    # Constraints
                    constraints = self.regime_analyzer.detect_constraint_boundaries(data, cdu_id)
                    results["constraints"][model][cdu_id] = constraints
                    
                except Exception as e:
                    logger.warning(f"Regime analysis failed for {model} CDU {cdu_id}: {e}")
        
        self.results["operating_regime"] = results
        return results
    
    # =========================================================================
    # Phase 5.5: Physics Constraint Validation
    # =========================================================================
    
    def run_physics_validation(self) -> Dict[str, Any]:
        """
        Validate physics constraints across all models.
        
        This checks if the physics constraints used for neural operator
        training hold consistently across different cooling architectures.
        
        Returns:
            Physics validation results for all models
        """
        logger.info("=" * 60)
        logger.info("PHASE 5.5: Physics Constraint Validation")
        logger.info("=" * 60)
        
        if not self.data:
            logger.warning("No data available for physics validation")
            return {}
        
        # Run comparison across all models
        physics_results = self.physics_validator.compare_models(self.data)
        
        # Log summary
        for row in physics_results.get('summary_table', []):
            model = row.get('model', 'Unknown')
            pass_rate = row.get('pass_rate', 0)
            logger.info(f"  {model}: {pass_rate*100:.0f}% constraints passed")
        
        # Store results
        self.results["physics_validation"] = physics_results
        
        return physics_results
    
    # =========================================================================
    # Phase 6: Cross-Model Comparison
    # =========================================================================
    
    def compare_across_models(self) -> Dict[str, Any]:
        """
        Compare results across all models.
        
        Returns:
            Cross-model comparison results
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Cross-Model Comparison")
        logger.info("=" * 60)
        
        comparison = {
            "sensitivity_comparison": {},
            "gain_comparison": {},
            "dynamics_comparison": {},
            "efficiency_comparison": {},
            "key_findings": []
        }
        
        for cdu_id in self.cdu_ids:
            logger.info(f"Cross-model comparison for CDU {cdu_id}")
            
            cdu_comparison = {
                "sensitivity": {},
                "gains": {},
                "time_constants": {},
                "efficiency": {}
            }
            
            # Compare sensitivities
            for model in self.config.models:
                static = self.results.get("static", {})
                sens = static.get("sensitivity_matrices", {}).get(model, {}).get(cdu_id)
                if sens:
                    cdu_comparison["sensitivity"][model] = sens
            
            # Compare DC gains
            for model in self.config.models:
                tf = self.results.get("transfer_function", {})
                gains = tf.get("dc_gain_matrices", {}).get(model, {}).get(cdu_id)
                if gains:
                    cdu_comparison["gains"][model] = gains
            
            # Compare time constants
            for model in self.config.models:
                dyn = self.results.get("dynamic", {})
                tc = dyn.get("time_constants", {}).get(model, {}).get(cdu_id)
                if tc:
                    cdu_comparison["time_constants"][model] = tc
            
            # Compare efficiency
            for model in self.config.models:
                regime = self.results.get("operating_regime", {})
                eff = regime.get("pumping_efficiency", {}).get(model, {}).get(cdu_id)
                if eff:
                    cdu_comparison["efficiency"][model] = eff
            
            comparison["sensitivity_comparison"][cdu_id] = cdu_comparison["sensitivity"]
            comparison["gain_comparison"][cdu_id] = cdu_comparison["gains"]
            comparison["dynamics_comparison"][cdu_id] = cdu_comparison["time_constants"]
            comparison["efficiency_comparison"][cdu_id] = cdu_comparison["efficiency"]
        
        # Generate key findings
        comparison["key_findings"] = self._generate_key_findings()
        
        self.results["comparison"] = comparison
        return comparison
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from analysis."""
        findings = []
        
        # Sensitivity findings
        static = self.results.get("static", {})
        if static.get("sensitivity_rankings"):
            findings.append("• Sensitivity analysis completed for all models")
        
        # Time constant findings
        dynamic = self.results.get("dynamic", {})
        if dynamic.get("time_constants"):
            # Find fastest and slowest models
            avg_tc = {}
            for model, cdus in dynamic["time_constants"].items():
                all_tc = []
                for cdu, pairs in cdus.items():
                    all_tc.extend([v for v in pairs.values() if v > 0])
                if all_tc:
                    avg_tc[model] = np.mean(all_tc)
            
            if avg_tc:
                fastest = min(avg_tc, key=avg_tc.get)
                slowest = max(avg_tc, key=avg_tc.get)
                findings.append(f"• Fastest response: {fastest} (avg τ = {avg_tc[fastest]:.1f}s)")
                findings.append(f"• Slowest response: {slowest} (avg τ = {avg_tc[slowest]:.1f}s)")
        
        # Efficiency findings
        regime = self.results.get("operating_regime", {})
        if regime.get("pumping_efficiency"):
            avg_cop = {}
            for model, cdus in regime["pumping_efficiency"].items():
                cops = []
                for cdu, eff in cdus.items():
                    if isinstance(eff, dict) and "cop_mean" in eff:
                        cops.append(eff["cop_mean"])
                if cops:
                    avg_cop[model] = np.mean(cops)
            
            if avg_cop:
                best = max(avg_cop, key=avg_cop.get)
                findings.append(f"• Highest COP: {best} ({avg_cop[best]:.2f})")
        
        if not findings:
            findings.append("• Analysis completed - see detailed results")
        
        return findings
    
    # =========================================================================
    # Phase 7: Visualization
    # =========================================================================
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        if not self.config.create_visualizations:
            return
        
        logger.info("=" * 60)
        logger.info("PHASE 7: Visualization")
        logger.info("=" * 60)
        
        for cdu_id in self.cdu_ids:
            logger.info(f"Generating visualizations for CDU {cdu_id}")
            
            # Sensitivity heatmaps
            sens_matrices = {}
            for model in self.config.models:
                sens = self.results.get("static", {}).get("sensitivity_matrices", {}).get(model, {}).get(cdu_id)
                if sens:
                    sens_matrices[model] = np.array(sens)
            
            if sens_matrices:
                self.cdu_visualizer.plot_sensitivity_comparison(sens_matrices, cdu_id)
            
            # DC gain comparison
            gain_matrices = {}
            for model in self.config.models:
                gains = self.results.get("transfer_function", {}).get("dc_gain_matrices", {}).get(model, {}).get(cdu_id)
                if gains:
                    gain_matrices[model] = np.array(gains)
            
            if gain_matrices:
                self.cdu_visualizer.plot_dc_gain_comparison(gain_matrices, cdu_id)
            
            # RGA comparison
            rga_matrices = {}
            for model in self.config.models:
                rga = self.results.get("transfer_function", {}).get("rga", {}).get(model, {}).get(cdu_id)
                if rga:
                    rga_matrices[model] = np.array(rga)
            
            if rga_matrices:
                self.cdu_visualizer.plot_rga_comparison(rga_matrices, cdu_id)
            
            # Time constant comparison
            tc_data = {}
            for model in self.config.models:
                tc = self.results.get("dynamic", {}).get("time_constants", {}).get(model, {}).get(cdu_id)
                if tc:
                    tc_data[model] = tc
            
            if tc_data:
                self.cdu_visualizer.plot_time_constant_comparison(tc_data, cdu_id)
            
            # Sobol indices
            sobol_data = {}
            for model in self.config.models:
                sobol = self.results.get("static", {}).get("sobol_indices", {}).get(model, {}).get(cdu_id)
                if sobol and "total_order" in sobol and sobol["total_order"]:
                    # Check that total_order has actual content
                    total_order = sobol["total_order"]
                    if isinstance(total_order, dict) and len(total_order) > 0:
                        sobol_data[model] = total_order
            
            if sobol_data:
                self.cdu_visualizer.plot_sobol_indices(sobol_data, cdu_id, order="total")
            
            # Response surfaces
            for model in self.config.models:
                surfaces = self.results.get("static", {}).get("response_surfaces", {}).get(model, {}).get(cdu_id, {})
                for key, surface in surfaces.items():
                    if surface:
                        self.surface_visualizer.plot_response_surface_2d(surface, model, cdu_id)
        
        # Create executive summary dashboard
        dashboard_data = self._prepare_dashboard_data()
        self.dashboard.create_executive_summary(dashboard_data)
        
        # Generate simple comparison visualizations (physics, thermal, flow, efficiency)
        self._generate_simple_comparison_visualizations()
        
        # Generate per-model I/O visualizations using StandardInputOutputVisualizer
        self._generate_per_model_io_visualizations()
        
        # Close all figures
        self.cdu_visualizer.close_all()
        self.surface_visualizer.close_all()
        self.dashboard.close_all()
        self.simple_visualizer.close_all()
        
        logger.info("Visualizations complete")
    
    def _generate_simple_comparison_visualizations(self) -> None:
        """Generate simple comparison visualizations for all models."""
        logger.info("Generating simple comparison visualizations...")
        
        # Get physics validation results if available
        physics_results = self.results.get("physics_validation", {})
        
        # Get sensitivity data if available
        sensitivity_data = self.results.get("static", {}).get("sensitivity_matrices", {})
        
        # Get time constants for first CDU across all models
        time_constants = {}
        cdu_id = self.cdu_ids[0] if self.cdu_ids else 1
        for model in self.config.models:
            tc = self.results.get("dynamic", {}).get("time_constants", {}).get(model, {}).get(cdu_id)
            if tc:
                time_constants[model] = tc
        
        # Get normalized metrics for radar chart
        summary_metrics = self._compute_normalized_metrics()
        
        # Generate all visualizations
        saved_files = self.simple_visualizer.generate_all_visualizations(
            data_dict=self.data,
            physics_results=physics_results,
            sensitivity_data=sensitivity_data,
            time_constants=time_constants,
            summary_metrics=summary_metrics
        )
        
        logger.info(f"Generated {len(saved_files)} comparison visualizations")
    
    def _generate_per_model_io_visualizations(self) -> None:
        """
        Generate per-model I/O visualizations using StandardInputOutputVisualizer.
        
        Creates detailed input/output plots for each cooling model showing:
        - Input distributions (Q_flow, T_Air, T_ext)
        - Output distributions (temperatures, flows, pressures, power)
        - Time series for selected CDUs
        """
        logger.info("Generating per-model I/O visualizations...")
        
        try:
            from fmu2ml.visualization.plotters.output_plots import StandardInputOutputVisualizer
        except ImportError:
            logger.warning("StandardInputOutputVisualizer not available, skipping per-model I/O plots")
            return
        
        import matplotlib.pyplot as plt
        
        for model in self.config.models:
            if model not in self.data:
                logger.warning(f"No data for {model}, skipping I/O visualization")
                continue
            
            df = self.data[model]
            logger.info(f"Generating I/O plots for {model}...")
            
            try:
                # Create model-specific output directory
                model_plots_dir = self.io_plots_dir / model
                model_plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize visualizer for this model
                visualizer = StandardInputOutputVisualizer(df)
                visualizer.system_name = model
                
                # Get CDU count from system config
                num_cdus = SYSTEM_CONFIGS.get(model, {}).get('num_cdus', 49)
                visualizer.cdu_count = num_cdus
                
                # Select CDUs to visualize (use configured or representative)
                if self.cdu_ids:
                    # Filter to CDUs that exist in this model
                    cdus_to_plot = [c for c in self.cdu_ids if c <= num_cdus]
                else:
                    cdus_to_plot = visualizer.select_cdus(random_select=min(5, num_cdus))
                
                if not cdus_to_plot:
                    logger.warning(f"No valid CDUs to plot for {model}")
                    continue
                
                # Plot inputs
                try:
                    fig = visualizer.plot_inputs(
                        selected_cdus=cdus_to_plot,
                        rolling_windows=10,
                        figsize=(14, 10)
                    )
                    if fig:
                        fig.savefig(
                            model_plots_dir / f"{model}_inputs.png",
                            dpi=150, bbox_inches='tight'
                        )
                        plt.close(fig)
                        logger.debug(f"Saved {model} inputs plot")
                except Exception as e:
                    logger.warning(f"Could not generate inputs plot for {model}: {e}")
                
                # Plot outputs if method exists
                if hasattr(visualizer, 'plot_outputs'):
                    try:
                        fig = visualizer.plot_outputs(
                            selected_cdus=cdus_to_plot,
                            rolling_windows=10,
                            figsize=(14, 12)
                        )
                        if fig:
                            fig.savefig(
                                model_plots_dir / f"{model}_outputs.png",
                                dpi=150, bbox_inches='tight'
                            )
                            plt.close(fig)
                            logger.debug(f"Saved {model} outputs plot")
                    except Exception as e:
                        logger.warning(f"Could not generate outputs plot for {model}: {e}")
                
                # Plot combined I/O if method exists
                if hasattr(visualizer, 'plot_io_relationships'):
                    try:
                        fig = visualizer.plot_io_relationships(
                            selected_cdus=cdus_to_plot[:3],  # Limit to 3 CDUs for clarity
                            figsize=(16, 10)
                        )
                        if fig:
                            fig.savefig(
                                model_plots_dir / f"{model}_io_relationships.png",
                                dpi=150, bbox_inches='tight'
                            )
                            plt.close(fig)
                            logger.debug(f"Saved {model} I/O relationships plot")
                    except Exception as e:
                        logger.warning(f"Could not generate I/O relationships plot for {model}: {e}")
                
                logger.info(f"Completed I/O plots for {model}")
                
            except Exception as e:
                logger.error(f"Failed to generate I/O visualizations for {model}: {e}")
        
        logger.info("Per-model I/O visualizations complete")
    
    def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare data for dashboard visualization."""
        return {
            "models": self.config.models,
            "n_cdus_analyzed": len(self.cdu_ids),
            "summary_metrics": self._compute_summary_metrics(),
            "normalized_metrics": self._compute_normalized_metrics(),
            "sensitivity_matrices": self._get_first_sensitivity_matrices(),
            "time_constants": self._get_first_time_constants(),
            "efficiency": self._get_efficiency_summary(),
            "key_findings": self.results.get("comparison", {}).get("key_findings", [])
        }
    
    def _compute_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute summary metrics per model."""
        summary = {}
        
        for model in self.config.models:
            summary[model] = {}
            
            # Average sensitivity
            sens = self.results.get("static", {}).get("sensitivity_matrices", {}).get(model, {})
            if sens:
                all_sens = [np.mean(np.abs(np.array(s))) for s in sens.values() if s]
                if all_sens:
                    summary[model]["mean_sensitivity"] = float(np.mean(all_sens))
            
            # Average time constant
            tc = self.results.get("dynamic", {}).get("time_constants", {}).get(model, {})
            if tc:
                all_tc = []
                for cdu, pairs in tc.items():
                    all_tc.extend([v for v in pairs.values() if v > 0])
                if all_tc:
                    summary[model]["dominant_time_constant"] = float(np.mean(all_tc))
            
            # Average COP
            eff = self.results.get("operating_regime", {}).get("pumping_efficiency", {}).get(model, {})
            if eff:
                cops = [e.get("cop_mean", 0) for e in eff.values() if isinstance(e, dict)]
                if cops:
                    summary[model]["cop_mean"] = float(np.mean(cops))
        
        return summary
    
    def _compute_normalized_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute normalized metrics for radar chart."""
        summary = self._compute_summary_metrics()
        normalized = {}
        
        # Get ranges
        metrics = ['mean_sensitivity', 'dominant_time_constant', 'cop_mean']
        ranges = {}
        for metric in metrics:
            values = [s.get(metric, 0) for s in summary.values()]
            if values:
                ranges[metric] = (min(values), max(values))
        
        for model, values in summary.items():
            normalized[model] = {}
            
            # Sensitivity (higher = more responsive)
            if 'mean_sensitivity' in values and ranges.get('mean_sensitivity'):
                lo, hi = ranges['mean_sensitivity']
                if hi > lo:
                    normalized[model]['sensitivity'] = (values['mean_sensitivity'] - lo) / (hi - lo)
                else:
                    normalized[model]['sensitivity'] = 0.5
            
            # Speed (lower time constant = faster = better)
            if 'dominant_time_constant' in values and ranges.get('dominant_time_constant'):
                lo, hi = ranges['dominant_time_constant']
                if hi > lo:
                    normalized[model]['speed'] = 1 - (values['dominant_time_constant'] - lo) / (hi - lo)
                else:
                    normalized[model]['speed'] = 0.5
            
            # Efficiency (higher COP = better)
            if 'cop_mean' in values and ranges.get('cop_mean'):
                lo, hi = ranges['cop_mean']
                if hi > lo:
                    normalized[model]['efficiency'] = (values['cop_mean'] - lo) / (hi - lo)
                else:
                    normalized[model]['efficiency'] = 0.5
            
            # Defaults for missing
            normalized[model].setdefault('stability', 0.5)
            normalized[model].setdefault('coverage', 0.5)
        
        return normalized
    
    def _get_first_sensitivity_matrices(self) -> Dict[str, np.ndarray]:
        """Get sensitivity matrix for first CDU per model."""
        result = {}
        cdu_id = self.cdu_ids[0] if self.cdu_ids else 1
        
        for model in self.config.models:
            sens = self.results.get("static", {}).get("sensitivity_matrices", {}).get(model, {}).get(cdu_id)
            if sens:
                result[model] = np.array(sens)
        
        return result
    
    def _get_first_time_constants(self) -> Dict[str, Dict[str, float]]:
        """Get time constants for first CDU per model."""
        result = {}
        cdu_id = self.cdu_ids[0] if self.cdu_ids else 1
        
        for model in self.config.models:
            tc = self.results.get("dynamic", {}).get("time_constants", {}).get(model, {}).get(cdu_id)
            if tc:
                result[model] = tc
        
        return result
    
    def _get_efficiency_summary(self) -> Dict[str, Dict[str, float]]:
        """Get efficiency summary per model."""
        result = {}
        
        for model in self.config.models:
            eff = self.results.get("operating_regime", {}).get("pumping_efficiency", {}).get(model, {})
            if eff:
                # Average across CDUs
                cop_values = [e.get("cop_mean", 0) for e in eff.values() if isinstance(e, dict)]
                if cop_values:
                    result[model] = {
                        "cop_mean": float(np.mean(cop_values)),
                        "thermal_eff": 0.8,  # Placeholder
                        "hydraulic_eff": 0.7  # Placeholder
                    }
        
        return result
    
    # =========================================================================
    # Phase 8: Report Generation
    # =========================================================================
    
    def save_results(self) -> None:
        """Save analysis results to disk"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        self.logger.info("Saving results...")
        
        # Create a clean copy of results without circular references
        results_copy = {}
        
        for key, value in self.results.items():
            if key == 'analyzers':
                # Don't save analyzer objects themselves
                results_copy[key] = {
                    model: f"<{type(analyzer).__name__}>" 
                    for model, analyzer in value.items()
                }
            elif key == 'visualizer':
                # Don't save visualizer object
                results_copy[key] = f"<{type(value).__name__}>"
            elif key == 'data':
                # Save data summary instead of full data
                results_copy[key] = {
                    model: {
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                    }
                    for model, df in value.items()
                }
            elif isinstance(value, dict):
                # Handle nested dicts
                try:
                    results_copy[key] = convert_for_json(value)
                except Exception as e:
                    self.logger.warning(f"Could not serialize {key}: {e}")
                    results_copy[key] = f"<serialization failed: {type(value).__name__}>"
            else:
                try:
                    results_copy[key] = convert_for_json(value)
                except Exception as e:
                    self.logger.warning(f"Could not serialize {key}: {e}")
                    results_copy[key] = f"<serialization failed: {type(value).__name__}>"
        
        # Save JSON results
        results_file = self.output_dir / "analysis_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results_copy, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            # Try saving a minimal version
            minimal_results = {
                'timestamp': str(datetime.now()),
                'models': self.models,
                'cdu_ids': self.cdu_ids,
                'output_dir': str(self.output_dir),
                'phases_completed': list(results_copy.keys())
            }
            with open(results_file, 'w') as f:
                json.dump(minimal_results, f, indent=2)
        
        # Save data separately as parquet
        if 'data' in self.results:
            data_dir = self.output_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            for model, df in self.results['data'].items():
                data_file = data_dir / f"{model}_data.parquet"
                df.to_parquet(data_file, index=False, engine='pyarrow')
                self.logger.info(f"Data saved: {data_file}")
        
        # Save static analysis separately
        if 'static_analysis' in self.results:
            static_dir = self.output_dir / "static_analysis"
            static_dir.mkdir(exist_ok=True)
            
            for model, analysis in self.results['static_analysis'].items():
                static_file = static_dir / f"{model}_static.json"
                try:
                    with open(static_file, 'w') as f:
                        json.dump(convert_for_json(analysis), f, indent=2)
                except Exception as e:
                    self.logger.warning(f"Could not save static analysis for {model}: {e}")
        
        # Save dynamic analysis separately
        if 'dynamic_analysis' in self.results:
            dynamic_dir = self.output_dir / "dynamic_analysis"
            dynamic_dir.mkdir(exist_ok=True)
            
            for model, analysis in self.results['dynamic_analysis'].items():
                dynamic_file = dynamic_dir / f"{model}_dynamic.json"
                try:
                    with open(dynamic_file, 'w') as f:
                        json.dump(convert_for_json(analysis), f, indent=2)
                except Exception as e:
                    self.logger.warning(f"Could not save dynamic analysis for {model}: {e}")
        
        self.logger.info("All results saved successfully")
    
    def generate_report(self) -> str:
        """Generate markdown summary report."""
        if not self.config.create_report:
            return ""
        
        logger.info("Generating report")
        
        lines = [
            "# CDU-Level Comparative Analysis Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Models:** {', '.join(self.config.models)}",
            f"\n**CDUs Analyzed:** {len(self.cdu_ids)}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Key findings
        findings = self.results.get("comparison", {}).get("key_findings", [])
        if findings:
            for finding in findings:
                lines.append(finding)
            lines.append("")
        
        # Analysis summary
        lines.extend([
            "## Analysis Phases Completed",
            "",
            "| Phase | Description | Status |",
            "|-------|-------------|--------|"
        ])
        
        phases = [
            ("Static Analysis", "response_statistics" in self.results.get("static", {})),
            ("Dynamic Analysis", "time_constants" in self.results.get("dynamic", {})),
            ("Transfer Functions", "dc_gain_matrices" in self.results.get("transfer_function", {})),
            ("Operating Regime", "thermal_efficiency" in self.results.get("operating_regime", {})),
            ("Physics Validation", "summary_table" in self.results.get("physics_validation", {})),
            ("Cross-Model Comparison", "key_findings" in self.results.get("comparison", {}))
        ]
        
        for phase, completed in phases:
            status = "✅ Complete" if completed else "❌ Not run"
            lines.append(f"| {phase} | - | {status} |")
        
        lines.extend([
            "",
            "## Output Files",
            "",
            f"- Results: `{self.output_dir}/results/`",
            f"- Plots: `{self.output_dir}/plots/`",
            f"- Data: `{self.output_dir}/data/`",
            "",
            "## Visualizations",
            "",
            "See the `plots/` directory for all generated figures."
        ])
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.output_dir / "REPORT.md"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def run(
        self,
        data_paths: Optional[Dict[str, str]] = None,
        simulator_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            data_paths: Optional dict of model->path for existing data
            simulator_func: Optional function to run simulation
            
        Returns:
            All analysis results
        """
        logger.info("=" * 70)
        logger.info("STARTING CDU-LEVEL COMPARATIVE ANALYSIS")
        logger.info("=" * 70)
        
        # Phase 1: Load or generate data
        if data_paths:
            self.load_data(data_paths)
        elif self.config.generate_data:
            self.generate_standardized_data(simulator_func)
        
        # Phase 2: Static analysis
        if self.config.run_static_analysis and self.data:
            self.run_static_analysis()
        
        # Phase 3: Dynamic analysis
        if self.config.run_dynamic_analysis and self.data:
            self.run_dynamic_analysis()
        
        # Phase 4: Transfer function analysis
        if self.config.run_transfer_function and self.data:
            self.run_transfer_function_analysis()
        
        # Phase 5: Operating regime analysis
        if self.config.run_operating_regime and self.data:
            self.run_operating_regime_analysis()
        
        # Phase 5.5: Physics constraint validation
        if self.data:
            self.run_physics_validation()
        
        # Phase 6: Cross-model comparison
        self.compare_across_models()
        
        # Phase 7: Visualization
        self.generate_visualizations()
        
        # Phase 8: Save and report
        self.save_results()
        self.generate_report()
        
        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 70)
        
        return self.results


def run_analysis(
    models: List[str] = None,
    cdu_ids: List[int] = None,
    data_paths: Dict[str, str] = None,
    output_dir: str = "analysis_results/cdu_comparison",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run analysis.
    
    Args:
        models: Models to compare
        cdu_ids: CDU IDs to analyze
        data_paths: Paths to existing data files
        output_dir: Output directory
        **kwargs: Additional config options
        
    Returns:
        Analysis results
    """
    config = AnalysisConfig(
        models=models or ["marconi100", "summit", "lassen"],
        cdu_ids=cdu_ids,
        output_dir=output_dir,
        **kwargs
    )
    
    analysis = CDUComparativeAnalysis(config)
    return analysis.run(data_paths=data_paths)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="CDU-Level Comparative Analysis for Cooling Models"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["marconi100", "summit", "lassen"],
        help="Models to compare"
    )
    parser.add_argument(
        "--cdus", "-c",
        nargs="+",
        type=int,
        default=None,
        help="Specific CDU IDs to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_results/cdu_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help="Data directory containing simulation outputs"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of LHS samples for sensitivity analysis"
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=10,
        help="Grid points per dimension for response surfaces"
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Skip FMU simulation (use existing data or inputs only)"
    )
    parser.add_argument(
        "--stabilization-hours",
        type=int,
        default=2,
        help="Hours for FMU stabilization before simulation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Build data paths if directory provided
    data_paths = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_paths = {}
        for model in args.models:
            for ext in [".parquet", ".csv"]:
                path = data_dir / f"{model}_simulation{ext}"
                if path.exists():
                    data_paths[model] = str(path)
                    break
    
    # Run analysis
    results = run_analysis(
        models=args.models,
        cdu_ids=args.cdus,
        data_paths=data_paths,
        output_dir=args.output,
        n_lhs_samples=args.n_samples,
        n_grid_points=args.n_grid,
        run_fmu_simulation=not args.no_simulation,
        stabilization_hours=args.stabilization_hours
    )
    
    print(f"\nAnalysis complete. Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
