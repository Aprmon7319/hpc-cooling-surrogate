#!/usr/bin/env python3
"""
CDU-Level Comparative Analysis Runner.

This script orchestrates the complete CDU-level comparative analysis
workflow for understanding how cooling models differ at the CDU level.

Usage:
    python run_cdu_analysis.py --models marconi100 summit lassen
    python run_cdu_analysis.py --models summit --cdus 1 10 50 100
    python run_cdu_analysis.py --config config/cdu_analysis.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fmu2ml.analysis.comparative.analyzers import (
    CDUResponseAnalyzer,
    CDUResponseConfig,
    SensitivityAnalyzer,
    SensitivityConfig,
    DynamicResponseAnalyzer,
    DynamicConfig,
    TransferFunctionAnalyzer,
    TransferFunctionConfig,
)
from fmu2ml.analysis.comparative.visualizers import CDUComparisonVisualizer
from fmu2ml.analysis.comparative.data_generators import (
    StandardizedInputGenerator,
    StepInputGenerator,
    RampInputGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# System configurations
SYSTEM_CONFIGS = {
    "marconi100": {
        "num_cdus": 49,
        "config_path": "config/marconi100/cooling.json",
        "description": "Marconi100 - 49 CDUs"
    },
    "summit": {
        "num_cdus": 257,
        "config_path": "config/summit/cooling.json",
        "description": "Summit - 257 CDUs"
    },
    "lassen": {
        "num_cdus": 25,
        "racks_per_cdu": 3,
        "config_path": "config/lassen/cooling.json",
        "description": "lassen - 25 CDUs (3 racks each)"
    }
}

# CDU I/O column naming conventions
CDU_INPUT_PATTERN = "simulator_1_datacenter_1_computeBlock_{cdu}_cabinet_1_sources_{var}"
CDU_OUTPUT_PATTERN = "simulator[1].datacenter[1].computeBlock[{cdu}].cdu[1].summary.{var}"


class CDUAnalysisRunner:
    """
    Orchestrates CDU-level comparative analysis.
    
    Workflow:
    1. Load/generate standardized test data
    2. Run static analysis (sensitivity, DC gains)
    3. Run dynamic analysis (step response, time constants)
    4. Run transfer function analysis
    5. Generate visualizations
    6. Create comparison report
    """
    
    def __init__(
        self,
        models: List[str],
        output_dir: Path,
        cdu_ids: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analysis runner.
        
        Args:
            models: List of model names to compare
            output_dir: Output directory for results
            cdu_ids: Specific CDU IDs to analyze (None for all)
            config: Optional configuration overrides
        """
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Determine CDU IDs to analyze
        if cdu_ids:
            self.cdu_ids = cdu_ids
        else:
            # Use representative sample from smallest system
            min_cdus = min(SYSTEM_CONFIGS[m]["num_cdus"] for m in models)
            self.cdu_ids = self._select_representative_cdus(min_cdus)
            
        # Initialize analyzers
        self.cdu_response_analyzer = CDUResponseAnalyzer(
            CDUResponseConfig(**self.config.get("cdu_response", {}))
        )
        self.sensitivity_analyzer = SensitivityAnalyzer(
            SensitivityConfig(**self.config.get("sensitivity", {}))
        )
        self.dynamic_analyzer = DynamicResponseAnalyzer(
            DynamicConfig(**self.config.get("dynamic", {}))
        )
        self.tf_analyzer = TransferFunctionAnalyzer(
            TransferFunctionConfig(**self.config.get("transfer_function", {}))
        )
        
        # Initialize visualizer
        self.visualizer = CDUComparisonVisualizer(
            output_dir=self.output_dir / "plots"
        )
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}
        
    def _select_representative_cdus(self, num_cdus: int, n_samples: int = 10) -> List[int]:
        """Select representative CDU IDs for analysis."""
        if num_cdus <= n_samples:
            return list(range(1, num_cdus + 1))
        
        # Select evenly spaced CDUs
        indices = np.linspace(1, num_cdus, n_samples, dtype=int)
        return list(indices)
    
    def load_data(self, data_paths: Dict[str, str]) -> None:
        """
        Load simulation data for each model.
        
        Args:
            data_paths: Dict mapping model name to data file path
        """
        for model, path in data_paths.items():
            if model not in self.models:
                continue
                
            logger.info(f"Loading data for {model} from {path}")
            
            try:
                if path.endswith(".parquet"):
                    self.data[model] = pd.read_parquet(path)
                elif path.endswith(".csv"):
                    self.data[model] = pd.read_csv(path)
                elif path.endswith(".h5") or path.endswith(".hdf5"):
                    self.data[model] = pd.read_hdf(path)
                else:
                    logger.warning(f"Unknown format for {path}")
                    continue
                    
                logger.info(f"Loaded {len(self.data[model])} rows for {model}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {model}: {e}")
                
    def generate_test_data(
        self,
        simulator_func: Optional[callable] = None,
        n_samples: int = 1000
    ) -> None:
        """
        Generate standardized test data for all models.
        
        Args:
            simulator_func: Optional function to run simulation
            n_samples: Number of samples to generate
        """
        logger.info("Generating standardized test data")
        
        # Create input generator
        input_gen = StandardizedInputGenerator()
        
        for model in self.models:
            if model in self.data:
                continue  # Already have data
                
            config = SYSTEM_CONFIGS.get(model, {})
            num_cdus = config.get("num_cdus", 1)
            
            # Generate realistic operating conditions
            inputs = input_gen.generate_realistic_operating_conditions(
                num_cdus=num_cdus,
                n_samples=n_samples
            )
            
            logger.info(f"Generated {len(inputs)} test inputs for {model}")
            
            # If simulator provided, run simulation
            if simulator_func:
                try:
                    outputs = simulator_func(model, inputs)
                    self.data[model] = pd.concat([inputs, outputs], axis=1)
                except Exception as e:
                    logger.error(f"Simulation failed for {model}: {e}")
                    self.data[model] = inputs
            else:
                self.data[model] = inputs
                
    def run_static_analysis(self) -> Dict[str, Any]:
        """
        Run static analysis: sensitivity, DC gains, operating ranges.
        
        Returns:
            Static analysis results
        """
        logger.info("="*50)
        logger.info("Running Static Analysis")
        logger.info("="*50)
        
        results = {
            "sensitivity": {},
            "dc_gains": {},
            "operating_ranges": {}
        }
        
        for model in self.models:
            if model not in self.data:
                logger.warning(f"No data for {model}, skipping")
                continue
                
            data = self.data[model]
            results["sensitivity"][model] = {}
            results["dc_gains"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Analyzing {model} CDU {cdu_id}")
                
                try:
                    # Sensitivity analysis
                    sens_result = self.sensitivity_analyzer.analyze_cdu(
                        data, cdu_id, model
                    )
                    results["sensitivity"][model][cdu_id] = {
                        "matrix": sens_result.mean_sensitivity_matrix,
                        "rankings": sens_result.sensitivity_rankings,
                        "sobol_first": sens_result.first_order_sobol,
                        "sobol_total": sens_result.total_order_sobol,
                        "high_sensitivity_regions": sens_result.high_sensitivity_regions
                    }
                    
                    # Transfer function (DC gains)
                    tf_result = self.tf_analyzer.analyze_cdu(data, cdu_id, model)
                    results["dc_gains"][model][cdu_id] = {
                        "matrix": tf_result.dc_gain_matrix,
                        "coupling": tf_result.coupling_strength,
                        "rga": tf_result.rga
                    }
                    
                except Exception as e:
                    logger.error(f"Static analysis failed for {model} CDU {cdu_id}: {e}")
                    
        self.results["static"] = results
        return results
    
    def run_dynamic_analysis(self, step_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Run dynamic analysis: step response, time constants.
        
        Args:
            step_data: Optional pre-generated step response data
            
        Returns:
            Dynamic analysis results
        """
        logger.info("="*50)
        logger.info("Running Dynamic Analysis")
        logger.info("="*50)
        
        results = {
            "step_response": {},
            "time_constants": {},
            "delays": {}
        }
        
        # Generate step test data if not provided
        if step_data is None:
            logger.info("Generating step test data")
            step_gen = StepInputGenerator()
            step_data = {}
            
            for model in self.models:
                if model not in self.data:
                    continue
                config = SYSTEM_CONFIGS.get(model, {})
                num_cdus = config.get("num_cdus", 1)
                
                # Generate step tests for each input
                steps = step_gen.generate_multi_step_sequence(
                    cdu_ids=self.cdu_ids,
                    input_vars=["Q_flow", "T_Air"],
                    step_duration=100,
                    step_magnitude=0.2
                )
                step_data[model] = steps
                
        for model in self.models:
            if model not in step_data:
                continue
                
            results["step_response"][model] = {}
            results["time_constants"][model] = {}
            results["delays"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Dynamic analysis for {model} CDU {cdu_id}")
                
                try:
                    # Analyze step response
                    dyn_results = self.dynamic_analyzer.analyze_from_data(
                        step_data[model], cdu_id, model
                    )
                    
                    results["step_response"][model][cdu_id] = {}
                    results["time_constants"][model][cdu_id] = {}
                    results["delays"][model][cdu_id] = {}
                    
                    for input_var, outputs in dyn_results.items():
                        for output_var, result in outputs.items():
                            key = f"{input_var}_{output_var}"
                            results["step_response"][model][cdu_id][key] = {
                                "rise_time": result.rise_time,
                                "settling_time": result.settling_time,
                                "overshoot": result.overshoot,
                                "gain": result.steady_state_gain
                            }
                            results["time_constants"][model][cdu_id][output_var] = result.dominant_time_constant
                            results["delays"][model][cdu_id][key] = result.delay
                            
                except Exception as e:
                    logger.error(f"Dynamic analysis failed for {model} CDU {cdu_id}: {e}")
                    
        self.results["dynamic"] = results
        return results
    
    def run_transfer_function_analysis(self) -> Dict[str, Any]:
        """
        Run transfer function analysis: system identification, poles/zeros.
        
        Returns:
            Transfer function analysis results
        """
        logger.info("="*50)
        logger.info("Running Transfer Function Analysis")
        logger.info("="*50)
        
        results = {
            "transfer_functions": {},
            "poles": {},
            "stability": {}
        }
        
        for model in self.models:
            if model not in self.data:
                continue
                
            results["transfer_functions"][model] = {}
            results["poles"][model] = {}
            results["stability"][model] = {}
            
            for cdu_id in self.cdu_ids:
                logger.info(f"Transfer function analysis for {model} CDU {cdu_id}")
                
                try:
                    tf_result = self.tf_analyzer.analyze_cdu(
                        self.data[model], cdu_id, model
                    )
                    
                    results["transfer_functions"][model][cdu_id] = tf_result.transfer_functions
                    results["poles"][model][cdu_id] = {
                        "dominant": [str(p) for p in tf_result.dominant_poles],
                        "all": {k: [str(p) for p in v] for k, v in tf_result.poles.items()}
                    }
                    results["stability"][model][cdu_id] = tf_result.stability_analysis
                    
                except Exception as e:
                    logger.error(f"TF analysis failed for {model} CDU {cdu_id}: {e}")
                    
        self.results["transfer_function"] = results
        return results
    
    def compare_across_models(self) -> Dict[str, Any]:
        """
        Compare results across all models for the same CDU.
        
        Returns:
            Cross-model comparison results
        """
        logger.info("="*50)
        logger.info("Comparing Results Across Models")
        logger.info("="*50)
        
        comparison = {
            "sensitivity_comparison": {},
            "gain_comparison": {},
            "dynamics_comparison": {},
            "stability_comparison": {}
        }
        
        for cdu_id in self.cdu_ids:
            logger.info(f"Cross-model comparison for CDU {cdu_id}")
            
            # Collect sensitivity matrices
            sens_matrices = {}
            for model in self.models:
                if model in self.results.get("static", {}).get("sensitivity", {}):
                    cdu_data = self.results["static"]["sensitivity"][model].get(cdu_id, {})
                    if "matrix" in cdu_data and cdu_data["matrix"] is not None:
                        sens_matrices[model] = cdu_data["matrix"]
                        
            if len(sens_matrices) >= 2:
                comparison["sensitivity_comparison"][cdu_id] = self.sensitivity_analyzer.compare_sensitivities_across_models(
                    {m: self.sensitivity_analyzer.results[m][cdu_id] for m in sens_matrices if m in self.sensitivity_analyzer.results}
                )
                
            # Collect DC gains
            gain_matrices = {}
            for model in self.models:
                if model in self.results.get("static", {}).get("dc_gains", {}):
                    cdu_data = self.results["static"]["dc_gains"][model].get(cdu_id, {})
                    if "matrix" in cdu_data and cdu_data["matrix"] is not None:
                        gain_matrices[model] = cdu_data["matrix"]
                        
            if len(gain_matrices) >= 2:
                comparison["gain_comparison"][cdu_id] = self._compare_gain_matrices(gain_matrices)
                
            # Collect dynamics
            dynamics = {}
            for model in self.models:
                if model in self.results.get("dynamic", {}).get("time_constants", {}):
                    dynamics[model] = self.results["dynamic"]["time_constants"][model].get(cdu_id, {})
                    
            if len(dynamics) >= 2:
                comparison["dynamics_comparison"][cdu_id] = self._compare_dynamics(dynamics)
                
        self.results["comparison"] = comparison
        return comparison
    
    def _compare_gain_matrices(self, matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare DC gain matrices."""
        models = list(matrices.keys())
        result = {"pairwise_diff": {}}
        
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                diff = np.abs(matrices[m1] - matrices[m2])
                result["pairwise_diff"][f"{m1}_vs_{m2}"] = {
                    "max_diff": float(np.max(diff)),
                    "mean_diff": float(np.mean(diff)),
                    "relative_diff": float(np.mean(diff) / (np.mean(np.abs(matrices[m1]) + np.abs(matrices[m2])) / 2 + 1e-10))
                }
                
        return result
    
    def _compare_dynamics(self, dynamics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare dynamic characteristics."""
        models = list(dynamics.keys())
        outputs = set()
        for d in dynamics.values():
            outputs.update(d.keys())
            
        result = {"time_constant_diff": {}}
        
        for output in outputs:
            values = {m: dynamics[m].get(output, 0) for m in models}
            vals = list(values.values())
            if len(vals) >= 2:
                result["time_constant_diff"][output] = {
                    "values": values,
                    "range": max(vals) - min(vals),
                    "cv": np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                }
                
        return result
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        logger.info("="*50)
        logger.info("Generating Visualizations")
        logger.info("="*50)
        
        for cdu_id in self.cdu_ids:
            logger.info(f"Generating plots for CDU {cdu_id}")
            
            # Sensitivity heatmaps
            sens_matrices = {}
            for model in self.models:
                static = self.results.get("static", {})
                if model in static.get("sensitivity", {}):
                    matrix = static["sensitivity"][model].get(cdu_id, {}).get("matrix")
                    if matrix is not None:
                        sens_matrices[model] = matrix
                        
            if sens_matrices:
                self.visualizer.plot_sensitivity_comparison(sens_matrices, cdu_id)
                
            # DC gain comparison
            gain_matrices = {}
            for model in self.models:
                static = self.results.get("static", {})
                if model in static.get("dc_gains", {}):
                    matrix = static["dc_gains"][model].get(cdu_id, {}).get("matrix")
                    if matrix is not None:
                        gain_matrices[model] = matrix
                        
            if gain_matrices:
                self.visualizer.plot_dc_gain_comparison(gain_matrices, cdu_id)
                
            # Time constant comparison
            tc_data = {}
            for model in self.models:
                dynamic = self.results.get("dynamic", {})
                if model in dynamic.get("time_constants", {}):
                    tc_data[model] = dynamic["time_constants"][model].get(cdu_id, {})
                    
            if tc_data:
                self.visualizer.plot_time_constant_comparison(tc_data, cdu_id)
                
            # RGA comparison
            rga_matrices = {}
            for model in self.models:
                static = self.results.get("static", {})
                if model in static.get("dc_gains", {}):
                    rga = static["dc_gains"][model].get(cdu_id, {}).get("rga")
                    if rga is not None:
                        rga_matrices[model] = rga
                        
            if rga_matrices:
                self.visualizer.plot_rga_comparison(rga_matrices, cdu_id)
                
            # Sobol indices
            sobol_data = {}
            for model in self.models:
                static = self.results.get("static", {})
                if model in static.get("sensitivity", {}):
                    sobol = static["sensitivity"][model].get(cdu_id, {}).get("sobol_total")
                    if sobol:
                        sobol_data[model] = sobol
                        
            if sobol_data:
                self.visualizer.plot_sobol_indices(sobol_data, cdu_id, order="total")
                
        self.visualizer.close_all()
        
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results")
        
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save main results as JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, complex):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj
            
        with open(results_dir / "analysis_results.json", "w") as f:
            json.dump(convert_for_json(self.results), f, indent=2)
            
        # Save analyzer-specific results
        self.sensitivity_analyzer.save_results(results_dir / "sensitivity")
        self.tf_analyzer.save_results(results_dir / "transfer_functions")
        self.dynamic_analyzer.save_results(results_dir / "dynamics")
        
        logger.info(f"Results saved to {results_dir}")
        
    def generate_report(self) -> str:
        """
        Generate summary report.
        
        Returns:
            Report as markdown string
        """
        report = []
        report.append("# CDU-Level Comparative Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Models:** {', '.join(self.models)}")
        report.append(f"\n**CDUs Analyzed:** {len(self.cdu_ids)}")
        
        report.append("\n## Executive Summary\n")
        
        # Key findings
        report.append("### Key Findings\n")
        
        comparison = self.results.get("comparison", {})
        
        # Sensitivity differences
        if "sensitivity_comparison" in comparison:
            report.append("#### Sensitivity Differences\n")
            for cdu_id, comp in comparison["sensitivity_comparison"].items():
                if "jacobian_comparison" in comp:
                    for pair, metrics in comp["jacobian_comparison"].items():
                        report.append(f"- **CDU {cdu_id}** ({pair}): {metrics.get('relative_diff', 0):.2%} relative difference")
                        
        # Gain differences
        if "gain_comparison" in comparison:
            report.append("\n#### DC Gain Differences\n")
            for cdu_id, comp in comparison["gain_comparison"].items():
                for pair, metrics in comp.get("pairwise_diff", {}).items():
                    report.append(f"- **CDU {cdu_id}** ({pair}): {metrics.get('relative_diff', 0):.2%} relative difference")
                    
        # Dynamics differences
        if "dynamics_comparison" in comparison:
            report.append("\n#### Dynamic Response Differences\n")
            for cdu_id, comp in comparison["dynamics_comparison"].items():
                max_cv = 0
                max_output = ""
                for output, metrics in comp.get("time_constant_diff", {}).items():
                    if metrics.get("cv", 0) > max_cv:
                        max_cv = metrics["cv"]
                        max_output = output
                if max_output:
                    report.append(f"- **CDU {cdu_id}**: Largest variation in {max_output} (CV={max_cv:.2f})")
                    
        report.append("\n## Detailed Results\n")
        report.append("See the `results/` directory for detailed JSON outputs and CSV summaries.")
        report.append("\n## Visualizations\n")
        report.append("See the `plots/` directory for all generated figures.")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "REPORT.md"
        with open(report_path, "w") as f:
            f.write(report_text)
            
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Returns:
            All analysis results
        """
        logger.info("="*60)
        logger.info("Starting CDU-Level Comparative Analysis")
        logger.info("="*60)
        
        # Run all analysis stages
        self.run_static_analysis()
        self.run_dynamic_analysis()
        self.run_transfer_function_analysis()
        
        # Compare across models
        self.compare_across_models()
        
        # Generate outputs
        self.generate_visualizations()
        self.save_results()
        self.generate_report()
        
        logger.info("="*60)
        logger.info("Analysis Complete")
        logger.info("="*60)
        
        return self.results


def main():
    """Main entry point."""
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
        default="data",
        help="Data directory containing simulation outputs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load configuration if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    
    # Initialize runner
    runner = CDUAnalysisRunner(
        models=args.models,
        output_dir=output_dir,
        cdu_ids=args.cdus,
        config=config
    )
    
    # Look for data files
    data_dir = Path(args.data_dir)
    data_paths = {}
    for model in args.models:
        for ext in [".parquet", ".csv", ".h5"]:
            path = data_dir / f"{model}_simulation{ext}"
            if path.exists():
                data_paths[model] = str(path)
                break
                
    if data_paths:
        runner.load_data(data_paths)
    else:
        logger.warning("No data files found, generating test data")
        runner.generate_test_data(n_samples=500)
        
    # Run analysis
    runner.run_full_analysis()
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
