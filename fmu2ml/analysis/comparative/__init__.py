"""
Comparative Analysis Module for Data Center Cooling Models.

This module provides tools for comparing cooling system behaviors across
different HPC data center configurations (e.g., Summit, Marconi100, Frontier).

Main Components:
================

Analyzers (fmu2ml.analysis.comparative.analyzers):
-------------------------------------------------
- CDUResponseAnalyzer: Per-CDU I/O response characterization
- SensitivityAnalyzer: Jacobian and Sobol sensitivity analysis
- DynamicResponseAnalyzer: Step response and time constants
- TransferFunctionAnalyzer: System identification and coupling
- OperatingRegimeAnalyzer: Efficiency, envelope, and constraints

Data Generators (fmu2ml.analysis.comparative.data_generators):
--------------------------------------------------------------
- StandardizedInputGenerator: Identical inputs for all models
- StepInputGenerator: Step test sequences
- RampInputGenerator: Ramp test sequences
- FrequencySweepGenerator: Frequency response testing
- GridInputGenerator: Response surface mapping

Visualizers (fmu2ml.analysis.comparative.visualizers):
------------------------------------------------------
- CDUComparisonVisualizer: Per-CDU comparison plots
- ResponseSurfaceVisualizer: 2D/3D response surfaces
- ModelComparisonDashboard: Executive summary dashboards

Runners (fmu2ml.analysis.comparative.runners):
----------------------------------------------
- CDUComparativeAnalysis: Main orchestrator for complete analysis
- run_analysis: Convenience function for quick analysis

Analysis Workflow:
==================
Phase 1: Data Generation - Standardized inputs for all models
Phase 2: Static Analysis - Response surfaces, sensitivity, correlations
Phase 3: Dynamic Analysis - Step response, time constants, delays
Phase 4: Transfer Function Analysis - Gains, coupling, stability
Phase 5: Operating Regime Analysis - Efficiency, constraints
Phase 6: Cross-Model Comparison
Phase 7: Visualization
Phase 8: Report Generation
"""

from .analyzers import (
    # System-level
    CoolingModelComparator,
    SystemProfiler,
    MetricsCalculator,
    # CDU-level
    CDUResponseAnalyzer,
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityResult,
    DynamicResponseAnalyzer,
    DynamicConfig,
    DynamicResult,
    TransferFunctionAnalyzer,
    TransferFunctionConfig,
    TransferFunctionResult,
    OperatingRegimeAnalyzer,
    OperatingRegimeConfig,
    OperatingRegimeResult,

    # Physics validation
    PhysicsConstraintValidator,
    PhysicsValidatorConfig,
)
from .visualizers import (
    # System-level
    create_system_comparison_chart,
    create_efficiency_comparison_plot,
    create_thermal_response_comparison,
    create_scaling_analysis_plot,
    create_comprehensive_report,
    create_radar_comparison_chart,
    create_power_profile_comparison,
    # CDU-level
    CDUComparisonVisualizer,
    ResponseSurfaceVisualizer,
    ModelComparisonDashboard,
)
from .data_generators import (
    StandardizedInputGenerator,
    StepInputGenerator,
    RampInputGenerator,
    FrequencySweepGenerator,
    GridInputGenerator,
)
from .runners import (
    CDUComparativeAnalysis,
    AnalysisConfig,
    run_analysis,
)

__all__ = [
    # System-Level Analyzers
    "CoolingModelComparator",
    "SystemProfiler",
    "MetricsCalculator",
    # CDU-Level Analyzers
    "CDUResponseAnalyzer",
    "SensitivityAnalyzer",
    "SensitivityConfig",
    "SensitivityResult",
    "DynamicResponseAnalyzer",
    "DynamicConfig",
    "DynamicResult",
    "TransferFunctionAnalyzer",
    "TransferFunctionConfig",
    "TransferFunctionResult",
    "OperatingRegimeAnalyzer",
    "OperatingRegimeConfig",
    "OperatingRegimeResult",
    # Physics validation
    "PhysicsConstraintValidator",
    "PhysicsValidatorConfig",
    # System-Level Visualizers
    "create_system_comparison_chart",
    "create_efficiency_comparison_plot",
    "create_thermal_response_comparison",
    "create_scaling_analysis_plot",
    "create_comprehensive_report",
    "create_radar_comparison_chart",
    "create_power_profile_comparison",
    # CDU-Level Visualizers
    "CDUComparisonVisualizer",
    "ResponseSurfaceVisualizer",
    "ModelComparisonDashboard",
    # Data Generators
    "StandardizedInputGenerator",
    "StepInputGenerator",
    "RampInputGenerator",
    "FrequencySweepGenerator",
    "GridInputGenerator",
    # Runners
    "CDUComparativeAnalysis",
    "AnalysisConfig",
    "run_analysis",
]

