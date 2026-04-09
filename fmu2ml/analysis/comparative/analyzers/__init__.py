"""
Comparative Analysis Analyzers Module.

This module provides analyzers for CDU-level comparative analysis
of cooling models across different HPC systems.

Analyzers:
- CDUResponseAnalyzer: Per-CDU I/O response characterization
- SensitivityAnalyzer: Jacobian and Sobol sensitivity analysis
- DynamicResponseAnalyzer: Step response and time constants
- TransferFunctionAnalyzer: System identification and coupling
- OperatingRegimeAnalyzer: Efficiency, envelope, and constraints
- PhysicsConstraintValidator: Physics constraint validation
"""

from .cooling_model_comparator import CoolingModelComparator
from .system_profiler import SystemProfiler
from .metrics_calculator import MetricsCalculator
from .cdu_response_analyzer import CDUResponseAnalyzer
from .sensitivity_analyzer import SensitivityAnalyzer, SensitivityConfig, SensitivityResult
from .dynamic_response_analyzer import DynamicResponseAnalyzer, DynamicConfig, DynamicResult
from .transfer_function_analyzer import TransferFunctionAnalyzer, TransferFunctionConfig, TransferFunctionResult
from .operating_regime_analyzer import OperatingRegimeAnalyzer, OperatingRegimeConfig, OperatingRegimeResult
from .physics_validator import PhysicsConstraintValidator, PhysicsValidatorConfig

__all__ = [
    # System-level analyzers
    "CoolingModelComparator",
    "SystemProfiler",
    "MetricsCalculator",
    # CDU-level analyzers
    "CDUResponseAnalyzer",
    # Sensitivity analysis
    "SensitivityAnalyzer",
    "SensitivityConfig",
    "SensitivityResult",
    # Dynamic response analysis
    "DynamicResponseAnalyzer",
    "DynamicConfig",
    "DynamicResult",
    # Transfer function analysis
    "TransferFunctionAnalyzer",
    "TransferFunctionConfig",
    "TransferFunctionResult",
    # Operating regime analysis
    "OperatingRegimeAnalyzer",
    "OperatingRegimeConfig",
    "OperatingRegimeResult",
    # Physics validation
    "PhysicsConstraintValidator",
    "PhysicsValidatorConfig",
]
