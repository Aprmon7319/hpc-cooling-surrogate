"""
Non-linear relationship characterization module for FMU input-output analysis.

This module provides tools for:
- Polynomial/spline fitting to characterize non-linear relationships
- Threshold/saturation detection using segmented regression
- Model comparison (linear vs. non-linear)
- Operating regime classification
"""

from .analyzers import NonlinearAnalyzer, ThresholdDetector
from .visualizers import (
    create_nonlinearity_plots,
    create_model_comparison_plots,
    create_residual_plots,
    create_threshold_plots,
    create_regime_scatter_plots,
    create_regime_summary_plot,
    create_nonlinearity_strength_chart
)

__all__ = [
    'NonlinearAnalyzer',
    'ThresholdDetector',
    'create_nonlinearity_plots',
    'create_model_comparison_plots',
    'create_residual_plots',
    'create_threshold_plots',
    'create_regime_scatter_plots',
    'create_regime_summary_plot',
    'create_nonlinearity_strength_chart'
]
