"""
Rate of Change Analysis Module

Analyzes dynamic effects of inputs on outputs including:
- Temporal derivatives analysis
- Impulse response characterization
- Level vs rate effect comparison
"""

from .analyzers import (
    RateOfChangeAnalyzer,
    ImpulseResponseAnalyzer,
    DynamicDataGenerator
)
from .visualizers import (
    create_rate_comparison_plots,
    create_impulse_response_plots,
    create_time_series_overlay_plots,
    create_response_characteristics_heatmap,
    create_level_rate_correlation_bars
)

__all__ = [
    'RateOfChangeAnalyzer',
    'ImpulseResponseAnalyzer',
    'DynamicDataGenerator',
    'create_rate_comparison_plots',
    'create_impulse_response_plots',
    'create_time_series_overlay_plots',
    'create_response_characteristics_heatmap',
    'create_level_rate_correlation_bars'
]
