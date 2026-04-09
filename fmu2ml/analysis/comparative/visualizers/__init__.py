"""
Comparative Analysis Visualizers Module.

Provides visualization tools for:
- System-level comparisons
- CDU-level analysis
- Sensitivity and dynamic response plots
- Response surfaces
- Model comparison dashboards
- Physics validation
"""

from .comparison_charts import (
    create_system_comparison_chart,
    create_efficiency_comparison_plot,
    create_radar_comparison_chart
)
from .thermal_visualizer import create_thermal_response_comparison
from .scaling_visualizer import create_scaling_analysis_plot
from .power_visualizer import create_power_profile_comparison
from .report_generator import create_comprehensive_report
from .cdu_comparison_visualizer import CDUComparisonVisualizer
from .response_surface_visualizer import ResponseSurfaceVisualizer
from .model_comparison_dashboard import ModelComparisonDashboard
from .simple_comparison_visualizer import SimpleComparisonVisualizer

__all__ = [
    # System-level visualization
    "create_system_comparison_chart",
    "create_efficiency_comparison_plot",
    "create_thermal_response_comparison",
    "create_scaling_analysis_plot",
    "create_comprehensive_report",
    "create_radar_comparison_chart",
    "create_power_profile_comparison",
    # CDU-level visualization
    "CDUComparisonVisualizer",
    # Response surface visualization
    "ResponseSurfaceVisualizer",
    # Dashboard
    "ModelComparisonDashboard",
    # Simple comparison
    "SimpleComparisonVisualizer",
]
