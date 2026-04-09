"""
Multivariate Effect Analysis module for FMU input-output relationships.

This module provides tools for:
- Partial correlation analysis to identify direct effects
- Multiple regression analysis with beta coefficients, significance, and VIF
- Lagged effect analysis using distributed lag models
- Autocorrelation and cross-correlation analysis for temporal dependencies

All analyses are parallelized using Dask and multiprocessing for efficiency.
"""

from .analyzers import (
    PartialCorrelationAnalyzer,
    MultipleRegressionAnalyzer,
    LaggedEffectAnalyzer,
    AutocorrelationAnalyzer
)
from .visualizers import (
    create_correlation_comparison_chart,
    create_correlation_heatmaps,
    create_network_diagram,
    create_confounding_analysis_plot,
    create_beta_coefficient_plot,
    create_significance_heatmap,
    create_variance_explained_chart,
    create_vif_plot,
    create_regression_summary_plot,
    create_lag_coefficient_plots,
    create_lag_heatmap,
    create_memory_length_plot,
    create_cumulative_effect_summary,
    create_lag_analysis_summary,
    create_acf_plots,
    create_ccf_plots,
    create_persistence_summary,
    create_ccf_heatmap,
    create_temporal_recommendations_plot
)

__all__ = [
    # Analyzers
    'PartialCorrelationAnalyzer',
    'MultipleRegressionAnalyzer',
    'LaggedEffectAnalyzer',
    'AutocorrelationAnalyzer',

    # Visualizers - Partial Correlation
    'create_correlation_comparison_chart',
    'create_correlation_heatmaps',
    'create_network_diagram',
    'create_confounding_analysis_plot',
    
    # Visualizers - Regression
    'create_beta_coefficient_plot',
    'create_significance_heatmap',
    'create_variance_explained_chart',
    'create_vif_plot',
    'create_regression_summary_plot',
    
    # Visualizers - Lagged Effects
    'create_lag_coefficient_plots',
    'create_lag_heatmap',
    'create_memory_length_plot',
    'create_cumulative_effect_summary',
    'create_lag_analysis_summary',
    
    # Visualizers - Autocorrelation
    'create_acf_plots',
    'create_ccf_plots',
    'create_persistence_summary',
    'create_ccf_heatmap',
    'create_temporal_recommendations_plot'
]
