import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import io 
# import streamlit as st

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    
    # System-specific parameters
    n_cdus: int
    system_name: str
    
    # Power parameters
    min_power: float
    max_power: float
    
    # Plot styling
    style: str = 'seaborn-v0_8-whitegrid'
    figsize_small: Tuple[int, int] = (12, 6)
    figsize_medium: Tuple[int, int] = (15, 10)
    figsize_large: Tuple[int, int] = (20, 16)
    dpi: int = 150
    
    # Colors
    color_normal: str = 'green'
    color_edge: str = 'orange'
    color_fault: str = 'red'
    
    # Rolling windows (relative to data length)
    window_power: float = 0.02      # 2% of data
    window_flow: float = 0.0133     # 1.33% of data
    window_temperature: float = 0.025  # 2.5% of data
    window_pressure: float = 0.01   # 1% of data
    window_efficiency: float = 0.033  # 3.33% of data
    
    # Output parameters
    cdu_output_vars: List[str] = None
    dc_output_vars: List[str] = None
    
    def __post_init__(self):
        if self.cdu_output_vars is None:
            self.cdu_output_vars = [
                'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
            ]
        if self.dc_output_vars is None:
            self.dc_output_vars = ['V_flow_prim_GPM', 'pue']


def get_default_config(system_name: str = "marconi100") -> VisualizationConfig:
    """Get default visualization config for a system."""
    from raps.config import ConfigManager
    
    config_manager = ConfigManager(system_name=system_name)
    system_config = config_manager.get_config()
    
    return VisualizationConfig(
        n_cdus=system_config.get('N_CDUS', 49),
        system_name=system_name,
        min_power=system_config.get('MIN_POWER', 5.0),
        max_power=system_config.get('MAX_POWER', 50.0)
    )


def setup_plot_style(style: str = 'seaborn-v0_8-whitegrid'):
    """Setup matplotlib style."""
    plt.style.use(style)
    sns.set_palette("husl")


def apply_rolling_window(series, window: int, center: bool = True, min_periods: int = 1):
    """Apply rolling window average to a series."""
    return series.rolling(window=window, center=center, min_periods=min_periods).mean()


def calculate_adaptive_windows(data_length: int, config: VisualizationConfig) -> Dict[str, int]:
    """Calculate adaptive rolling window sizes based on data length."""
    return {
        'power': max(10, min(300, int(data_length * config.window_power))),
        'flow': max(15, min(300, int(data_length * config.window_flow))),
        'temperature': max(20, min(400, int(data_length * config.window_temperature))),
        'pressure': max(10, min(200, int(data_length * config.window_pressure))),
        'efficiency': max(25, min(500, int(data_length * config.window_efficiency))),
    }


def add_hour_markers(ax, time_indices: List[int], alpha: float = 0.3):
    """Add vertical lines at hour boundaries."""
    if not time_indices:
        return
    
    min_time = min(time_indices)
    max_time = max(time_indices)
    hour_markers = list(range(int(min_time // 3600) * 3600, int(max_time) + 3600, 3600))
    
    for hour in hour_markers:
        ax.axvline(x=hour, color='gray', linestyle='--', alpha=alpha, linewidth=0.8)


def save_figure(fig, output_path: str, dpi: int = 150, bbox_inches: str = 'tight'):
    """Save figure to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to {output_path}")


def get_cdu_column_name(compute_block: int, variable: str, system_name: str = "marconi100") -> str:
    """Get standardized CDU column name for outputs."""
    return f'simulator[1].datacenter[1].computeBlock[{compute_block}].cdu[1].summary.{variable}'


def get_input_column_names(system_name: str = "marconi100", num_cdus: int = 49) -> Dict[str, List[str]]:
    """Get input column names for a system."""
    
    return {
        'Q_flow_total': [
            f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total'
            for i in range(1, num_cdus + 1)
        ],
        'T_Air': [
            f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air'
            for i in range(1, num_cdus + 1)
        ],
        'T_ext': 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
    }


def fig_to_streamlit(fig):
    pass
#     """
#     Convert matplotlib figure to streamlit display
    
#     Parameters:
#     -----------
#     fig : matplotlib.figure.Figure
#         Figure to convert
#     """
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#     buf.seek(0)
#     st.image(buf, use_container_width=True)
#     plt.close(fig)