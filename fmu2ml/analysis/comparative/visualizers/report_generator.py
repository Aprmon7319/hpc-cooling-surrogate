"""
Comprehensive Report Generator.

Creates a full comparison report with all visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from .comparison_charts import (
    create_system_comparison_chart,
    create_efficiency_comparison_plot,
    create_radar_comparison_chart
)
from .thermal_visualizer import create_thermal_response_comparison
from .scaling_visualizer import create_scaling_analysis_plot
from .power_visualizer import create_power_profile_comparison

logger = logging.getLogger(__name__)


def create_comprehensive_report(
    comparison_results: Dict[str, pd.DataFrame],
    system_metrics: Dict[str, Dict],
    output_dir: Union[str, Path],
    system_data: Optional[Dict[str, pd.DataFrame]] = None,
    create_pdf: bool = True,
    figsize: tuple = (14, 10)
) -> Dict[str, Path]:
    """
    Create a comprehensive comparison report with all visualizations.
    
    Args:
        comparison_results: Dictionary of comparison DataFrames
        system_metrics: Dictionary of metrics per system
        output_dir: Directory to save the report
        system_data: Optional raw simulation data per system
        create_pdf: Whether to create a combined PDF report
        figsize: Default figure size
        
    Returns:
        Dictionary mapping report names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths = {}
    figures = []
    
    logger.info(f"Generating comprehensive report in {output_dir}")
    
    # 1. Efficiency Comparison
    if 'efficiency' in comparison_results:
        fig = create_efficiency_comparison_plot(
            comparison_results['efficiency'],
            output_path=output_dir / f"01_efficiency_comparison_{timestamp}.png",
            figsize=figsize
        )
        if fig:
            figures.append(fig)
            report_paths['efficiency'] = output_dir / f"01_efficiency_comparison_{timestamp}.png"
    
    # 2. Thermal Comparison
    if 'thermal' in comparison_results:
        fig = create_thermal_response_comparison(
            comparison_results['thermal'],
            output_path=output_dir / f"02_thermal_comparison_{timestamp}.png",
            figsize=figsize
        )
        if fig:
            figures.append(fig)
            report_paths['thermal'] = output_dir / f"02_thermal_comparison_{timestamp}.png"
    
    # 3. Power Profile Comparison
    if 'efficiency' in comparison_results and 'flow' in comparison_results:
        fig = create_power_profile_comparison(
            comparison_results['efficiency'],
            comparison_results['flow'],
            output_path=output_dir / f"03_power_profile_{timestamp}.png",
            figsize=figsize
        )
        if fig:
            figures.append(fig)
            report_paths['power_profile'] = output_dir / f"03_power_profile_{timestamp}.png"
    
    # 4. Scaling Analysis
    if 'normalized' in comparison_results:
        fig = create_scaling_analysis_plot(
            comparison_results['normalized'],
            output_path=output_dir / f"04_scaling_analysis_{timestamp}.png",
            figsize=figsize
        )
        if fig:
            figures.append(fig)
            report_paths['scaling'] = output_dir / f"04_scaling_analysis_{timestamp}.png"
    
    # 5. Radar Chart Comparison
    if 'normalized' in comparison_results:
        radar_metrics = [
            'cdup_power_per_cdu_kw',
            'sec_flow_per_cdu_gpm',
            'mean_delta_t_c',
            'mean_rack_return_temp_c'
        ]
        available_metrics = [m for m in radar_metrics 
                           if m in comparison_results['normalized'].columns]
        
        if len(available_metrics) >= 3:
            fig = create_radar_comparison_chart(
                comparison_results['normalized'],
                metrics=available_metrics,
                output_path=output_dir / f"05_radar_comparison_{timestamp}.png",
                figsize=(10, 10)
            )
            if fig:
                figures.append(fig)
                report_paths['radar'] = output_dir / f"05_radar_comparison_{timestamp}.png"
    
    # 6. Flow Comparison
    if 'flow' in comparison_results:
        flow_metrics = ['total_sec_flow_gpm', 'sec_flow_per_cdu_gpm', 
                       'total_prim_flow_gpm', 'prim_flow_per_cdu_gpm']
        available_flow = [m for m in flow_metrics 
                         if m in comparison_results['flow'].columns]
        
        if available_flow:
            fig = create_system_comparison_chart(
                comparison_results['flow'],
                metric_columns=available_flow,
                title="Flow Rate Comparison",
                output_path=output_dir / f"06_flow_comparison_{timestamp}.png",
                figsize=(12, 6)
            )
            if fig:
                figures.append(fig)
                report_paths['flow'] = output_dir / f"06_flow_comparison_{timestamp}.png"
    
    # 7. Dynamic Response Comparison
    if 'dynamic' in comparison_results:
        dynamic_metrics = ['mean_temp_rate_c_per_s', 'max_temp_rate_c_per_s',
                          'thermal_time_constant_approx_s']
        available_dynamic = [m for m in dynamic_metrics 
                            if m in comparison_results['dynamic'].columns]
        
        if available_dynamic:
            fig = create_system_comparison_chart(
                comparison_results['dynamic'],
                metric_columns=available_dynamic,
                title="Dynamic Response Comparison",
                output_path=output_dir / f"07_dynamic_comparison_{timestamp}.png",
                figsize=(12, 6)
            )
            if fig:
                figures.append(fig)
                report_paths['dynamic'] = output_dir / f"07_dynamic_comparison_{timestamp}.png"
    
    # Create combined PDF report
    if create_pdf and figures:
        pdf_path = output_dir / f"cooling_model_comparison_report_{timestamp}.pdf"
        with PdfPages(pdf_path) as pdf:
            # Add title page
            title_fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.text(0.5, 0.7, 'Cooling Model Comparison Report', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='center', va='center', fontsize=14)
            
            systems = list(system_metrics.keys())
            ax.text(0.5, 0.35, f'Systems Compared: {", ".join(s.upper() for s in systems)}',
                   ha='center', va='center', fontsize=12)
            
            pdf.savefig(title_fig, bbox_inches='tight')
            plt.close(title_fig)
            
            # Add all figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        report_paths['pdf_report'] = pdf_path
        logger.info(f"Created PDF report: {pdf_path}")
    else:
        # Close figures if not creating PDF
        for fig in figures:
            plt.close(fig)
    
    # Save summary text report
    summary_path = output_dir / f"comparison_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(generate_text_summary(comparison_results, system_metrics))
    report_paths['summary'] = summary_path
    
    logger.info(f"Report generation complete. {len(report_paths)} files created.")
    
    return report_paths


def generate_text_summary(
    comparison_results: Dict[str, pd.DataFrame],
    system_metrics: Dict[str, Dict]
) -> str:
    """Generate a text summary of the comparison."""
    lines = [
        "=" * 80,
        "COOLING MODEL COMPARISON SUMMARY REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Systems Compared: {', '.join(system_metrics.keys())}",
        "",
        "=" * 80,
        "SYSTEM OVERVIEW",
        "=" * 80,
    ]
    
    for system_name, metrics in system_metrics.items():
        info = metrics.get('system_info', {})
        lines.append(f"\n{system_name.upper()}:")
        lines.append(f"  - CDUs: {info.get('num_cdus', 'N/A')}")
        lines.append(f"  - Nodes per Rack: {info.get('nodes_per_rack', 'N/A')}")
        lines.append(f"  - GPUs per Node: {info.get('gpus_per_node', 'N/A')}")
        lines.append(f"  - Cooling Efficiency: {info.get('cooling_efficiency', 'N/A')}")
    
    if 'efficiency' in comparison_results:
        lines.extend([
            "",
            "=" * 80,
            "EFFICIENCY METRICS",
            "=" * 80,
        ])
        df = comparison_results['efficiency']
        for _, row in df.iterrows():
            lines.append(f"\n{row['system'].upper()}:")
            for col in df.columns:
                if col != 'system' and not pd.isna(row[col]):
                    lines.append(f"  - {col}: {row[col]:.4f}")
    
    if 'thermal' in comparison_results:
        lines.extend([
            "",
            "=" * 80,
            "THERMAL METRICS",
            "=" * 80,
        ])
        df = comparison_results['thermal']
        for _, row in df.iterrows():
            lines.append(f"\n{row['system'].upper()}:")
            for col in df.columns:
                if col not in ['system', 'num_cdus'] and not pd.isna(row[col]):
                    lines.append(f"  - {col}: {row[col]:.4f}")
    
    if 'normalized' in comparison_results:
        lines.extend([
            "",
            "=" * 80,
            "NORMALIZED METRICS (Per-CDU)",
            "=" * 80,
        ])
        df = comparison_results['normalized']
        for _, row in df.iterrows():
            lines.append(f"\n{row['system'].upper()}:")
            for col in df.columns:
                if col not in ['system', 'num_cdus'] and not pd.isna(row[col]):
                    lines.append(f"  - {col}: {row[col]:.4f}")
    
    lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    return "\n".join(lines)
