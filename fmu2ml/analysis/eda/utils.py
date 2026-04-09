import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Optional, Dict, Any

# Import the existing data loader
try:
    from fmu2ml.data.processors.data_loader import DataLoader
    from fmu2ml.utils.io_utils import load_config
    HAS_DATA_LOADER = True
except ImportError:
    HAS_DATA_LOADER = False
    print("Warning: DataLoader not available. Using fallback loading methods.")


def load_simulation_data(
    data_path: str, 
    config: Optional[Dict[str, Any]] = None,
    use_data_loader: bool = True
) -> pd.DataFrame:
    """
    Load simulation data from file using the integrated DataLoader or fallback method.
    
    Args:
        data_path: Path to data file
        config: Optional configuration dictionary for loading parameters
        use_data_loader: Whether to use the DataLoader (if available)
        
    Returns:
        DataFrame with simulation data
    """
    data_path = Path(data_path)
    
    # Try to use DataLoader if available and requested
    if use_data_loader and HAS_DATA_LOADER:
        try:
            print(f"Loading data using DataLoader from {data_path}")
            data_loader = DataLoader(config=config or {})
            df = data_loader.load_data(str(data_path), format='auto')
            
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
        except Exception as e:
            print(f"Warning: DataLoader failed ({e}), falling back to direct loading")
    
    # Fallback: Direct loading based on file format
    print(f"Loading data directly from {data_path}")
    
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.feather':
        df = pd.read_feather(data_path)
    elif data_path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def _detect_outliers_for_column(col_data):
    """Detect outliers for a single column (global function for multiprocessing)."""
    col, data = col_data
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        return None
    
    try:
        # IQR method
        Q1 = data_clean.quantile(0.25)
        Q3 = data_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((data_clean < lower_bound) | (data_clean > upper_bound)).sum()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data_clean))
        z_outliers = (z_scores > 3).sum()
        
        return {
            'Variable': col,
            'Total_Points': len(data_clean),
            'IQR_Outliers': int(iqr_outliers),
            'IQR_Outliers_%': 100 * iqr_outliers / len(data_clean),
            'Z_Score_Outliers': int(z_outliers),
            'Z_Score_Outliers_%': 100 * z_outliers / len(data_clean)
        }
    except Exception as e:
        print(f"Error processing {col}: {e}")
        return None

def _process_cdu_wrapper(args):
    """Wrapper for processing a single CDU (global function for multiprocessing)."""
    cdu_idx, data_dict, output_patterns, cdu_dir = args
    
    try:
        # Create CDU-specific directory
        cdu_output_dir = Path(cdu_dir) / f'cdu_{cdu_idx:03d}'
        cdu_output_dir.mkdir(exist_ok=True)
        
        # Extract input parameters for this CDU
        # Q_flow input
        q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
        q_flow_data = data_dict.get(q_flow_col, None)
        
        # T_air input
        t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
        t_air_data = data_dict.get(t_air_col, None)
        
        # T_ext input (global - same for all CDUs)
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        t_ext_data = data_dict.get(t_ext_col, None)
        
        # Get time data
        time_data = data_dict.get('time', None)
        
        # Get output variables prefix
        cdu_prefix = output_patterns.get(
            'cdu_summary_prefix',
            'simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.'
        ).format(cdu_idx)
        
        cdu_metrics = output_patterns.get(
            'cdu_metrics',
            ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
             'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
             'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        )
        
        # Process each metric separately
        for metric in cdu_metrics:
            var_name = f'{cdu_prefix}{metric}'
            
            if var_name not in data_dict:
                continue
            
            data_var = data_dict[var_name]
            
            if len(data_var) == 0:
                continue
            
            # Create comprehensive plot for this variable with inputs
            _create_single_variable_plot(
                data=data_var, 
                var_name=var_name, 
                metric_name=metric,
                cdu_idx=cdu_idx,
                output_dir=cdu_output_dir,
                time_data=time_data,
                q_flow=q_flow_data,
                t_air=t_air_data,
                t_ext=t_ext_data
            )
        
        return True
    except Exception as e:
        print(f"Error processing CDU {cdu_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _create_single_variable_plot(data, var_name, metric_name, cdu_idx, output_dir, time_data,
                                 q_flow=None, t_air=None, t_ext=None):    
    """Create comprehensive plot for a single variable with input overlays."""
    try:
        # Check if we have valid input data
        has_inputs = any([
            q_flow is not None and len(q_flow) > 0,
            t_air is not None and len(t_air) > 0,
            t_ext is not None and len(t_ext) > 0
        ])
        
        # Adjust figure layout based on whether we have inputs
        if has_inputs:
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, 
                                 height_ratios=[1, 0.8, 1, 1])
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        x_axis = time_data if time_data is not None else range(len(data))
        
        # ===================================================================
        # 1. OUTPUT TIME SERIES (top row)
        # ===================================================================
        ax1_main = fig.add_subplot(gs[0, :])
        
        ax1_main.plot(x_axis, data, alpha=0.8, linewidth=1.5, 
                     color='steelblue', label=f'{metric_name}')
        ax1_main.set_ylabel(f'{metric_name}\n{_get_unit_for_metric(metric_name)}', 
                           fontsize=10, color='steelblue', fontweight='bold')
        ax1_main.tick_params(axis='y', labelcolor='steelblue')
        ax1_main.grid(alpha=0.3)
        
        # Add statistical annotations
        mean_val = data.mean()
        std_val = data.std()
        ax1_main.axhline(mean_val, color='darkblue', linestyle='--', alpha=0.6, 
                        linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax1_main.axhline(mean_val + std_val, color='orange', linestyle=':', 
                        alpha=0.5, linewidth=1.5, label=f'±1σ')
        ax1_main.axhline(mean_val - std_val, color='orange', linestyle=':', 
                        alpha=0.5, linewidth=1.5)
        
        ax1_main.set_title(f'CDU {cdu_idx} - {metric_name} (Output Variable)', 
                          fontsize=12, fontweight='bold')
        ax1_main.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        if has_inputs:
            plt.setp(ax1_main.get_xticklabels(), visible=False)
        else:
            ax1_main.set_xlabel('Time' if time_data is not None else 'Sample', fontsize=10)
        
        # ===================================================================
        # 2. INPUT PARAMETERS TIME SERIES (second row, only if inputs exist)
        # ===================================================================
        if has_inputs:
            ax1_inputs = fig.add_subplot(gs[1, :], sharex=ax1_main)
            
            input_lines = []
            input_labels = []
            
            # Plot Q_flow on primary y-axis
            if q_flow is not None and len(q_flow) > 0:
                line = ax1_inputs.plot(x_axis, q_flow, alpha=0.75, linewidth=1.3, 
                                       color='crimson', label='Q_flow (Heat Load)', 
                                       marker='', linestyle='-')
                input_lines.append(line[0])
                input_labels.append(f'Q_flow: μ={q_flow.mean():.1f} kW, σ={q_flow.std():.1f}')
                ax1_inputs.set_ylabel('Heat Load [kW]', fontsize=9, color='crimson', fontweight='bold')
                ax1_inputs.tick_params(axis='y', labelcolor='crimson')
            
            # Plot T_air on secondary y-axis
            if t_air is not None and len(t_air) > 0:
                ax1_temp = ax1_inputs.twinx()
                line = ax1_temp.plot(x_axis, t_air, alpha=0.75, linewidth=1.3, 
                                    color='forestgreen', label='T_air', 
                                    marker='', linestyle='-')
                input_lines.append(line[0])
                input_labels.append(f'T_air: μ={t_air.mean():.1f}°C, σ={t_air.std():.1f}')
                ax1_temp.set_ylabel('Air Temp [°C]', fontsize=9, color='forestgreen', fontweight='bold')
                ax1_temp.tick_params(axis='y', labelcolor='forestgreen')
            
            # Plot T_ext on tertiary y-axis
            if t_ext is not None and len(t_ext) > 0:
                ax1_ext = ax1_inputs.twinx()
                if t_air is not None and len(t_air) > 0:
                    ax1_ext.spines['right'].set_position(('outward', 60))
                line = ax1_ext.plot(x_axis, t_ext, alpha=0.75, linewidth=1.3, 
                                   color='darkorange', label='T_ext', 
                                   marker='', linestyle='-')
                input_lines.append(line[0])
                input_labels.append(f'T_ext: μ={t_ext.mean():.1f}°C, σ={t_ext.std():.1f}')
                ax1_ext.set_ylabel('External Temp [°C]', fontsize=9, color='darkorange', fontweight='bold')
                ax1_ext.tick_params(axis='y', labelcolor='darkorange')
            
            ax1_inputs.set_xlabel('Time' if time_data is not None else 'Sample', fontsize=10)
            ax1_inputs.set_title('Input Parameters', fontsize=11, fontweight='bold')
            ax1_inputs.grid(alpha=0.3)
            
            # Combined legend
            if input_lines:
                ax1_inputs.legend(input_lines, input_labels, loc='upper left', 
                                fontsize=8, framealpha=0.9, ncol=len(input_lines))
        
        # ===================================================================
        # Adjust row index for remaining plots
        # ===================================================================
        row_offset = 2 if has_inputs else 1
        
        # ===================================================================
        # 3. HISTOGRAM (third/second row, left)
        # ===================================================================
        ax2 = fig.add_subplot(gs[row_offset, 0])
        ax2.hist(data, bins=50, alpha=0.7, edgecolor='black', color='steelblue', density=False)
        ax2.set_xlabel('Value', fontsize=9)
        ax2.set_ylabel('Frequency', fontsize=9)
        ax2.set_title('Distribution', fontsize=10, fontweight='bold')
        ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        ax2.axvline(data.median(), color='green', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Median: {data.median():.2f}')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis='y')
        
        # ===================================================================
        # 4. BOX PLOT (third/second row, center)
        # ===================================================================
        ax3 = fig.add_subplot(gs[row_offset, 1])
        bp = ax3.boxplot([data], vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
        ax3.set_ylabel('Value', fontsize=9)
        ax3.set_title('Box Plot', fontsize=10, fontweight='bold')
        ax3.set_xticklabels([metric_name], fontsize=8)
        ax3.grid(alpha=0.3, axis='y')
        
        # ===================================================================
        # 5. QQ PLOT (third/second row, right)
        # ===================================================================
        ax4 = fig.add_subplot(gs[row_offset, 2])
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality)', fontsize=10, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax4.set_ylabel('Sample Quantiles', fontsize=9)
        
        # Add normality test result
        if len(data) <= 5000:
            _, p_value = stats.shapiro(data)
            normality_text = f'Shapiro p={p_value:.4f}\n{"Normal" if p_value > 0.05 else "Non-normal"}'
            ax4.text(0.05, 0.95, normality_text, transform=ax4.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # ===================================================================
        # 6. INPUT-OUTPUT CORRELATION (fourth/third row, left)
        # ===================================================================
        ax5 = fig.add_subplot(gs[row_offset + 1, 0])
        
        if q_flow is not None and len(q_flow) > 0:
            # Scatter plot with Q_flow
            ax5.scatter(q_flow, data, alpha=0.3, s=15, c='blue', edgecolors='none')
            
            # Add regression line
            if len(q_flow) > 1:
                z = np.polyfit(q_flow, data, 1)
                p = np.poly1d(z)
                q_sorted = np.sort(q_flow)
                ax5.plot(q_sorted, p(q_sorted), "r--", alpha=0.8, linewidth=2.5, 
                        label=f'y={z[0]:.2e}x+{z[1]:.2f}')
            
            corr = np.corrcoef(q_flow, data)[0, 1]
            ax5.set_xlabel('Q_flow [kW]', fontsize=9, fontweight='bold')
            ax5.set_ylabel(f'{metric_name}', fontsize=9)
            ax5.set_title(f'vs Q_flow (r={corr:.3f})', fontsize=10, fontweight='bold')
            ax5.legend(fontsize=8)
        else:
            # Fallback: CDF
            data_sorted = np.sort(data)
            cumulative = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax5.plot(data_sorted, cumulative, linewidth=2, color='steelblue')
            ax5.set_xlabel('Value', fontsize=9)
            ax5.set_ylabel('Cumulative Probability', fontsize=9)
            ax5.set_title('CDF', fontsize=10, fontweight='bold')
        
        ax5.grid(alpha=0.3)
        
        # ===================================================================
        # 7. CORRELATION HEATMAP (fourth/third row, center)
        # ===================================================================
        ax6 = fig.add_subplot(gs[row_offset + 1, 1])
        
        if has_inputs:
            # Build correlation dataframe
            corr_dict = {'Output': data}
            col_names = ['Output']
            
            if q_flow is not None and len(q_flow) > 0:
                corr_dict['Q_flow'] = q_flow
                col_names.append('Q_flow')
            if t_air is not None and len(t_air) > 0:
                corr_dict['T_air'] = t_air
                col_names.append('T_air')
            if t_ext is not None and len(t_ext) > 0:
                corr_dict['T_ext'] = t_ext
                col_names.append('T_ext')
            
            # Create correlation matrix
            corr_df = pd.DataFrame(corr_dict)
            corr_matrix = corr_df.corr()
            
            # Plot heatmap
            im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', 
                           vmin=-1, vmax=1, interpolation='nearest')
            
            ax6.set_xticks(range(len(col_names)))
            ax6.set_yticks(range(len(col_names)))
            ax6.set_xticklabels(col_names, rotation=45, ha='right', fontsize=9)
            ax6.set_yticklabels(col_names, fontsize=9)
            
            # Add correlation values as text
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color=color, 
                                   fontsize=9, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', fontsize=8)
            ax6.set_title('Input-Output Correlations', fontsize=10, fontweight='bold')
        else:
            # Fallback: Violin plot
            parts = ax6.violinplot([data], positions=[0], showmeans=True, showmedians=True, 
                                  widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor('lightgreen')
                pc.set_alpha(0.7)
            ax6.set_ylabel('Value', fontsize=9)
            ax6.set_title('Violin Plot', fontsize=10, fontweight='bold')
            ax6.set_xticks([])
            ax6.grid(alpha=0.3, axis='y')
        
        # ===================================================================
        # 8. STATISTICAL SUMMARY (fourth/third row, right)
        # ===================================================================
        ax7 = fig.add_subplot(gs[row_offset + 1, 2])
        ax7.axis('off')
        
        # Calculate statistics
        stats_text = [
            f"📊 CDU {cdu_idx} Statistics",
            f"Variable: {metric_name}",
            "=" * 35,
            "",
            "OUTPUT STATISTICS:",
            "-" * 35,
            f"Count:     {len(data):,}",
            f"Mean:      {data.mean():.4f}",
            f"Std:       {data.std():.4f}",
            f"Min:       {data.min():.4f}",
            f"25%:       {np.percentile(data, 25):.4f}",
            f"Median:    {data.median():.4f}",
            f"75%:       {np.percentile(data, 75):.4f}",
            f"Max:       {data.max():.4f}",
            f"Range:     {data.max() - data.min():.4f}",
            f"CV:        {data.std()/data.mean():.4f}" if data.mean() != 0 else "CV:        N/A",
            f"Skewness:  {data.skew():.4f}",
            f"Kurtosis:  {data.kurtosis():.4f}",
        ]
        
        # Add input statistics if available
        if has_inputs:
            stats_text.extend(["", "INPUT STATISTICS:", "-" * 35])
            
            if q_flow is not None and len(q_flow) > 0:
                corr_q = np.corrcoef(q_flow, data)[0, 1]
                stats_text.extend([
                    f"Q_flow:    μ={q_flow.mean():.2f}, σ={q_flow.std():.2f}",
                    f"  Corr:    {corr_q:.3f}"
                ])
            
            if t_air is not None and len(t_air) > 0:
                corr_t = np.corrcoef(t_air, data)[0, 1]
                stats_text.extend([
                    f"T_air:     μ={t_air.mean():.2f}, σ={t_air.std():.2f}",
                    f"  Corr:    {corr_t:.3f}"
                ])
            
            if t_ext is not None and len(t_ext) > 0:
                corr_e = np.corrcoef(t_ext, data)[0, 1]
                stats_text.extend([
                    f"T_ext:     μ={t_ext.mean():.2f}, σ={t_ext.std():.2f}",
                    f"  Corr:    {corr_e:.3f}"
                ])
        
        ax7.text(0.05, 0.95, '\n'.join(stats_text), 
                transform=ax7.transAxes,
                fontsize=8,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.8))
        
        # ===================================================================
        # Overall title
        # ===================================================================
        input_info = []
        if q_flow is not None and len(q_flow) > 0:
            input_info.append("Q_flow")
        if t_air is not None and len(t_air) > 0:
            input_info.append("T_air")
        if t_ext is not None and len(t_ext) > 0:
            input_info.append("T_ext")
        
        input_str = " + ".join(input_info) if input_info else "No Inputs"
        
        plt.suptitle(f'Comprehensive Analysis: CDU {cdu_idx} - {metric_name}\nInputs: {input_str}', 
                    fontsize=14, fontweight='bold', y=0.998)
        
        # Save figure
        safe_metric_name = metric_name.replace('/', '_').replace('.', '_')
        plt.savefig(output_dir / f'{safe_metric_name}_with_inputs.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot for {metric_name}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def _get_unit_for_metric(metric_name: str) -> str:
    """Get appropriate unit for a metric."""
    if 'V_flow' in metric_name:
        return 'GPM'
    elif 'T_' in metric_name:
        return '°C'
    elif 'p_' in metric_name:
        return 'psig'
    elif 'W_flow' in metric_name or 'kW' in metric_name:
        return 'kW'
    elif 'Q_flow' in metric_name:
        return 'kW'
    else:
        return 'Value'


