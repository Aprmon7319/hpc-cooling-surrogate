import pandas as pd
from pathlib import Path
from ..visualization import (
    plot_power_patterns,
    plot_basic_statistics,
    plot_time_series_overview,
    plot_cdu_output_metrics,
    plot_cdu_means_with_dc_metrics,
    CDUVisualizer,
    StandardInputOutputVisualizer
)

def visualize_data(input_file: str, output_file: str = None, selected_cdus: list = None, num_cdus: int = 49):
    """
    Visualize FMU input and output data
    
    Parameters:
    -----------
    input_file : str
        Path to input parquet/csv file
    output_file : str, optional
        Path to output parquet/csv file
    selected_cdus : list, optional
        List of CDU IDs to highlight (default: [13, 16, 17, 38, 42])
    num_cdus : int
        Number of CDUs in the system (default: 49)
    """
    
    if selected_cdus is None:
        selected_cdus = [13, 16, 17, 38, 42]
    
    print("Loading input data...")
    # Load input data
    input_path = Path(input_file)
    if input_path.suffix == '.parquet':
        input_df = pd.read_parquet(input_file)
    else:
        input_df = pd.read_csv(input_file)
    
    print("\n=== INPUT DATA VISUALIZATIONS ===\n")
    
    # 1. Power patterns
    print("Plotting power patterns...")
    fig1 = plot_power_patterns(input_df, num_cdus=num_cdus)
    fig1.savefig('input_power_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved: input_power_patterns.png")
    
    # 2. Basic statistics
    print("Plotting basic statistics...")
    fig2 = plot_basic_statistics(input_df, num_cdus=num_cdus)
    fig2.savefig('input_statistics.png', dpi=150, bbox_inches='tight')
    print("  Saved: input_statistics.png")
    
    # 3. Time series overview
    print("Plotting time series overview...")
    fig3 = plot_time_series_overview(input_df, num_cdus=num_cdus)
    fig3.savefig('input_timeseries.png', dpi=150, bbox_inches='tight')
    print("  Saved: input_timeseries.png")
    


    # If output file provided, visualize outputs
    if output_file:
        print("\nLoading output data...")
        output_path = Path(output_file)
        if output_path.suffix == '.parquet':
            output_df = pd.read_parquet(output_file)
        else:
            output_df = pd.read_csv(output_file)
        
        print("\n=== OUTPUT DATA VISUALIZATIONS ===\n")
        
        print("Standard input visualization")
        visualizer = StandardInputOutputVisualizer(output_df)
        fig_std_input = visualizer.plot_inputs(
            rolling_windows=300,
            selected_cdus=selected_cdus,
            figsize=(20, 16)
        )
        fig_std_input.savefig('input_standard_overview.png', dpi=150, bbox_inches='tight')
        print("  Saved: input_standard_overview.png")
        
        fig_std_output = visualizer.plot_outputs(
            rolling_windows=300,
            selected_cdus=selected_cdus,
            figsize=(20, 16)
        )
        fig_std_output.savefig('output_standard_overview.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_standard_overview.png")

        fig_data_center_stats = visualizer.plot_datacenter_statistics(
            rolling_windows=300,
            figsize=(20, 16),

        )
        fig_data_center_stats.savefig('output_datacenter_statistics.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_datacenter_statistics.png")
        
        # 4. CDU output metrics
        print("Plotting CDU output metrics...")
        fig4 = plot_cdu_output_metrics(output_df, selected_cbs=selected_cdus, num_cdus=num_cdus)
        fig4.savefig('output_cdu_metrics.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_cdu_metrics.png")
        
        # 5. CDU means with datacenter metrics
        print("Plotting CDU means and DC metrics...")
        fig5 = plot_cdu_means_with_dc_metrics(output_df, compute_blocks=selected_cdus, num_cdus=num_cdus)
        fig5.savefig('output_dc_metrics.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_dc_metrics.png")
        
        # 6. Detailed CDU visualization
        print("Creating detailed CDU visualization...")
        visualizer = CDUVisualizer()
        fig6, _ = visualizer.plot_random_cdus_overview(
            output_df, 
            selected_cbs=selected_cdus,
            figsize=(20, 16)
        )
        fig6.savefig('output_detailed_cdus.png', dpi=150, bbox_inches='tight')
        print("  Saved: output_detailed_cdus.png")
    
    print("\n✓ Visualization complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize FMU input/output data")
    parser.add_argument('--input', type=str, required=True,
                       help='Input data file (parquet or csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output data file (parquet or csv)')
    parser.add_argument('--cdus', type=int, nargs='+', default=None,
                       help='CDU IDs to highlight (e.g., --cdus 13 16 17 38 42)')
    
    args = parser.parse_args()
    
    visualize_data(args.input, args.output, args.cdus)