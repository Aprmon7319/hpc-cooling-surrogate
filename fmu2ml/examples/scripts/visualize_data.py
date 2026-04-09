import argparse
from pathlib import Path
import pandas as pd

from fmu2ml.visualization import (
    InputVisualizer, OutputVisualizer, CDUVisualizer, 
    ComparisonPlotter, get_default_config
)


def visualize_input_data(args):
    """Visualize input data."""
    config = get_default_config(args.system)
    visualizer = InputVisualizer(config)
    
    # Load input data
    df = pd.read_parquet(args.input_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating power pattern plots...")
    visualizer.plot_power_patterns(df, save_path=output_dir / "power_patterns.png")
    
    print("Generating anomaly detection plots...")
    visualizer.plot_anomaly_detection(df, save_path=output_dir / "anomalies.png")
    
    if args.scenario_file:
        scenario_df = pd.read_parquet(args.scenario_file)
        print("Generating time series with scenarios...")
        visualizer.plot_time_series_with_scenarios(
            df, scenario_df, 
            save_path=output_dir / "time_series_scenarios.png"
        )
    
    if args.compare_cdus:
        compute_blocks = [int(x) for x in args.compare_cdus.split(',')]
        print(f"Generating CDU comparison for blocks: {compute_blocks}...")
        visualizer.plot_cdu_power_comparison(
            df, compute_blocks,
            save_path=output_dir / "cdu_comparison.png"
        )
    
    print(f"Input visualizations saved to {output_dir}")


def visualize_output_data(args):
    """Visualize output data."""
    config = get_default_config(args.system)
    visualizer = OutputVisualizer(config)
    
    # Load output data
    df = pd.read_parquet(args.output_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selected_cbs = None
    if args.selected_cdus:
        selected_cbs = [int(x) for x in args.selected_cdus.split(',')]
    
    print("Generating CDU metrics grid...")
    visualizer.plot_cdu_metrics_grid(
        df, selected_cbs=selected_cbs,
        save_path=output_dir / "cdu_metrics_grid.png"
    )
    
    print("Generating CDU metrics heatmap...")
    visualizer.plot_cdu_metrics_heatmap(
        df, save_path=output_dir / "cdu_metrics_heatmap.png"
    )
    
    if args.compare_cdus:
        compute_blocks = [int(x) for x in args.compare_cdus.split(',')]
        print("Generating datacenter summary...")
        visualizer.plot_datacenter_summary(
            df, compute_blocks,
            save_path=output_dir / "datacenter_summary.png"
        )
    
    print(f"Output visualizations saved to {output_dir}")


def visualize_cdu_detailed(args):
    """Visualize detailed CDU data."""
    config = get_default_config(args.system)
    visualizer = CDUVisualizer(config)
    
    # Load data
    df = pd.read_parquet(args.data_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compute_block:
        print(f"Generating detailed view for CDU {args.compute_block}...")
        visualizer.plot_single_cdu_detailed(
            df, args.compute_block,
            save_path=output_dir / f"cdu_{args.compute_block}_detailed.png"
        )
    else:
        selected_cbs = None
        if args.selected_cdus:
            selected_cbs = [int(x) for x in args.selected_cdus.split(',')]
        
        print("Generating CDU overview...")
        visualizer.plot_random_cdus_overview(
            df, n_cdus=args.n_cdus, selected_cbs=selected_cbs,
            save_path=output_dir / "cdu_overview.png"
        )
    
    print(f"CDU visualizations saved to {output_dir}")


def compare_models(args):
    """Compare FMU and ML model outputs."""
    config = get_default_config(args.system)
    plotter = ComparisonPlotter(config)
    
    # Load FMU data
    fmu_df = pd.read_parquet(args.fmu_file)
    
    # Load ML predictions
    ml_predictions = {}
    for model_file in args.ml_files:
        model_name = Path(model_file).stem
        ml_predictions[model_name] = pd.read_parquet(model_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compute_blocks = [int(x) for x in args.compare_cdus.split(',')]
    
    print("Generating model comparison plot...")
    plotter.plot_model_comparison(
        fmu_df, ml_predictions, compute_blocks,
        save_path=output_dir / "model_comparison.png"
    )
    
    print("Generating error distributions...")
    plotter.plot_error_distributions(
        fmu_df, ml_predictions, compute_blocks,
        save_path=output_dir / "error_distributions.png"
    )
    
    print(f"Comparison plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize datacenter cooling data')
    parser.add_argument('--system', type=str, default='marconi100',
                       help='System name (marconi100, summit, etc.)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for plots')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization type')
    
    # Input visualization
    input_parser = subparsers.add_parser('input', help='Visualize input data')
    input_parser.add_argument('input_file', help='Input parquet file')
    input_parser.add_argument('--scenario_file', help='Scenario data file')
    input_parser.add_argument('--compare_cdus', help='Comma-separated CDU IDs to compare')
    
    # Output visualization
    output_parser = subparsers.add_parser('output', help='Visualize output data')
    output_parser.add_argument('output_file', help='Output parquet file')
    output_parser.add_argument('--selected_cdus', help='Comma-separated CDU IDs')
    output_parser.add_argument('--compare_cdus', help='Comma-separated CDU IDs for summary')
    
    # CDU detailed visualization
    cdu_parser = subparsers.add_parser('cdu', help='Visualize CDU details')
    cdu_parser.add_argument('data_file', help='Data parquet file')
    cdu_parser.add_argument('--compute_block', type=int, 
                           help='Single compute block for detailed view')
    cdu_parser.add_argument('--selected_cdus', help='Comma-separated CDU IDs')
    cdu_parser.add_argument('--n_cdus', type=int, default=5, 
                           help='Number of CDUs for overview')
    
    # Model comparison
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('fmu_file', help='FMU output parquet file')
    compare_parser.add_argument('ml_files', nargs='+', help='ML model output files')
    compare_parser.add_argument('--compare_cdus', required=True,
                               help='Comma-separated CDU IDs to compare')
    
    args = parser.parse_args()
    
    if args.command == 'input':
        visualize_input_data(args)
    elif args.command == 'output':
        visualize_output_data(args)
    elif args.command == 'cdu':
        visualize_cdu_detailed(args)
    elif args.command == 'compare':
        compare_models(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()