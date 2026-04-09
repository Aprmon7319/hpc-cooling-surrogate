import os
import sys
import pandas as pd
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmu2ml.analysis.spatial_correlation.basic_correlations import SpatialCorrelationAnalyzer


def inspect_data(data_path: str):
    """Inspect the data structure and available metrics."""
    print("="*60)
    print("DATA INSPECTION")
    print("="*60)
    
    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError("Data must be CSV or Parquet format")
    
    print(f"\nData shape: {data.shape}")
    print(f"Number of columns: {len(data.columns)}")
    
    # Find CDU-related columns
    cdu_cols = [col for col in data.columns if 'computeBlock[' in col and 'cdu[1].summary.' in col]
    print(f"\nFound {len(cdu_cols)} CDU summary columns")
    
    # Infer number of CDUs
    cdu_indices = set()
    for col in cdu_cols:
        try:
            idx = int(col.split('computeBlock[')[1].split(']')[0])
            cdu_indices.add(idx)
        except:
            pass
    
    num_cdus = len(cdu_indices)
    print(f"Number of CDUs detected: {num_cdus}")
    print(f"CDU indices: {sorted(cdu_indices)}")
    
    # Extract unique metrics
    metrics = set()
    for col in cdu_cols:
        metric = col.split('summary.')[-1]
        metrics.add(metric)
    
    print(f"\nAvailable metrics ({len(metrics)}):")
    for metric in sorted(metrics):
        print(f"  - {metric}")
    
    # Show sample data
    print("\nSample CDU columns (first 5):")
    for col in cdu_cols[:5]:
        print(f"  {col}")
        print(f"    Sample values: {data[col].head(3).values}")
        print(f"    Data type: {data[col].dtype}")
        print(f"    Non-null: {data[col].notna().sum()}/{len(data)}")
    
    return data, num_cdus, sorted(metrics)


def main():
    parser = argparse.ArgumentParser(description='Inspect data and run spatial correlation analysis')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the FMU simulation data')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save results')
    parser.add_argument('--method', type=str, default='pearson',
                       choices=['pearson', 'spearman'])
    parser.add_argument('--inspect_only', action='store_true',
                       help='Only inspect data without running analysis')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Specific metrics to analyze (space-separated)')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of parallel workers. Default: cpu_count() - 1')
    
    args = parser.parse_args()
    
    # Inspect data
    data, num_cdus, available_metrics = inspect_data(args.data_path)
    
    if args.inspect_only:
        print("\n✓ Inspection complete.")
        return
    
    # Determine metrics to analyze
    if args.metrics:
        metrics_to_analyze = [m for m in args.metrics if m in available_metrics]
        missing = set(args.metrics) - set(metrics_to_analyze)
        if missing:
            print(f"\n⚠ Warning: These metrics were not found: {missing}")
    else:
        # Use a sensible default subset
        default_metrics = [
            'T_prim_r_C', 'T_prim_s_C', 'T_sec_r_C', 'T_sec_s_C',
            'V_flow_prim_GPM', 'V_flow_sec_GPM', 
            'W_flow_CDUP_kW',
            'p_prim_r_psig', 'p_prim_s_psig'
        ]
        metrics_to_analyze = [m for m in default_metrics if m in available_metrics]
    
    if not metrics_to_analyze:
        print("\n❌ No valid metrics to analyze!")
        return
    
    print(f"\n{'='*60}")
    print(f"SPATIAL CORRELATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Analyzing {len(metrics_to_analyze)} metrics:")
    for m in metrics_to_analyze:
        print(f"  - {m}")
    print(f"Workers: {args.n_workers if args.n_workers else 'auto'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    analyzer = SpatialCorrelationAnalyzer(num_cdus=num_cdus, n_workers=args.n_workers)
    
    results = analyzer.analyze_all_metrics(
        data=data,
        metrics=metrics_to_analyze,
        method=args.method,
        save_dir=args.output_dir
    )
    
    # Save summary
    if results:
        summary_data = []
        for metric, result in results.items():
            hyp = result['hypothesis_results']
            summary_data.append({
                'metric': metric,
                'correlation_range': hyp['correlation_range'],
                'decay_rate': hyp['decay_rate'],
                'decay_r_squared': hyp['decay_r_squared'],
                'nearby_corr': hyp['nearby_correlation'],
                'distant_corr': hyp['distant_correlation'],
                'clustering_ratio': hyp['clustering_ratio'],
                'hypothesis_ok': hyp['hypothesis_supported']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(args.output_dir, 'spatial_correlation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()