import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmu2ml.analysis.spatial_correlation.temperature_gradient import TemperatureGradientAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Run Temperature Gradient Analysis on FMU data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
        
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the FMU simulation data file (Parquet format)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/temperature_gradient',
        help='Directory to save analysis results (default: results/temperature_gradient)'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=10000,
        help='Number of samples to use for analysis. Use -1 for full dataset (default: 10000)'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers. Default: cpu_count() - 1'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    if data_path.suffix not in ['.parquet', '.pq']:
        print(f"Warning: Expected .parquet file, got {data_path.suffix}")
        print("Attempting to load anyway...")
    
    # Determine sample size
    sample_size = None if args.sample_size == -1 else args.sample_size
    
    # Initialize analyzer
    print(f"Initializing Temperature Gradient Analyzer...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample size: {'Full dataset' if sample_size is None else sample_size}")
    print(f"Workers: {args.n_workers if args.n_workers else 'auto'}")
    print()
    
    analyzer = TemperatureGradientAnalyzer(
        data_path=str(data_path),
        output_dir=args.output_dir,
        n_workers=args.n_workers
    )
    
    # Run analysis
    try:
        analyzer.run_full_analysis(sample_size=sample_size)
        print("\n✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Analysis failed with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()