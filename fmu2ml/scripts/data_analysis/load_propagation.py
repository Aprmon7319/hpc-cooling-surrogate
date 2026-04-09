import os, sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmu2ml.analysis.spatial_correlation.load_propagation import LoadPropagationAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run Load Propagation Analysis on FMU data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the FMU simulation data file (CSV or Parquet)')
    parser.add_argument('--output_dir', type=str, default='results/load_propagation',
                        help='Directory to save analysis results')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='Number of samples to use for analysis (default: 10000)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers. Default: cpu_count() - 1')
    
    args = parser.parse_args()
    
    # Validate input file
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print(f"Initializing Load Propagation Analyzer...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample size: {args.sample_size}")
    print(f"Workers: {args.n_workers if args.n_workers else 'auto'}")
    print()
    
    analyzer = LoadPropagationAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_workers=args.n_workers
    )
    
    try:
        analyzer.run_full_analysis(sample_size=args.sample_size)
        print("\n✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n✗ Analysis failed with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()