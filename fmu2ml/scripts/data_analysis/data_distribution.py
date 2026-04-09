import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fmu2ml.analysis import DataQualityAnalyzer, load_simulation_data


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 EDA on cooling model data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to simulation data file')
    parser.add_argument('--num_cdus', type=int, required=True,
                       help='Number of CDUs in the configuration')
    parser.add_argument('--output_dir', type=str, default='eda_results/phase1',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to EDA configuration file (optional)')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        num_cdus = config.get('NUM_CDUS', args.num_cdus)
        output_dir = config.get('output_dir', args.output_dir)
    else:
        num_cdus = args.num_cdus
        output_dir = args.output_dir
    
    print(f"Loading data from: {args.data_path}")
    data = load_simulation_data(args.data_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer(
        data=data,
        num_cdus=num_cdus,
        output_dir=output_dir
    )
    
    # Run full Phase 1 analysis
    analyzer.run_full_analysis()
    
    print("\n" + "="*60)
    print("Phase 1 EDA Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()