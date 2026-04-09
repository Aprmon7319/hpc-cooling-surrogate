import argparse
import logging
import pandas as pd
from pathlib import Path
from fmu2ml.simulation import FMUOutputGenerator
from raps.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_fmu_output(
    input_file: str,
    output_file: str = None,
    system_name: str = 'marconi100',
    stabilization_hours: int = 3,
    stabilization_threshold: float = 0.1,
    step_size: int = 1
):
    """Generate FMU output from a single input file"""
    
    # Load input data
    logger.info(f"Loading input: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Input shape: {df.shape}")
    
    # Set output file
    if output_file is None:
        actual_hours = len(df) * step_size / 3600
        output_file = f"fmu_output_{actual_hours:.1f}hrs.parquet"
    
    # Load configuration
    config = ConfigManager(system_name=system_name).get_config()
    config_overrides = {k: v for k, v in config.items() if k != 'system_name'}
    
    # Generate output
    logger.info("Starting FMU simulation...")
    with FMUOutputGenerator(
        system_name=system_name,
        stabilization_hours=stabilization_hours,
        stabilization_threshold=stabilization_threshold,
        step_size=step_size,
        output_dir=str(Path(output_file).parent),
        **config_overrides
    ) as generator:
        
        output_df = generator.generate_from_input(
            input_data=df,
            output_file=output_file,
            save_stabilization=False
        )
    
    logger.info(f"✓ Output saved to: {output_file}")
    logger.info(f"Output shape: {output_df.shape}")
    
    return output_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FMU output from single input file")
    
    parser.add_argument("--input_file", type=str, help="Input parquet file")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Output parquet file (default: auto-generated)")
    parser.add_argument("--system_name", type=str, default="marconi100",
                       help="System configuration name")
    parser.add_argument("--stabilization_hours", type=int, default=3,
                       help="Hours for stabilization phase")
    parser.add_argument("--stabilization_threshold", type=float, default=0.1,
                       help="Threshold for stabilization detection")
    parser.add_argument("--step_size", type=int, default=1,
                       help="Step size for processing")
    
    args = parser.parse_args()
    
    generate_fmu_output(
        input_file=args.input_file,
        output_file=args.output_file,
        system_name=args.system_name,
        stabilization_hours=args.stabilization_hours,
        stabilization_threshold=args.stabilization_threshold,
        step_size=args.step_size
    )