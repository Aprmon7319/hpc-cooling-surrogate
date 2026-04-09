#!/usr/bin/env python3
"""
Quick Start Example for FMU2ML

This script demonstrates the basic workflow:
1. Load system configuration
2. Generate sample data
3. Train a simple model
4. Make predictions
"""

import numpy as np
from pathlib import Path

# Import fmu2ml components
from fmu2ml import (
    get_system_config,
    ScenarioGenerator,
    PowerGenerator,
    create_model,
    Trainer,
    CoolingModelPredictor,
    setup_logger
)


def main():
    # Setup logging
    logger = setup_logger('quick_start', level='INFO')
    logger.info("=" * 80)
    logger.info("FMU2ML Quick Start Example")
    logger.info("=" * 80)
    
    # 1. Load system configuration
    logger.info("\n1. Loading system configuration...")
    config = get_system_config('marconi100', use_raps=True)
    logger.info(f"   System: {config.system_name}")
    logger.info(f"   CDUs: {config.num_cdus}")
    logger.info(f"   Power range: {config.min_power}-{config.max_power} kW")
    
    # 2. Generate sample data
    logger.info("\n2. Generating sample data...")
    
    # Generate power scenarios
    scenario_gen = ScenarioGenerator(
        num_cdus=config.num_cdus,
        min_power=config.min_power,
        max_power=config.max_power,
        seed=42
    )
    scenarios = scenario_gen.generate_scenarios(num_scenarios=10)
    logger.info(f"   Generated {len(scenarios)} scenarios")
    
    # Generate continuous power data
    power_gen = PowerGenerator(
        num_cdus=config.num_cdus,
        min_power=config.min_power,
        max_power=config.max_power
    )
    power_df = power_gen.generate_continuous_power(
        scenarios=scenarios,
        duration_hours=2.0,
        timestep_seconds=60
    )
    logger.info(f"   Generated {len(power_df)} timesteps of power data")
    
    # 3. Create and configure model
    logger.info("\n3. Creating FNO model...")
    model_config = config.to_model_config()
    model_config.modes = 8  # Smaller for quick demo
    model_config.width = 32
    model_config.num_layers = 2
    
    model = create_model('fno', model_config)
    logger.info(f"   Model created: {model.__class__.__name__}")
    logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Prepare simple training data (dummy for demo)
    logger.info("\n4. Preparing dummy training data...")
    n_samples = 100
    input_data = np.random.randn(n_samples, config.num_cdus + 1)  # Powers + Tamb
    output_data = np.random.randn(n_samples, config.num_cdus + 3)  # Outlets + facility
    
    logger.info(f"   Input shape: {input_data.shape}")
    logger.info(f"   Output shape: {output_data.shape}")
    
    # 5. Make a prediction (before training)
    logger.info("\n5. Making prediction with untrained model...")
    import torch
    sample_input = torch.randn(1, config.num_cdus + 1)
    with torch.no_grad():
        prediction = model(sample_input)
    logger.info(f"   Prediction shape: {prediction.shape}")
    
    # 6. Summary
    logger.info("\n" + "=" * 80)
    logger.info("Quick Start Complete!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Generate real data: python -m fmu2ml.scripts.generate_data")
    logger.info("2. Train model: python -m fmu2ml.scripts.train_model")
    logger.info("3. Evaluate: python -m fmu2ml.scripts.evaluate_model")
    logger.info("\nSee examples/notebooks/ for detailed tutorials")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()