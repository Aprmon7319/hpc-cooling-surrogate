#!/usr/bin/env python3
"""
Inference Example

Demonstrates how to use a trained model for inference:
- Load trained model
- Prepare input data
- Make predictions
- Post-process outputs
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from fmu2ml import (
    get_system_config,
    CoolingModelPredictor,
    BatchProcessor,
    setup_logger
)
from fmu2ml.data.utils import InputFormatter, OutputParser


def main():
    logger = setup_logger('inference_example', level='INFO')
    logger.info("=" * 80)
    logger.info("FMU2ML Inference Example")
    logger.info("=" * 80)
    
    # Configuration
    model_path = 'outputs/train_example/final_model.pt'
    system_name = 'marconi100'
    
    # 1. Load system configuration
    logger.info("\n1. Loading configuration...")
    config = get_system_config(system_name, use_raps=True)
    logger.info(f"   System: {config.system_name}")
    
    # 2. Check if model exists
    logger.info("\n2. Loading trained model...")
    if not Path(model_path).exists():
        logger.warning(f"   Model not found: {model_path}")
        logger.info("   Creating dummy model for demonstration...")
        
        # Create dummy model
        from fmu2ml import create_model
        model_config = config.to_model_config()
        predictor = CoolingModelPredictor(
            model=create_model('fno', model_config),
            model_type='fno'
        )
    else:
        # Load trained model
        predictor = CoolingModelPredictor(
            model_path=model_path,
            model_type='fno'
        )
        logger.info(f"   Model loaded from: {model_path}")
    
    # 3. Prepare input data
    logger.info("\n3. Preparing input data...")
    formatter = InputFormatter(num_cdus=config.num_cdus, timestep=60.0)
    
    # Create sample power data (24 hours)
    n_timesteps = 24 * 60  # 1 minute timestep
    power_data = np.random.uniform(
        config.min_power,
        config.max_power,
        size=(n_timesteps, config.num_cdus)
    )
    
    # Format power data
    power_df = formatter.format_power_data(power_data)
    
    # Add ambient temperature
    temp_data = 20.0 + 5.0 * np.sin(np.linspace(0, 2*np.pi, n_timesteps))  # Daily cycle
    temp_df = formatter.format_temperature_data(temp_data)
    
    # Merge
    input_df = formatter.merge_inputs(power_df, temp_df)
    logger.info(f"   Input shape: {input_df.shape}")
    logger.info(f"   Duration: {n_timesteps / 60:.1f} hours")
    
    # 4. Make predictions
    logger.info("\n4. Making predictions...")
    
    # Convert to tensor
    input_cols = [col for col in input_df.columns if col != 'time']
    input_tensor = torch.tensor(input_df[input_cols].values, dtype=torch.float32)
    
    # Predict
    logger.info("   Running inference...")
    predictions = predictor.predict(input_tensor)
    logger.info(f"   Predictions shape: {predictions.shape}")
    
    # 5. Post-process outputs
    logger.info("\n5. Post-processing outputs...")
    parser = OutputParser(num_cdus=config.num_cdus)
    
    # Create output DataFrame
    output_cols = []
    # Facility vars
    output_cols.extend(['Facility_Tsup', 'Facility_Tret', 'Facility_mdot'])
    # CDU outlets
    output_cols.extend([f'CDU_{i}_Tout' for i in range(1, config.num_cdus + 1)])
    
    output_df = pd.DataFrame(predictions, columns=output_cols)
    output_df['time'] = input_df['time'].values
    
    # Parse and add features
    parsed_output = parser.parse_simulation_output(output_df, extract_features=True)
    logger.info(f"   Parsed output shape: {parsed_output.shape}")
    
    # Compute statistics
    stats = parser.compute_statistics(parsed_output)
    
    logger.info("\n   Output Statistics:")
    for var in ['Facility_Tsup', 'Facility_Tret', 'CDU_Tout_mean']:
        if var in stats:
            logger.info(f"     {var}:")
            logger.info(f"       Mean: {stats[var]['mean']:.2f}")
            logger.info(f"       Std:  {stats[var]['std']:.2f}")
            logger.info(f"       Range: [{stats[var]['min']:.2f}, {stats[var]['max']:.2f}]")
    
    # 6. Batch processing example
    logger.info("\n6. Batch processing example...")
    
    # Split into chunks
    chunks = formatter.split_into_chunks(input_df, chunk_duration_hours=4.0)
    logger.info(f"   Split into {len(chunks)} chunks")
    
    # Create batch processor
    batch_processor = BatchProcessor(predictor=predictor)
    
    # Process chunks
    logger.info("   Processing chunks...")
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_input = torch.tensor(
            chunk[input_cols].values,
            dtype=torch.float32
        )
        chunk_pred = batch_processor.process_batch(chunk_input)
        chunk_results.append(chunk_pred)
        logger.info(f"     Chunk {i+1}/{len(chunks)}: {chunk_pred.shape}")
    
    # 7. Save results
    logger.info("\n7. Saving results...")
    output_dir = Path('outputs/inference_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = output_dir / 'predictions.parquet'
    parsed_output.to_parquet(pred_path, index=False)
    logger.info(f"   Predictions saved to: {pred_path}")
    
    # Save statistics
    stats_df = pd.DataFrame([{
        f"{var}_{stat}": value
        for var, var_stats in stats.items()
        for stat, value in var_stats.items()
    }])
    stats_path = output_dir / 'statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"   Statistics saved to: {stats_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Inference Example Complete!")
    logger.info("=" * 80)
    logger.info(f"\nProcessed {n_timesteps} timesteps")
    logger.info(f"Average Facility Supply Temp: {stats['Facility_Tsup']['mean']:.2f}°C")
    logger.info(f"Average Facility Return Temp: {stats['Facility_Tret']['mean']:.2f}°C")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()