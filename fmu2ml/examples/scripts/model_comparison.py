#!/usr/bin/env python3
"""
Model Comparison Example

Demonstrates how to compare multiple models:
- Load multiple trained models
- Run predictions on same test data
- Compare metrics and physics constraints
- Generate comparison plots
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from fmu2ml import (
    get_system_config,
    CoolingModelPredictor,
    ModelComparator,
    MetricsCalculator,
    PhysicsValidator,
    setup_logger
)


def main():
    logger = setup_logger('comparison_example', level='INFO')
    logger.info("=" * 80)
    logger.info("FMU2ML Model Comparison Example")
    logger.info("=" * 80)
    
    # Configuration
    system_name = 'marconi100'
    model_paths = {
        'FNO': 'checkpoints/fno/best.pt',
        'Hybrid-FNO': 'checkpoints/hybrid_fno/best.pt',
        'DeepONet': 'checkpoints/deeponet/best.pt'
    }
    output_dir = Path('outputs/comparison_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load configuration
    logger.info("\n1. Loading configuration...")
    config = get_system_config(system_name, use_raps=True)
    
    # 2. Load models
    logger.info("\n2. Loading models...")
    predictors = {}
    
    for name, path in model_paths.items():
        if Path(path).exists():
            try:
                predictor = CoolingModelPredictor(
                    model_path=path,
                    model_type=name.lower().replace('-', '_')
                )
                predictors[name] = predictor
                logger.info(f"   ✓ {name} loaded")
            except Exception as e:
                logger.warning(f"   ✗ {name} failed to load: {e}")
        else:
            logger.warning(f"   ✗ {name} not found: {path}")
    
    if len(predictors) == 0:
        logger.warning("   No models found, creating dummy models for demo...")
        from fmu2ml import create_model
        
        model_config = config.to_model_config()
        for name in ['FNO', 'Hybrid-FNO']:
            model_type = name.lower().replace('-', '_')
            model = create_model(model_type, model_config)
            predictors[name] = CoolingModelPredictor(
                model=model,
                model_type=model_type
            )
        logger.info(f"   Created {len(predictors)} dummy models")
    
    # 3. Prepare test data
    logger.info("\n3. Preparing test data...")
    n_samples = 1000
    n_features = config.num_cdus + 1
    n_outputs = config.num_cdus + 3
    
    # Create dummy test data
    test_inputs = np.random.randn(n_samples, n_features)
    test_targets = np.random.randn(n_samples, n_outputs)
    
    test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    
    logger.info(f"   Test samples: {n_samples}")
    logger.info(f"   Input features: {n_features}")
    logger.info(f"   Output features: {n_outputs}")
    
    # 4. Run predictions
    logger.info("\n4. Running predictions for all models...")
    all_predictions = {}
    
    for name, predictor in predictors.items():
        logger.info(f"   Predicting with {name}...")
        predictions = predictor.predict(test_inputs_tensor)
        all_predictions[name] = predictions
        logger.info(f"     Shape: {predictions.shape}")
    
    # 5. Calculate metrics
    logger.info("\n5. Calculating metrics...")
    metrics_calc = MetricsCalculator()
    
    comparison_results = {}
    for name, predictions in all_predictions.items():
        metrics = metrics_calc.calculate_all_metrics(predictions, test_targets)
        comparison_results[name] = metrics
        
        logger.info(f"\n   {name}:")
        logger.info(f"     MSE:  {metrics['mse']:.6f}")
        logger.info(f"     MAE:  {metrics['mae']:.6f}")
        logger.info(f"     RMSE: {metrics['rmse']:.6f}")
        logger.info(f"     R²:   {metrics['r2']:.6f}")
    
    # 6. Physics validation
    logger.info("\n6. Physics validation...")
    physics_validator = PhysicsValidator()
    
    physics_results = {}
    for name, predictions in all_predictions.items():
        physics = physics_validator.validate_predictions(
            predictions=predictions,
            targets=test_targets,
            inputs=test_inputs
        )
        physics_results[name] = physics
        
        logger.info(f"\n   {name}:")
        for key, value in physics.items():
            if isinstance(value, (int, float)):
                logger.info(f"     {key}: {value:.6f}")
    
    # 7. Create comparison plots
    logger.info("\n7. Creating comparison plots...")
    
    # Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['mse', 'mae', 'rmse', 'r2']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        model_names = list(comparison_results.keys())
        values = [comparison_results[name][metric] for name in model_names]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(model_names)]
        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plot_path = output_dir / 'metrics_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"   Metrics plot saved to: {plot_path}")
    plt.close()
    
    # 8. Save comparison results
    logger.info("\n8. Saving comparison results...")
    
    # Metrics table
    metrics_df = pd.DataFrame(comparison_results).T
    metrics_path = output_dir / 'metrics_comparison.csv'
    metrics_df.to_csv(metrics_path)
    logger.info(f"   Metrics saved to: {metrics_path}")
    
    # Physics validation table
    physics_df = pd.DataFrame(physics_results).T
    physics_path = output_dir / 'physics_comparison.csv'
    physics_df.to_csv(physics_path)
    logger.info(f"   Physics results saved to: {physics_path}")
    
    # 9. Rank models
    logger.info("\n9. Model ranking...")
    
    # Rank by MSE (lower is better)
    ranked = sorted(comparison_results.items(), key=lambda x: x[1]['mse'])
    
    logger.info("\n   Rankings (by MSE):")
    for rank, (name, metrics) in enumerate(ranked, 1):
        logger.info(f"     {rank}. {name:15s} - MSE: {metrics['mse']:.6f}, "
                   f"R²: {metrics['r2']:.6f}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison Complete!")
    logger.info("=" * 80)
    logger.info(f"\nCompared {len(predictors)} models on {n_samples} test samples")
    logger.info(f"Best model (by MSE): {ranked[0][0]}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()