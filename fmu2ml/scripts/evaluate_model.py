"""
Model Evaluation CLI

Evaluates trained models on test data with comprehensive metrics.

Usage:
    python -m fmu2ml.scripts.evaluate_model --checkpoint checkpoints/best.pt --data data/ --output results/
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from fmu2ml.config import get_system_config
from fmu2ml.models import create_model
from fmu2ml.inference import CoolingModelInference
from fmu2ml.evaluation import MetricsCalculator, PhysicsValidator
from fmu2ml.data.processors import create_data_loaders
from fmu2ml.utils.logging_utils import setup_logger
from fmu2ml.utils.io_utils import save_results


def evaluate_model(args):
    """Evaluate model on test data"""
    
    logger = setup_logger('evaluate')
    
    logger.info("=" * 80)
    logger.info("FMU2ML Model Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    predictor = CoolingModelInference(
        model_path=args.checkpoint,
        model_type=args.model
    )
    
    # Create data loader
    logger.info("Loading test data...")
    _, _, test_loader = create_data_loaders(
        data_path=args.data,
        test_chunks=args.test_chunks,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=False
    )
    
    # Initialize evaluators
    metrics_calc = MetricsCalculator()
    physics_validator = PhysicsValidator()
    
    # Evaluation loop
    logger.info("Evaluating model...")
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.model.to(device)
    predictor.model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Predict
            predictions = predictor.model(inputs)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
            
            # Calculate batch metrics
            batch_metrics = metrics_calc.calculate_batch_metrics(
                predictions.cpu().numpy(),
                targets.cpu().numpy()
            )
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: MSE={batch_metrics['mse']:.6f}, MAE={batch_metrics['mae']:.6f}")
    
    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)
    
    logger.info(f"Total samples evaluated: {len(predictions)}")
    
    # Calculate overall metrics
    logger.info("Calculating metrics...")
    metrics = metrics_calc.calculate_all_metrics(predictions, targets)
    
    logger.info("Overall Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    # Physics validation
    logger.info("Validating physics constraints...")
    physics_results = physics_validator.validate_predictions(
        predictions=predictions,
        targets=targets,
        inputs=inputs
    )
    
    logger.info("Physics Validation:")
    for key, value in physics_results.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v:.6f}")
        else:
            logger.info(f"  {key}: {value:.6f}")
    
    # Save results
    logger.info("Saving results...")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Save physics validation
    physics_df = pd.DataFrame([physics_results])
    physics_df.to_csv(output_dir / 'physics_validation.csv', index=False)
    
    # Save predictions (sample)
    if args.save_predictions:
        sample_size = min(10000, len(predictions))
        pred_df = pd.DataFrame({
            f'pred_{i}': predictions[:sample_size, i] for i in range(predictions.shape[1])
        })
        pred_df.to_parquet(output_dir / 'predictions_sample.parquet')
    
    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return metrics, physics_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m fmu2ml.scripts.evaluate_model --checkpoint checkpoints/best.pt --data data/ --output results/

  # Custom test chunks
  python -m fmu2ml.scripts.evaluate_model --checkpoint checkpoints/best.pt \\
      --data data/ --test-chunks 10 11 12 --output results/
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory containing parquet files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='fno',
                        choices=['fno', 'hybrid_fno', 'deeponet'],
                        help='Model architecture (default: fno)')
    parser.add_argument('--test-chunks', type=int, nargs='+', default=[10],
                        help='Test chunk IDs (default: [10])')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save sample predictions')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(args)


if __name__ == '__main__':
    main()