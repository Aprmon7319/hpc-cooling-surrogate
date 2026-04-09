#!/usr/bin/env python3
"""
Complete Training Example

Demonstrates the full training workflow:
- Data loading and preprocessing
- Model creation and configuration
- Training with physics losses
- Validation and checkpointing
- Model evaluation
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from fmu2ml import (
    get_system_config,
    create_model,
    create_data_loaders,
    Trainer,
    MetricsCalculator,
    PhysicsValidator,
    setup_logger
)


def main():
    # Setup
    logger = setup_logger('train_example', level='INFO')
    logger.info("=" * 80)
    logger.info("FMU2ML Training Example")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)
    
    # Configuration
    system_name = 'marconi100'
    model_type = 'fno'
    data_path = 'data/'
    output_dir = Path('outputs/train_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    
    # 1. Load system configuration
    logger.info("\n1. Loading configuration...")
    config = get_system_config(system_name, use_raps=True)
    logger.info(f"   System: {config.system_name}")
    logger.info(f"   CDUs: {config.num_cdus}")
    
    # 2. Create data loaders
    logger.info("\n2. Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=data_path,
            train_chunks=[0, 1, 2],
            val_chunks=[3],
            test_chunks=[4],
            batch_size=32,
            num_workers=2,
            distributed=False
        )
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"   Data loading failed: {e}")
        logger.info("   Using dummy data for demonstration...")
        train_loader = None
        val_loader = None
        test_loader = None
    
    # 3. Create model
    logger.info("\n3. Creating model...")
    model_config = config.to_model_config()
    model = create_model(model_type, model_config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model: {model.__class__.__name__}")
    logger.info(f"   Parameters: {n_params:,}")
    
    # 4. Setup training configuration
    logger.info("\n4. Setting up training...")
    from fmu2ml.config import TrainingConfig
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=10,  # Small for demo
        optimizer='adamw',
        weight_decay=0.0001,
        scheduler='cosine',
        use_physics_loss=True,
        physics_loss_weight=0.1,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        log_interval=5
    )
    
    logger.info(f"   Epochs: {training_config.epochs}")
    logger.info(f"   Learning rate: {training_config.learning_rate}")
    logger.info(f"   Physics loss: {training_config.use_physics_loss}")
    
    # 5. Create trainer
    logger.info("\n5. Creating trainer...")
    if train_loader is not None:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        # 6. Train model
        logger.info("\n6. Training model...")
        logger.info("-" * 80)
        
        try:
            trainer.train()
            logger.info("\n   Training completed successfully!")
            
            # Get final metrics
            final_metrics = trainer.get_final_metrics()
            logger.info(f"\n   Final training loss: {final_metrics.get('train_loss', 'N/A'):.6f}")
            logger.info(f"   Final validation loss: {final_metrics.get('val_loss', 'N/A'):.6f}")
            
        except Exception as e:
            logger.error(f"\n   Training failed: {e}")
    else:
        logger.warning("   Skipping training (no data available)")
    
    # 7. Evaluate model
    logger.info("\n7. Evaluating model...")
    if test_loader is not None:
        metrics_calc = MetricsCalculator()
        physics_validator = PhysicsValidator()
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                predictions = model(inputs)
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = metrics_calc.calculate_all_metrics(predictions, targets)
        
        logger.info("\n   Test Metrics:")
        for key, value in metrics.items():
            logger.info(f"     {key}: {value:.6f}")
        
        # Physics validation
        physics_results = physics_validator.validate_predictions(
            predictions=predictions,
            targets=targets,
            inputs=np.concatenate([x.cpu().numpy() for x, _ in test_loader], axis=0)
        )
        
        logger.info("\n   Physics Validation:")
        for key, value in physics_results.items():
            if isinstance(value, dict):
                logger.info(f"     {key}:")
                for k, v in value.items():
                    logger.info(f"       {k}: {v:.6f}")
            else:
                logger.info(f"     {key}: {value:.6f}")
    else:
        logger.warning("   Skipping evaluation (no test data)")
    
    # 8. Save final model
    logger.info("\n8. Saving model...")
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_config': model_config,
        'training_config': training_config
    }, final_model_path)
    logger.info(f"   Model saved to: {final_model_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Example Complete!")
    logger.info(f"Finished: {datetime.now()}")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  - Checkpoints: {output_dir / 'checkpoints'}")
    logger.info(f"  - Final model: {final_model_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()