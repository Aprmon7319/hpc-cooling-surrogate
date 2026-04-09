#!/usr/bin/env python3
"""
Model Training CLI

Trains neural operator models for datacenter cooling prediction.

Usage:
    # Single GPU training
    python -m fmu2ml.scripts.train_model --system marconi100 --model fno --data data/
    
    # Multi-GPU DDP training
    python -m fmu2ml.scripts.train_model --system summit --model hybrid_fno --gpus 4 --data data/
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fmu2ml.config import get_system_config, get_training_config
from fmu2ml.models import create_model
from fmu2ml.training import Trainer
from fmu2ml.data.processors import create_data_loaders
from fmu2ml.utils.logging_utils import setup_logging
from fmu2ml.utils.distributed import setup_distributed, cleanup_distributed


def train_worker(rank, world_size, args):
    """Training worker for distributed training"""
    
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Setup logging
    logger = setup_logging(f'train_rank_{rank}')
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("FMU2ML Model Training")
        logger.info("=" * 80)
        logger.info(f"System: {args.system}")
        logger.info(f"Model: {args.model}")
        logger.info(f"GPUs: {world_size}")
        logger.info(f"Data: {args.data}")
    
    # Load configurations
    system_config = get_system_config(args.system, use_raps=args.use_raps)
    training_config = get_training_config(args.model)
    
    # Override with CLI arguments
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.lr:
        training_config.learning_rate = args.lr
    if args.epochs:
        training_config.epochs = args.epochs
    
    # Create model
    model_config = system_config.to_model_config()
    model = create_model(args.model, model_config)
    
    # Move to GPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=args.data,
        train_chunks=args.train_chunks,
        val_chunks=args.val_chunks,
        test_chunks=args.test_chunks,
        batch_size=training_config.batch_size,
        num_workers=args.workers,
        distributed=(world_size > 1)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        rank=rank,
        world_size=world_size
    )
    
    # Train
    if rank == 0:
        logger.info("Starting training...")
    
    trainer.train()
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info(f"Model saved to: {trainer.checkpoint_dir}")
        logger.info("=" * 80)
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description='Train neural operator models for datacenter cooling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU training
  python -m fmu2ml.scripts.train_model --system marconi100 --model fno --data data/

  # Multi-GPU DDP training
  python -m fmu2ml.scripts.train_model --system summit --model hybrid_fno --gpus 4 --data data/
  
  # Custom hyperparameters
  python -m fmu2ml.scripts.train_model --system frontier --model deeponet \\
      --data data/ --batch-size 64 --lr 0.001 --epochs 100
        """
    )
    
    # Required arguments
    parser.add_argument('--system', type=str, required=True,
                        choices=['marconi100', 'summit', 'frontier', 'fugaku', 'lassen'],
                        help='HPC system name')
    parser.add_argument('--model', type=str, required=True,
                        choices=['fno', 'hybrid_fno', 'deeponet'],
                        help='Model architecture')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory containing parquet files')
    
    # Data arguments
    parser.add_argument('--train-chunks', type=int, nargs='+', default=None,
                        help='Training chunk IDs (default: 0-7)')
    parser.add_argument('--val-chunks', type=int, nargs='+', default=None,
                        help='Validation chunk IDs (default: 8-9)')
    parser.add_argument('--test-chunks', type=int, nargs='+', default=None,
                        help='Test chunk IDs (default: 10)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (default: 1)')
    parser.add_argument('--use-raps', action='store_true', default=True,
                        help='Use RAPS configuration (default: True)')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='fmu2ml',
                        help='W&B project name (default: fmu2ml)')
    
    args = parser.parse_args()
    
    # Set default chunk splits if not provided
    if args.train_chunks is None:
        args.train_chunks = list(range(8))
    if args.val_chunks is None:
        args.val_chunks = [8, 9]
    if args.test_chunks is None:
        args.test_chunks = [10]
    
    # Launch training
    world_size = args.gpus
    
    if world_size == 1:
        # Single GPU
        train_worker(0, 1, args)
    else:
        # Multi-GPU with torch.multiprocessing
        import torch.multiprocessing as mp
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()