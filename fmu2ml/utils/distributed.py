"""
Distributed training utilities.
"""
import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(backend: str = 'nccl', init_method: str = 'env://') -> int:
    """
    Setup distributed training
    
    Parameters:
    -----------
    backend : str
        Distributed backend ('nccl', 'gloo')
    init_method : str
        Initialization method
    
    Returns:
    --------
    int : Local rank
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        
        torch.cuda.set_device(local_rank)
        
        print(f"✓ Initialized distributed training: rank={rank}, world_size={world_size}")
        return local_rank
    else:
        print("✓ Running in single-process mode")
        return 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("✓ Cleaned up distributed training")
