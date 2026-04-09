"""
Parallel processing utilities for multivariate effect analysis.
"""

import logging
from typing import Optional
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def setup_dask_cluster(
    n_workers: int = 8,
    threads_per_worker: int = 1,
    memory_limit: str = '5GB',
    dashboard_address: str = ':8788'
) -> Client:
    """
    Setup Dask distributed cluster.
    
    Args:
        n_workers: Number of workers
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        dashboard_address: Dashboard address
        
    Returns:
        Dask client
    """
    logger.info(f"Setting up Dask cluster with {n_workers} workers...")
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        silence_logs=logging.WARNING
    )
    
    client = Client(cluster)
    logger.info(f"Dask cluster ready: {client.dashboard_link}")
    
    return client


def close_dask_cluster(client: Optional[Client]):
    """
    Close Dask distributed cluster.
    
    Args:
        client: Dask client to close
    """
    if client is not None:
        logger.info("Closing Dask cluster...")
        client.close()
        logger.info("Dask cluster closed")
