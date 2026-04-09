import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing as mp
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _create_single_response_surface(args):
    """Create response surface plot for a single output/input combination. Used for parallel execution."""
    output_name, input1, input2, X, Y, Z, output_dir = args
    
    try:
        fig = plt.figure(figsize=(16, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(
            X, Y, Z,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        ax1.set_xlabel(input1, fontsize=11)
        ax1.set_ylabel(input2, fontsize=11)
        ax1.set_zlabel(output_name, fontsize=11)
        ax1.set_title(f'3D Surface: {output_name}', fontsize=12)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
        ax2.set_xlabel(input1, fontsize=11)
        ax2.set_ylabel(input2, fontsize=11)
        ax2.set_title(f'Contour: {output_name}', fontsize=12)
        fig.colorbar(contour, ax=ax2)
        
        plt.suptitle(
            f'Response Surface: {output_name} vs {input1} × {input2}',
            fontsize=14
        )
        plt.tight_layout()
        
        filename = f'response_surface_{output_name}_{input1}_{input2}.png'
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        return (output_name, input1, input2, True)
    except Exception as e:
        logger.error(f"Failed to create surface plot for {output_name}, {input1}, {input2}: {e}")
        return (output_name, input1, input2, False)


def create_response_surface_plots(
    surfaces: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str,
    n_workers: int = 8
):
    """Create 3D surface and contour plots for key relationships using parallel processing."""
    logger.info("Creating response surface plots in parallel...")
    
    # Prepare arguments for parallel plot creation
    args_list = []
    for (output_name, input1, input2), (X, Y, Z) in surfaces.items():
        args_list.append((output_name, input1, input2, X, Y, Z, output_dir))
    
    # Use multiprocessing pool for plot creation
    with mp.Pool(processes=min(n_workers, len(args_list))) as pool:
        results = pool.map(_create_single_response_surface, args_list)
    
    successful = sum(1 for r in results if r[3])
    logger.info(f"Created {successful}/{len(results)} response surface plots")