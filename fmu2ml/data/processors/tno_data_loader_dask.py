"""
Optimized CDU Sequence Dataset for TNO-DeepM&Mnet Training
==========================================================

Features:
1. Column pruning at read time
2. Lazy loading with Dask integration
3. Memory-mapped NumPy storage
4. Hierarchical lazy dataset architecture
5. Split-aware data management
7. Parallel I/O strategy

"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Literal
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import hashlib
import json
import warnings
import logging
from abc import ABC, abstractmethod
import os
import threading

from .normalization import NormalizationHandler
from raps.config import ConfigManager


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TNOSequenceConfig:
    """Configuration for TNO-DeepM&Mnet sequence dataset."""
    history_length: int = 30
    prediction_horizon: int = 10
    stride: int = 1
    pool_cdus: bool = True
    include_cdu_id: bool = True


@dataclass
class LazyDataConfig:
    """
    Configuration for lazy data loading.
    
    Attributes:
        cdu_lim: Maximum number of CDUs to load (None = all)
        chunk_indices: Specific chunks to load (None = auto-discover)
        cache_dir: Directory for memory-mapped cache (None = temp dir)
        backend: Data loading backend ('dask', 'pandas', 'auto')
        num_workers: Number of parallel workers for I/O
        use_mmap: Whether to use memory-mapped storage
        compression: Compression for intermediate storage (None, 'gzip')
        prefetch_chunks: Number of chunks to prefetch
    """
    cdu_lim: Optional[int] = None
    chunk_indices: Optional[List[int]] = None
    cache_dir: Optional[str] = None
    backend: Literal['dask', 'pandas', 'auto'] = 'auto'
    num_workers: int = 4
    use_mmap: bool = True
    compression: Optional[str] = None
    prefetch_chunks: int = 2


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_by: Literal['temporal', 'chunks', 'auto'] = 'auto'
    train_chunks: Optional[List[int]] = None
    val_chunks: Optional[List[int]] = None
    test_chunks: Optional[List[int]] = None


# =============================================================================
# Column Builder - Handles column name generation and filtering
# =============================================================================

class ColumnBuilder:
    """
    Builds column lists based on CDU limits and data requirements.
    
    This class handles the mapping between CDU IDs and their corresponding
    column names in the parquet files.
    """
    
    # Output variable names for each CDU
    CDU_OUTPUT_VARS = [
        'm_flow_prim', 'V_flow_prim_GPM', 'm_flow_sec', 'V_flow_sec_GPM',
        'W_flow_CDUP', 'W_flow_CDUP_kW',
        'T_prim_s', 'T_prim_s_C', 'T_prim_r', 'T_prim_r_C',
        'T_sec_s', 'T_sec_s_C', 'T_sec_r', 'T_sec_r_C',
        'p_prim_s', 'p_prim_s_psig', 'p_prim_r', 'p_prim_r_psig',
        'p_sec_s', 'p_sec_s_psig', 'p_sec_r', 'p_sec_r_psig'
    ]
    
    # Subset used for training (matching original code)
    CDU_OUTPUT_VARS_TRAINING = [
        'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
        'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
        'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
    ]
    
    def __init__(self, num_cdus_total: int, cdu_lim: Optional[int] = None):
        """
        Initialize column builder.
        
        Args:
            num_cdus_total: Total number of CDUs in the system
            cdu_lim: Maximum CDUs to include (None = all)
        """
        self.num_cdus_total = num_cdus_total
        self.num_cdus = min(num_cdus_total, cdu_lim) if cdu_lim else num_cdus_total
        self._build_column_mappings()
    
    def _build_column_mappings(self):
        """Build all column name mappings."""
        self.global_input_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        self.time_col = 'time'
        self.pue_col = 'pue'
        
        # Datacenter-level summary columns
        self.dc_summary_cols = [
            'simulator[1].datacenter[1].summary.m_flow_prim',
            'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        ]
        
        # Per-CDU column mappings
        self.input_cols_per_cdu: Dict[int, List[str]] = {}
        self.output_cols_per_cdu: Dict[int, List[str]] = {}
        self.output_cols_training_per_cdu: Dict[int, List[str]] = {}
        
        for cdu_id in range(1, self.num_cdus + 1):
            # Input columns
            self.input_cols_per_cdu[cdu_id] = [
                f'simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_Q_flow_total',
                f'simulator_1_datacenter_1_computeBlock_{cdu_id}_cabinet_1_sources_T_Air'
            ]
            
            # All output columns
            self.output_cols_per_cdu[cdu_id] = [
                f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.{var}'
                for var in self.CDU_OUTPUT_VARS
            ]
            # Add HTC
            self.output_cols_per_cdu[cdu_id].append(
                f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cabinet[1].summary.htc'
            )
            
            # Training subset of output columns
            self.output_cols_training_per_cdu[cdu_id] = [
                f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.{var}'
                for var in self.CDU_OUTPUT_VARS_TRAINING
            ]
            self.output_cols_training_per_cdu[cdu_id].append(
                f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cabinet[1].summary.htc'
            )
    
    def get_required_columns(self, include_all_outputs: bool = False) -> List[str]:
        """
        Get list of all required columns for loading.
        
        Args:
            include_all_outputs: If True, include all output vars; else training subset
            
        Returns:
            List of column names to load from parquet
        """
        columns = [self.global_input_col, self.time_col]
        
        # Add PUE if available
        columns.append(self.pue_col)
        
        # Add datacenter summary columns
        columns.extend(self.dc_summary_cols)
        
        # Add per-CDU columns
        for cdu_id in range(1, self.num_cdus + 1):
            columns.extend(self.input_cols_per_cdu[cdu_id])
            if include_all_outputs:
                columns.extend(self.output_cols_per_cdu[cdu_id])
            else:
                columns.extend(self.output_cols_training_per_cdu[cdu_id])
        
        return columns
    
    def get_input_columns_flat(self) -> List[str]:
        """Get flat list of all input columns."""
        cols = [self.global_input_col]
        for cdu_id in range(1, self.num_cdus + 1):
            cols.extend(self.input_cols_per_cdu[cdu_id])
        return cols
    
    def get_output_columns_flat(self, training_subset: bool = True) -> List[str]:
        """Get flat list of all output columns."""
        cols = []
        output_mapping = (self.output_cols_training_per_cdu if training_subset 
                         else self.output_cols_per_cdu)
        for cdu_id in range(1, self.num_cdus + 1):
            cols.extend(output_mapping[cdu_id])
        return cols
    
    @property
    def num_input_features(self) -> int:
        """Number of input features per CDU: [Q_flow, T_Air, T_ext]"""
        return 3
    
    @property
    def num_output_features(self) -> int:
        """Number of output features per CDU (training subset + HTC)"""
        return len(self.CDU_OUTPUT_VARS_TRAINING) + 1  # +1 for HTC


# =============================================================================
# Memory-Mapped Storage Backend
# =============================================================================

class MMapStorage:
    """
    Memory-mapped storage for preprocessed CDU data.
    
    Stores data in numpy .npy files with memory-mapping for efficient
    random access without loading entire arrays into RAM.
    """
    
    def __init__(self, cache_dir: Path, num_cdus: int, num_timesteps: int,
                 num_input_features: int, num_output_features: int):
        """
        Initialize memory-mapped storage.
        
        Args:
            cache_dir: Directory to store .npy files
            num_cdus: Number of CDUs
            num_timesteps: Total number of timesteps
            num_input_features: Features per CDU input
            num_output_features: Features per CDU output
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_cdus = num_cdus
        self.num_timesteps = num_timesteps
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        
        # File paths
        self.inputs_path = self.cache_dir / 'inputs.npy'
        self.outputs_path = self.cache_dir / 'outputs.npy'
        self.inputs_raw_path = self.cache_dir / 'inputs_raw.npy'
        self.outputs_raw_path = self.cache_dir / 'outputs_raw.npy'
        self.metadata_path = self.cache_dir / 'metadata.json'
        
        # Memory-mapped arrays (initialized lazily)
        self._inputs: Optional[np.memmap] = None
        self._outputs: Optional[np.memmap] = None
        self._inputs_raw: Optional[np.memmap] = None
        self._outputs_raw: Optional[np.memmap] = None
        
        self._lock = threading.Lock()
    
    def create(self, inputs: np.ndarray, outputs: np.ndarray,
               inputs_raw: np.ndarray, outputs_raw: np.ndarray,
               metadata: Dict) -> None:
        """
        Create memory-mapped files from arrays.
        
        Args:
            inputs: Normalized inputs (num_cdus, T, num_input_features)
            outputs: Normalized outputs (num_cdus, T, num_output_features)
            inputs_raw: Raw inputs (num_cdus, T, num_input_features)
            outputs_raw: Raw outputs (num_cdus, T, num_output_features)
            metadata: Additional metadata to store
        """
        # Save arrays
        np.save(self.inputs_path, inputs)
        np.save(self.outputs_path, outputs)
        np.save(self.inputs_raw_path, inputs_raw)
        np.save(self.outputs_raw_path, outputs_raw)
        
        # Save metadata
        metadata.update({
            'num_cdus': self.num_cdus,
            'num_timesteps': self.num_timesteps,
            'num_input_features': self.num_input_features,
            'num_output_features': self.num_output_features,
            'inputs_shape': list(inputs.shape),
            'outputs_shape': list(outputs.shape)
        })
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def exists(self) -> bool:
        """Check if cache files exist."""
        return all(p.exists() for p in [
            self.inputs_path, self.outputs_path,
            self.inputs_raw_path, self.outputs_raw_path,
            self.metadata_path
        ])
    
    def load_metadata(self) -> Dict:
        """Load metadata from cache."""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    @property
    def inputs(self) -> np.memmap:
        """Get memory-mapped inputs array."""
        if self._inputs is None:
            with self._lock:
                if self._inputs is None:
                    self._inputs = np.load(self.inputs_path, mmap_mode='r')
        return self._inputs
    
    @property
    def outputs(self) -> np.memmap:
        """Get memory-mapped outputs array."""
        if self._outputs is None:
            with self._lock:
                if self._outputs is None:
                    self._outputs = np.load(self.outputs_path, mmap_mode='r')
        return self._outputs
    
    @property
    def inputs_raw(self) -> np.memmap:
        """Get memory-mapped raw inputs array."""
        if self._inputs_raw is None:
            with self._lock:
                if self._inputs_raw is None:
                    self._inputs_raw = np.load(self.inputs_raw_path, mmap_mode='r')
        return self._inputs_raw
    
    @property
    def outputs_raw(self) -> np.memmap:
        """Get memory-mapped raw outputs array."""
        if self._outputs_raw is None:
            with self._lock:
                if self._outputs_raw is None:
                    self._outputs_raw = np.load(self.outputs_raw_path, mmap_mode='r')
        return self._outputs_raw
    
    def get_cdu_data(self, cdu_idx: int, start: int, end: int, 
                     normalized: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific CDU and time range.
        
        Args:
            cdu_idx: CDU index (0-based)
            start: Start timestep
            end: End timestep
            normalized: Return normalized or raw data
            
        Returns:
            Tuple of (inputs, outputs) arrays
        """
        if normalized:
            return (
                np.array(self.inputs[cdu_idx, start:end]),
                np.array(self.outputs[cdu_idx, start:end])
            )
        else:
            return (
                np.array(self.inputs_raw[cdu_idx, start:end]),
                np.array(self.outputs_raw[cdu_idx, start:end])
            )
    
    def cleanup(self):
        """Clean up memory-mapped files."""
        self._inputs = None
        self._outputs = None
        self._inputs_raw = None
        self._outputs_raw = None
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


# =============================================================================
# Dask Data Backend - Handles lazy loading
# =============================================================================

class DaskDataBackend:
    """
    Dask-based data backend for lazy loading and parallel I/O.
    
    Handles:
    - Lazy loading of parquet files
    - Column pruning at read time
    - Parallel chunk loading
    - Computation of normalization statistics
    """
    
    def __init__(self, data_path: Path, column_builder: ColumnBuilder,
                 chunk_indices: Optional[List[int]] = None,
                 num_workers: int = 4,
                 verbose: bool = True):
        """
        Initialize Dask backend.
        
        Args:
            data_path: Path to data directory
            column_builder: ColumnBuilder instance for column selection
            chunk_indices: Specific chunks to load (None = auto-discover)
            num_workers: Number of parallel workers
            verbose: Print progress information
        """
        self.data_path = Path(data_path)
        self.column_builder = column_builder
        self.num_workers = num_workers
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        

        # Initialize _chunk_file_map BEFORE calling _discover_chunks
        self._chunk_file_map: Dict[int, List[Path]] = {}
        
        # Lazy Dask DataFrame (initialized on first access)
        self._ddf: Optional[dd.DataFrame] = None
        
        
        # Discover available chunks
        self.available_chunks = self._discover_chunks()
        
        # Filter to requested chunks
        if chunk_indices is not None:
            self.chunk_indices = [c for c in chunk_indices if c in self.available_chunks]
        else:
            self.chunk_indices = self.available_chunks
        
        if not self.chunk_indices:
            raise ValueError(f"No valid chunks found. Available: {self.available_chunks}")
        
        # Required columns for loading
        self.required_columns = column_builder.get_required_columns()
        
        
        if self.verbose:
            print(f"DaskDataBackend initialized:")
            print(f"  - Available chunks: {len(self.available_chunks)}")
            print(f"  - Selected chunks: {len(self.chunk_indices)}")
            print(f"  - Required columns: {len(self.required_columns)}")
    
    def _discover_chunks(self) -> List[int]:
        """Discover available chunk indices."""
        chunks = []
        
        if self.verbose:
            print(f"Discovering chunks in: {self.data_path}")
        
        for chunk_dir in sorted(self.data_path.glob("chunk_*")):
            if not chunk_dir.is_dir():
                continue
                
            try:
                # Extract chunk number - handles "chunk_0", "chunk_1", etc.
                chunk_name = chunk_dir.name
                parts = chunk_name.split("_")
                if len(parts) >= 2:
                    chunk_idx = int(parts[1])
                else:
                    continue
                
                # Find parquet files with multiple patterns
                parquet_files = []
                
                # Pattern 1: Direct parquet files
                parquet_files.extend(list(chunk_dir.glob("*.parquet")))
                
                # Pattern 2: Output parquet files
                parquet_files.extend(list(chunk_dir.glob("*_output_*.parquet")))
                
                # Pattern 3: Parquet files in subdirectories (e.g., chunk_0/data/*.parquet)
                for subdir in chunk_dir.iterdir():
                    if subdir.is_dir():
                        parquet_files.extend(list(subdir.glob("*.parquet")))
                
                # Remove duplicates
                parquet_files = list(set(parquet_files))
                
                if parquet_files:
                    chunks.append(chunk_idx)
                    self._chunk_file_map[chunk_idx] = sorted(parquet_files)
                    if self.verbose:
                        print(f"  Found chunk_{chunk_idx}: {len(parquet_files)} parquet file(s)")
                        # Show first file for debugging
                        print(f"    First file: {parquet_files[0].name}")
                else:
                    if self.verbose:
                        print(f"  Skipping chunk_{chunk_idx}: no parquet files found")
                        # List what IS in the directory for debugging
                        contents = list(chunk_dir.iterdir())[:5]
                        if contents:
                            print(f"    Contains: {[c.name for c in contents]}")
                            
            except (ValueError, IndexError) as e:
                if self.verbose:
                    print(f"  Skipping {chunk_dir.name}: {e}")
                continue
        
        if not chunks:
            print(f"WARNING: No valid chunks found in {self.data_path}")
            print(f"Directory contents:")
            for item in sorted(self.data_path.iterdir())[:10]:
                item_type = 'DIR' if item.is_dir() else 'FILE'
                print(f"  - {item.name} [{item_type}]")
        
        return sorted(chunks)
    
    def _get_available_columns(self, parquet_file: Path) -> List[str]:
        """Get available columns from a parquet file without loading data."""
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet_file)
        return schema.names
    
    def _filter_columns_to_available(self, available: List[str]) -> List[str]:
        """Filter required columns to those actually available."""
        available_set = set(available)
        return [c for c in self.required_columns if c in available_set]
    
    def get_lazy_dataframe(self) -> dd.DataFrame:
        """
        Get lazy Dask DataFrame with column pruning.
        
        Returns:
            Dask DataFrame with only required columns loaded
        """
        if self._ddf is not None:
            return self._ddf
        
        # Get available columns from first file
        first_chunk = self.chunk_indices[0]
        first_file = self._chunk_file_map[first_chunk][0]
        available_cols = self._get_available_columns(first_file)
        columns_to_load = self._filter_columns_to_available(available_cols)
        
        if self.verbose:
            print(f"Loading {len(columns_to_load)}/{len(self.required_columns)} columns")
            missing = set(self.required_columns) - set(columns_to_load)
            if missing:
                print(f"  Missing columns: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        
        # Build list of parquet files to load
        parquet_files = []
        for chunk_idx in self.chunk_indices:
            parquet_files.extend(self._chunk_file_map.get(chunk_idx, []))
        
        if not parquet_files:
            raise ValueError("No parquet files found for selected chunks")
        
        # Load with Dask - column pruning happens at read time
        self._ddf = dd.read_parquet(
            parquet_files,
            columns=columns_to_load,
            engine='pyarrow'
        )
        
        return self._ddf
    
    def compute_normalization_stats(self, input_cols: List[str], 
                                    output_cols: List[str]) -> Tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray, np.ndarray]:
        """
        Compute normalization statistics using Dask aggregations.
        
        Args:
            input_cols: List of input column names
            output_cols: List of output column names
            
        Returns:
            Tuple of (mean_in, std_in, mean_out, std_out)
        """
        ddf = self.get_lazy_dataframe()
        
        # Filter to available columns
        available = set(ddf.columns)
        input_cols = [c for c in input_cols if c in available]
        output_cols = [c for c in output_cols if c in available]
        
        if self.verbose:
            print(f"Computing normalization stats for {len(input_cols)} inputs, {len(output_cols)} outputs...")
        
        # Compute mean and std using Dask (lazy until .compute())
        all_cols = input_cols + output_cols
        
        # Use Dask aggregations - computed in parallel
        means = ddf[all_cols].mean().compute()
        stds = ddf[all_cols].std().compute()
        
        n_in = len(input_cols)
        mean_in = means.iloc[:n_in].values.astype(np.float32)
        std_in = stds.iloc[:n_in].values.astype(np.float32)
        mean_out = means.iloc[n_in:].values.astype(np.float32)
        std_out = stds.iloc[n_in:].values.astype(np.float32)
        
        # Prevent division by zero
        std_in = np.where(np.abs(std_in) < 1e-8, 1.0, std_in)
        std_out = np.where(np.abs(std_out) < 1e-8, 1.0, std_out)
        
        return mean_in, std_in, mean_out, std_out
    
    def load_chunk_data(self, chunk_idx: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load a specific chunk's data.
        
        Args:
            chunk_idx: Chunk index to load
            columns: Specific columns to load (None = all required)
            
        Returns:
            Pandas DataFrame with chunk data
        """
        if chunk_idx not in self._chunk_file_map:
            raise ValueError(f"Chunk {chunk_idx} not available")
        
        files = self._chunk_file_map[chunk_idx]
        cols = columns or self.required_columns
        
        # Filter to available columns
        available = self._get_available_columns(files[0])
        cols = [c for c in cols if c in available]
        
        dfs = []
        for f in files:
            df = pd.read_parquet(f, columns=cols)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_all_data_parallel(self) -> pd.DataFrame:
        """
        Load all chunks in parallel using thread pool.
        
        Returns:
            Concatenated pandas DataFrame
        """
        if self.verbose:
            print(f"Loading {len(self.chunk_indices)} chunks in parallel...")
        
        dfs = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.load_chunk_data, idx): idx 
                for idx in self.chunk_indices
            }
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    df = future.result()
                    df['_chunk_idx'] = chunk_idx
                    dfs.append(df)
                    if self.verbose:
                        print(f"  Loaded chunk {chunk_idx}: {len(df)} rows")
                except Exception as e:
                    self.logger.error(f"Error loading chunk {chunk_idx}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded from any chunk")
        
        # Sort by chunk index to maintain order
        dfs.sort(key=lambda x: x['_chunk_idx'].iloc[0])
        
        return pd.concat(dfs, ignore_index=True)
    
    def materialize(self) -> pd.DataFrame:
        """
        Materialize the lazy Dask DataFrame to pandas.
        
        For large datasets, prefer load_all_data_parallel() for better memory control.
        
        Returns:
            Pandas DataFrame
        """
        return self.get_lazy_dataframe().compute()
    
    def get_num_rows(self) -> int:
        """Get total number of rows (computed lazily)."""
        return len(self.get_lazy_dataframe())


# =============================================================================
# CDU Data Manager - Central data management
# =============================================================================

class CDUDataManager:
    """
    Central data manager that owns data loading, caching, and normalization.
    
    This class:
    - Manages the data lifecycle
    - Handles lazy loading and caching
    - Computes normalization statistics
    - Creates memory-mapped storage
    - Provides views for train/val/test splits
    """
    
    def __init__(
        self,
        data_path: str,
        data_config: Optional[LazyDataConfig] = None,
        tno_config: Optional[TNOSequenceConfig] = None,
        system_name: str = 'summit',
        config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize CDU Data Manager.
        
        Args:
            data_path: Path to data directory
            data_config: Lazy data loading configuration
            tno_config: TNO sequence configuration
            system_name: System name for config lookup
            config: Optional pre-loaded config dict
            verbose: Print progress information
        """
        self.data_path = Path(data_path)
        self.data_config = data_config or LazyDataConfig()
        self.tno_config = tno_config or TNOSequenceConfig()
        self.system_name = system_name
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Get system configuration
        if config is None:
            self.config = ConfigManager(system_name=system_name).get_config()
        else:
            self.config = config
        
        # Determine number of CDUs
        num_cdus_total = self.config.get('NUM_CDUS', 257)
        cdu_lim = self.data_config.cdu_lim
        self.num_cdus = min(num_cdus_total, cdu_lim) if cdu_lim else num_cdus_total
        
        # Initialize column builder
        self.column_builder = ColumnBuilder(num_cdus_total, cdu_lim)
        
        # Initialize Dask backend
        self.dask_backend = DaskDataBackend(
            data_path=self.data_path,
            column_builder=self.column_builder,
            chunk_indices=self.data_config.chunk_indices,
            num_workers=self.data_config.num_workers,
            verbose=verbose
        )
        
        # Setup cache directory
        self._setup_cache_dir()
        
        # State tracking
        self._is_prepared = False
        self._mmap_storage: Optional[MMapStorage] = None
        self._norm_handler: Optional[NormalizationHandler] = None
        self._num_timesteps: int = 0
        
        # Column tracking
        self._available_input_cols: List[str] = []
        self._available_output_cols: List[str] = []
    
    def _setup_cache_dir(self):
        """Setup cache directory for memory-mapped storage."""
        if self.data_config.cache_dir:
            self.cache_dir = Path(self.data_config.cache_dir)
        else:
            # Create temp directory with hash of config for uniqueness
            config_hash = hashlib.md5(
                json.dumps({
                    'data_path': str(self.data_path),
                    'cdu_lim': self.data_config.cdu_lim,
                    'chunks': self.dask_backend.chunk_indices
                }, sort_keys=True).encode()
            ).hexdigest()[:8]
            self.cache_dir = Path(tempfile.gettempdir()) / f'cdu_cache_{config_hash}'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_cache_valid(self) -> bool:
        """Check if cache exists and is valid."""
        if not self.data_config.use_mmap:
            return False
        
        storage = MMapStorage(
            self.cache_dir / 'mmap',
            self.num_cdus,
            0,  # Will be checked from metadata
            self.column_builder.num_input_features,
            self.column_builder.num_output_features
        )
        
        if not storage.exists():
            return False
        
        try:
            metadata = storage.load_metadata()
            # Verify configuration matches
            if (metadata.get('num_cdus') != self.num_cdus or
                metadata.get('chunks') != self.dask_backend.chunk_indices):
                return False
            return True
        except Exception:
            return False
    
    def prepare(self, force_reload: bool = False) -> 'CDUDataManager':
        """
        Prepare data manager by loading data and computing statistics.
        
        This method:
        1. Checks for valid cache
        2. If no cache or force_reload, loads data with Dask
        3. Computes normalization statistics
        4. Extracts and processes CDU data
        5. Creates memory-mapped storage
        
        Args:
            force_reload: Force reload even if cache exists
            
        Returns:
            Self for chaining
        """
        if self._is_prepared and not force_reload:
            return self
        
        # Check for valid cache
        if self._check_cache_valid() and not force_reload:
            if self.verbose:
                print("Loading from cache...")
            self._load_from_cache()
            self._is_prepared = True
            return self
        
        if self.verbose:
            print("Preparing data (no valid cache found)...")
        
        # Load data using parallel I/O
        raw_data = self.dask_backend.load_all_data_parallel()
        self._num_timesteps = len(raw_data)
        
        if self.verbose:
            print(f"Loaded {self._num_timesteps} timesteps")
        
        # Determine available columns
        available_cols = set(raw_data.columns)
        self._available_input_cols = [
            c for c in self.column_builder.get_input_columns_flat() 
            if c in available_cols
        ]
        self._available_output_cols = [
            c for c in self.column_builder.get_output_columns_flat() 
            if c in available_cols
        ]
        
        # Compute normalization statistics
        self._compute_normalization(raw_data)
        
        # Extract and process CDU data
        self._extract_and_cache_cdu_data(raw_data)
        
        self._is_prepared = True
        return self
    
    def _compute_normalization(self, raw_data: pd.DataFrame):
        """Compute normalization statistics from data."""
        if self.verbose:
            print("Computing normalization statistics...")
        
        self._norm_handler = NormalizationHandler()
        self._norm_handler.compute_stats(
            raw_data,
            self._available_input_cols,
            self._available_output_cols
        )
    

    def _extract_and_cache_cdu_data(self, raw_data: pd.DataFrame):
        """Extract per-CDU data and create memory-mapped cache."""
        T = len(raw_data)
        n_in = self.column_builder.num_input_features
        n_out = self.column_builder.num_output_features
        
        if self.verbose:
            print(f"Extracting CDU data: {self.num_cdus} CDUs × {T} timesteps")
            # Debug: Check which output columns are actually available
            sample_cdu_cols = self.column_builder.output_cols_training_per_cdu[1]
            available_set = set(raw_data.columns)
            missing_cols = [c for c in sample_cdu_cols if c not in available_set]
            if missing_cols:
                print(f"  WARNING: Missing output columns for CDU 1: {missing_cols[:3]}...")
                print(f"  Available columns sample: {list(raw_data.columns)[:5]}")
        
        # Pre-allocate arrays
        inputs_all = np.zeros((self.num_cdus, T, n_in), dtype=np.float32)
        outputs_all = np.zeros((self.num_cdus, T, n_out), dtype=np.float32)
        inputs_raw_all = np.zeros((self.num_cdus, T, n_in), dtype=np.float32)
        outputs_raw_all = np.zeros((self.num_cdus, T, n_out), dtype=np.float32)
        
        # Extract global input (T_ext)
        T_ext = self._safe_get_column(raw_data, self.column_builder.global_input_col)
        
        # Track if any outputs were found
        found_any_outputs = False
        
        # Process each CDU
        for cdu_idx, cdu_id in enumerate(range(1, self.num_cdus + 1)):
            # Extract inputs: [Q_flow, T_Air, T_ext]
            Q_flow = self._safe_get_column(
                raw_data, 
                self.column_builder.input_cols_per_cdu[cdu_id][0]
            )
            T_Air = self._safe_get_column(
                raw_data,
                self.column_builder.input_cols_per_cdu[cdu_id][1]
            )
            
            inputs_raw = np.column_stack([Q_flow, T_Air, T_ext]).astype(np.float32)
            inputs_raw_all[cdu_idx] = inputs_raw
            
            # Extract outputs - check if columns exist
            output_cols = self.column_builder.output_cols_training_per_cdu[cdu_id]
            output_arrays = []
            for col in output_cols:
                col_data = self._safe_get_column(raw_data, col)
                output_arrays.append(col_data)
                if col in raw_data.columns and np.any(col_data != 0):
                    found_any_outputs = True
            
            outputs_raw = np.column_stack(output_arrays).astype(np.float32)
            outputs_raw_all[cdu_idx] = outputs_raw
            
            # Normalize
            inputs_all[cdu_idx] = self._normalize_inputs(inputs_raw, cdu_id)
            outputs_all[cdu_idx] = self._normalize_outputs(outputs_raw, cdu_id)
        
        # Warn if no outputs found
        if not found_any_outputs and self.verbose:
            print("  WARNING: All output columns are zero or missing!")
            print("  This likely means the parquet files don't contain CDU output data.")
            print("  Check that your data files include columns like:")
            print(f"    {self.column_builder.output_cols_training_per_cdu[1][0]}")
        
        # Create memory-mapped storage
        if self.data_config.use_mmap:
            if self.verbose:
                print("Creating memory-mapped storage...")
            
            self._mmap_storage = MMapStorage(
                self.cache_dir / 'mmap',
                self.num_cdus,
                T,
                n_in,
                n_out
            )
            
            self._mmap_storage.create(
                inputs_all, outputs_all,
                inputs_raw_all, outputs_raw_all,
                metadata={
                    'chunks': self.dask_backend.chunk_indices,
                    'system_name': self.system_name,
                    'norm_mean_in': self._norm_handler.mean_in.tolist() if self._norm_handler.mean_in is not None else None,
                    'norm_std_in': self._norm_handler.std_in.tolist() if self._norm_handler.std_in is not None else None,
                    'norm_mean_out': self._norm_handler.mean_out.tolist() if self._norm_handler.mean_out is not None else None,
                    'norm_std_out': self._norm_handler.std_out.tolist() if self._norm_handler.std_out is not None else None,
                    'input_cols': self._available_input_cols,
                    'output_cols': self._available_output_cols
                }
            )
        else:
            # Keep in memory (for smaller datasets)
            self._inputs = inputs_all
            self._outputs = outputs_all
            self._inputs_raw = inputs_raw_all
            self._outputs_raw = outputs_raw_all
    
    def _load_from_cache(self):
        """Load data from existing cache."""
        self._mmap_storage = MMapStorage(
            self.cache_dir / 'mmap',
            self.num_cdus,
            0,  # Will be loaded from metadata
            self.column_builder.num_input_features,
            self.column_builder.num_output_features
        )
        
        metadata = self._mmap_storage.load_metadata()
        self._num_timesteps = metadata['num_timesteps']
        self._mmap_storage.num_timesteps = self._num_timesteps
        
        # Restore normalization handler
        self._norm_handler = NormalizationHandler()
        if metadata.get('norm_mean_in'):
            self._norm_handler.mean_in = np.array(metadata['norm_mean_in'], dtype=np.float32)
            self._norm_handler.std_in = np.array(metadata['norm_std_in'], dtype=np.float32)
            self._norm_handler.mean_out = np.array(metadata['norm_mean_out'], dtype=np.float32)
            self._norm_handler.std_out = np.array(metadata['norm_std_out'], dtype=np.float32)
        
        self._available_input_cols = metadata.get('input_cols', [])
        self._available_output_cols = metadata.get('output_cols', [])
    
    def _safe_get_column(self, df: pd.DataFrame, col: str) -> np.ndarray:
        """Safely get column data, returning zeros if not found."""
        if col in df.columns:
            return df[col].values.astype(np.float32)
        return np.zeros(len(df), dtype=np.float32)
    
    def _normalize_inputs(self, inputs: np.ndarray, cdu_id: int) -> np.ndarray:
        """Normalize inputs for a specific CDU."""
        if self._norm_handler is None or self._norm_handler.mean_in is None:
            return inputs
        
        # Build column-to-index mapping
        input_cols = (
            self.column_builder.input_cols_per_cdu[cdu_id] + 
            [self.column_builder.global_input_col]
        )
        
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
        
        for i, col in enumerate(input_cols):
            if col in self._available_input_cols:
                idx = self._available_input_cols.index(col)
                if idx < len(self._norm_handler.mean_in):
                    mean[i] = self._norm_handler.mean_in[idx]
                    std[i] = self._norm_handler.std_in[idx]
        
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return ((inputs - mean) / std).astype(np.float32)
    
    def _normalize_outputs(self, outputs: np.ndarray, cdu_id: int) -> np.ndarray:
        """Normalize outputs for a specific CDU."""
        if self._norm_handler is None or self._norm_handler.mean_out is None:
            return outputs
        
        output_cols = self.column_builder.output_cols_training_per_cdu[cdu_id]
        n_out = outputs.shape[1]
        
        mean = np.zeros(n_out, dtype=np.float32)
        std = np.ones(n_out, dtype=np.float32)
        
        for i, col in enumerate(output_cols):
            if col in self._available_output_cols:
                idx = self._available_output_cols.index(col)
                if idx < len(self._norm_handler.mean_out):
                    mean[i] = self._norm_handler.mean_out[idx]
                    std[i] = self._norm_handler.std_out[idx]
        
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return ((outputs - mean) / std).astype(np.float32)
    
    def get_data(self, cdu_idx: int, start: int, end: int,
                 normalized: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific CDU and time range.
        
        Args:
            cdu_idx: CDU index (0-based)
            start: Start timestep
            end: End timestep
            normalized: Return normalized or raw data
            
        Returns:
            Tuple of (inputs, outputs)
        """
        if not self._is_prepared:
            raise RuntimeError("Data manager not prepared. Call prepare() first.")
        
        if self._mmap_storage is not None:
            return self._mmap_storage.get_cdu_data(cdu_idx, start, end, normalized)
        else:
            if normalized:
                return (
                    self._inputs[cdu_idx, start:end].copy(),
                    self._outputs[cdu_idx, start:end].copy()
                )
            else:
                return (
                    self._inputs_raw[cdu_idx, start:end].copy(),
                    self._outputs_raw[cdu_idx, start:end].copy()
                )
    
    def create_view(
        self,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        cdu_range: Optional[Tuple[int, int]] = None
    ) -> 'CDUDataView':
        """
        Create a view into the data for a specific split.
        
        Args:
            start_ratio: Start position as ratio of total data [0, 1]
            end_ratio: End position as ratio of total data [0, 1]
            cdu_range: Optional (start_cdu, end_cdu) range (1-indexed, inclusive)
            
        Returns:
            CDUDataView for the specified range
        """
        if not self._is_prepared:
            raise RuntimeError("Data manager not prepared. Call prepare() first.")
        
        return CDUDataView(
            data_manager=self,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            cdu_range=cdu_range,
            tno_config=self.tno_config
        )
    
    def create_train_val_test_views(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple['CDUDataView', 'CDUDataView', 'CDUDataView']:
        """
        Create train, validation, and test views.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            
        Returns:
            Tuple of (train_view, val_view, test_view)
        """
        train_end = train_ratio
        val_end = train_ratio + val_ratio
        
        train_view = self.create_view(0.0, train_end)
        val_view = self.create_view(train_end, val_end)
        test_view = self.create_view(val_end, 1.0)
        
        return train_view, val_view, test_view
    
    @property
    def num_timesteps(self) -> int:
        """Get total number of timesteps."""
        return self._num_timesteps
    
    @property
    def norm_handler(self) -> Optional[NormalizationHandler]:
        """Get normalization handler."""
        return self._norm_handler
    
    def cleanup(self):
        """Clean up resources."""
        if self._mmap_storage is not None:
            self._mmap_storage.cleanup()


# =============================================================================
# CDU Data View - View-based dataset for splits
# =============================================================================

class CDUDataView(Dataset):
    """
    View into CDUDataManager for a specific data split.
    
    This is a lightweight wrapper that doesn't copy data, just
    defines the view boundaries and accesses data through the manager.
    """
    
    def __init__(
        self,
        data_manager: CDUDataManager,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        cdu_range: Optional[Tuple[int, int]] = None,
        tno_config: Optional[TNOSequenceConfig] = None
    ):
        """
        Initialize data view.
        
        Args:
            data_manager: Parent CDUDataManager
            start_ratio: Start position as ratio [0, 1]
            end_ratio: End position as ratio [0, 1]
            cdu_range: Optional (start_cdu, end_cdu) 1-indexed inclusive
            tno_config: TNO sequence configuration
        """
        self.data_manager = data_manager
        self.tno_config = tno_config or data_manager.tno_config
        
        # Time range
        T = data_manager.num_timesteps
        self.time_start = int(T * start_ratio)
        self.time_end = int(T * end_ratio)
        self.num_timesteps = self.time_end - self.time_start
        
        # CDU range (0-indexed internally)
        if cdu_range is not None:
            self.cdu_start = cdu_range[0] - 1  # Convert to 0-indexed
            self.cdu_end = cdu_range[1]  # End is exclusive
        else:
            self.cdu_start = 0
            self.cdu_end = data_manager.num_cdus
        self.num_cdus = self.cdu_end - self.cdu_start
        
        # Create sequence indices
        self.sequences = self._create_sequence_indices()
    
    def _create_sequence_indices(self) -> List[Dict]:
        """Create list of valid sequence indices for this view."""
        L = self.tno_config.history_length
        K = self.tno_config.prediction_horizon
        stride = self.tno_config.stride
        
        # Valid timestep range within our view
        # Need L-1 history before current, and K future after
        valid_start = L - 1
        valid_end = self.num_timesteps - K
        
        if valid_end <= valid_start:
            return []
        
        sequences = []
        for cdu_offset in range(self.num_cdus):
            cdu_idx = self.cdu_start + cdu_offset  # Global CDU index
            
            for t_local in range(valid_start, valid_end, stride):
                # Convert to global time indices
                t_global = self.time_start + t_local
                
                sequences.append({
                    'cdu_idx': cdu_idx,
                    'cdu_id': cdu_idx + 1,  # 1-indexed for external use
                    't_local': t_local,
                    'hist_start': t_global - L + 1,
                    'hist_end': t_global + 1,
                    'future_start': t_global + 1,
                    'future_end': t_global + 1 + K
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        
        Returns dict with:
            - u_hist: (L, 3) input history [Q_flow, T_Air, T_ext]
            - y_hist: (L, num_output_features) output history
            - y_future: (K, num_output_features) future targets
            - Q_flow_raw: (L,) unnormalized Q_flow for physics losses
            - cdu_id: CDU identifier (1-indexed)
            - timestep: Local timestep index within view
        """
        seq = self.sequences[idx]
        cdu_idx = seq['cdu_idx']
        
        # Get history data
        inputs_hist, outputs_hist = self.data_manager.get_data(
            cdu_idx, seq['hist_start'], seq['hist_end'], normalized=True
        )
        
        # Get future data
        _, outputs_future = self.data_manager.get_data(
            cdu_idx, seq['future_start'], seq['future_end'], normalized=True
        )
        
        # Get raw Q_flow for physics losses
        inputs_raw, _ = self.data_manager.get_data(
            cdu_idx, seq['hist_start'], seq['hist_end'], normalized=False
        )
        Q_flow_raw = inputs_raw[:, 0]  # First column is Q_flow
        
        return {
            'u_hist': torch.from_numpy(inputs_hist),
            'y_hist': torch.from_numpy(outputs_hist),
            'y_future': torch.from_numpy(outputs_future),
            'Q_flow_raw': torch.from_numpy(Q_flow_raw),
            'cdu_id': torch.tensor(seq['cdu_id'], dtype=torch.long),
            'timestep': torch.tensor(seq['t_local'], dtype=torch.long)
        }
    
    @property
    def num_input_features(self) -> int:
        """Number of input features per CDU."""
        return self.data_manager.column_builder.num_input_features
    
    @property
    def num_output_features(self) -> int:
        """Number of output features per CDU."""
        return self.data_manager.column_builder.num_output_features
    
    def get_denormalization_params(self, cdu_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (mean, std) for denormalizing a specific CDU's outputs."""
        norm_handler = self.data_manager.norm_handler
        output_cols = self.data_manager.column_builder.output_cols_training_per_cdu[cdu_id]
        available_cols = self.data_manager._available_output_cols
        
        n_out = len(output_cols)
        mean = np.zeros(n_out, dtype=np.float32)
        std = np.ones(n_out, dtype=np.float32)
        
        if norm_handler is None or norm_handler.mean_out is None:
            return mean, std
        
        for i, col in enumerate(output_cols):
            if col in available_cols:
                idx = available_cols.index(col)
                if idx < len(norm_handler.mean_out):
                    mean[i] = norm_handler.mean_out[idx]
                    std[i] = norm_handler.std_out[idx]
        
        return mean, std


# =============================================================================
# Collate Function and DataLoader Factory
# =============================================================================

def tno_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for TNO sequences."""
    return {
        'u_hist': torch.stack([b['u_hist'] for b in batch]),
        'y_hist': torch.stack([b['y_hist'] for b in batch]),
        'y_future': torch.stack([b['y_future'] for b in batch]),
        'Q_flow_raw': torch.stack([b['Q_flow_raw'] for b in batch]),
        'cdu_id': torch.stack([b['cdu_id'] for b in batch]),
        'timestep': torch.stack([b['timestep'] for b in batch])
    }


def create_data_manager(
    data_path: str,
    cdu_lim: Optional[int] = None,
    chunk_indices: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    use_mmap: bool = True,
    tno_config: Optional[TNOSequenceConfig] = None,
    system_name: str = 'summit',
    config: Optional[Dict] = None,
    verbose: bool = True
) -> CDUDataManager:
    """
    Factory function to create a CDUDataManager.
    
    Args:
        data_path: Path to data directory
        cdu_lim: Maximum number of CDUs to load
        chunk_indices: Specific chunks to load
        cache_dir: Directory for cache storage
        num_workers: Number of parallel I/O workers
        use_mmap: Whether to use memory-mapped storage
        tno_config: TNO sequence configuration
        system_name: System name for config lookup
        config: Optional pre-loaded config
        verbose: Print progress information
        
    Returns:
        Prepared CDUDataManager instance
    """
    data_config = LazyDataConfig(
        cdu_lim=cdu_lim,
        chunk_indices=chunk_indices,
        cache_dir=cache_dir,
        num_workers=num_workers,
        use_mmap=use_mmap
    )
    
    manager = CDUDataManager(
        data_path=data_path,
        data_config=data_config,
        tno_config=tno_config,
        system_name=system_name,
        config=config,
        verbose=verbose
    )
    
    return manager.prepare()


def create_tno_data_loaders(
    data_path: str,
    tno_config: Optional[TNOSequenceConfig] = None,
    batch_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    system_name: str = 'summit',
    cdu_lim: Optional[int] = None,
    chunk_indices: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    use_mmap: bool = True,
    config: Optional[Dict] = None,
    verbose: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, CDUDataManager]:
    """
    Create train, validation, and test data loaders with optimized loading.
    
    Args:
        data_path: Path to data directory
        tno_config: TNO sequence configuration
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        distributed: Whether using distributed training
        system_name: System name for config lookup
        cdu_lim: Maximum number of CDUs
        chunk_indices: Specific chunks to load
        cache_dir: Cache directory for mmap storage
        use_mmap: Whether to use memory-mapped storage
        config: Optional pre-loaded config
        verbose: Print progress information
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, data_manager)
    """
    tno_config = tno_config or TNOSequenceConfig()
    
    # Create and prepare data manager
    manager = create_data_manager(
        data_path=data_path,
        cdu_lim=cdu_lim,
        chunk_indices=chunk_indices,
        cache_dir=cache_dir,
        num_workers=num_workers,
        use_mmap=use_mmap,
        tno_config=tno_config,
        system_name=system_name,
        config=config,
        verbose=verbose
    )
    
    # Create views for each split
    train_view, val_view, test_view = manager.create_train_val_test_views(
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    if verbose:
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_view)} sequences")
        print(f"  Val: {len(val_view)} sequences")
        print(f"  Test: {len(test_view)} sequences")
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_view, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_view, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_view, shuffle=False) if distributed else None
    
    # Determine if we can use persistent workers
    use_persistent = persistent_workers and num_workers > 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_view,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_view,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_view,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    if verbose:
        print(f"\nDataLoader Summary:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Persistent workers: {use_persistent}")
    
    return train_loader, val_loader, test_loader, manager


# =============================================================================
# Multi-Chunk Data Manager - For chunk-based splitting
# =============================================================================

class MultiChunkDataManager:
    """
    Data manager that handles chunk-based train/val/test splitting.
    
    Unlike temporal splitting within a single data manager, this class
    creates separate managers for different chunk sets while sharing
    normalization statistics computed from training chunks.
    """
    
    def __init__(
        self,
        data_path: str,
        train_chunks: List[int],
        val_chunks: List[int],
        test_chunks: List[int],
        data_config: Optional[LazyDataConfig] = None,
        tno_config: Optional[TNOSequenceConfig] = None,
        system_name: str = 'summit',
        config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize multi-chunk data manager.
        
        Args:
            data_path: Path to data directory
            train_chunks: Chunk indices for training
            val_chunks: Chunk indices for validation
            test_chunks: Chunk indices for testing
            data_config: Lazy data loading configuration
            tno_config: TNO sequence configuration
            system_name: System name for config lookup
            config: Optional pre-loaded config
            verbose: Print progress information
        """
        self.data_path = Path(data_path)
        self.train_chunks = train_chunks
        self.val_chunks = val_chunks
        self.test_chunks = test_chunks
        self.data_config = data_config or LazyDataConfig()
        self.tno_config = tno_config or TNOSequenceConfig()
        self.system_name = system_name
        self.config = config
        self.verbose = verbose
        
        self._train_manager: Optional[CDUDataManager] = None
        self._val_manager: Optional[CDUDataManager] = None
        self._test_manager: Optional[CDUDataManager] = None
        self._is_prepared = False
    
    def prepare(self) -> 'MultiChunkDataManager':
        """
        Prepare all data managers.
        
        Training manager is prepared first to compute normalization stats,
        which are then shared with validation and test managers.
        """
        if self._is_prepared:
            return self
        
        if self.verbose:
            print(f"Preparing multi-chunk data managers...")
            print(f"  Train chunks: {self.train_chunks}")
            print(f"  Val chunks: {self.val_chunks}")
            print(f"  Test chunks: {self.test_chunks}")
        
        # Create training manager first (computes normalization)
        train_config = LazyDataConfig(
            cdu_lim=self.data_config.cdu_lim,
            chunk_indices=self.train_chunks,
            cache_dir=self._get_cache_dir('train'),
            num_workers=self.data_config.num_workers,
            use_mmap=self.data_config.use_mmap
        )
        
        self._train_manager = CDUDataManager(
            data_path=str(self.data_path),
            data_config=train_config,
            tno_config=self.tno_config,
            system_name=self.system_name,
            config=self.config,
            verbose=self.verbose
        ).prepare()
        
        # Get normalization stats from training
        train_norm = self._train_manager.norm_handler
        
        # Create validation manager with shared normalization
        if self.val_chunks:
            self._val_manager = self._create_split_manager(
                'val', self.val_chunks, train_norm
            )
        
        # Create test manager with shared normalization
        if self.test_chunks:
            self._test_manager = self._create_split_manager(
                'test', self.test_chunks, train_norm
            )
        
        self._is_prepared = True
        return self
    
    def _get_cache_dir(self, split: str) -> str:
        """Get cache directory for a split."""
        if self.data_config.cache_dir:
            return str(Path(self.data_config.cache_dir) / split)
        return None
    
    def _create_split_manager(
        self,
        split_name: str,
        chunks: List[int],
        norm_handler: NormalizationHandler
    ) -> CDUDataManager:
        """Create a data manager for a non-training split."""
        split_config = LazyDataConfig(
            cdu_lim=self.data_config.cdu_lim,
            chunk_indices=chunks,
            cache_dir=self._get_cache_dir(split_name),
            num_workers=self.data_config.num_workers,
            use_mmap=self.data_config.use_mmap
        )
        
        manager = CDUDataManager(
            data_path=str(self.data_path),
            data_config=split_config,
            tno_config=self.tno_config,
            system_name=self.system_name,
            config=self.config,
            verbose=self.verbose
        )
        
        # Inject training normalization stats before prepare
        manager._norm_handler = norm_handler
        
        return manager.prepare()
    
    def get_train_view(self) -> CDUDataView:
        """Get training data view (full range of training chunks)."""
        if not self._is_prepared:
            raise RuntimeError("Manager not prepared. Call prepare() first.")
        return self._train_manager.create_view(0.0, 1.0)
    
    def get_val_view(self) -> CDUDataView:
        """Get validation data view."""
        if not self._is_prepared:
            raise RuntimeError("Manager not prepared. Call prepare() first.")
        if self._val_manager is None:
            raise ValueError("No validation chunks specified")
        return self._val_manager.create_view(0.0, 1.0)
    
    def get_test_view(self) -> CDUDataView:
        """Get test data view."""
        if not self._is_prepared:
            raise RuntimeError("Manager not prepared. Call prepare() first.")
        if self._test_manager is None:
            raise ValueError("No test chunks specified")
        return self._test_manager.create_view(0.0, 1.0)
    
    @property
    def norm_handler(self) -> NormalizationHandler:
        """Get normalization handler (from training data)."""
        return self._train_manager.norm_handler
    
    def cleanup(self):
        """Clean up all managers."""
        if self._train_manager:
            self._train_manager.cleanup()
        if self._val_manager:
            self._val_manager.cleanup()
        if self._test_manager:
            self._test_manager.cleanup()


def create_tno_data_loaders_multi_chunk(
    data_path: str,
    train_chunks: List[int],
    val_chunks: List[int],
    test_chunks: List[int],
    tno_config: Optional[TNOSequenceConfig] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    system_name: str = 'summit',
    cdu_lim: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_mmap: bool = True,
    config: Optional[Dict] = None,
    verbose: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, MultiChunkDataManager]:
    """
    Create data loaders with chunk-based splitting.
    
    Args:
        data_path: Path to data directory
        train_chunks: Chunk indices for training
        val_chunks: Chunk indices for validation
        test_chunks: Chunk indices for testing
        tno_config: TNO sequence configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        distributed: Whether using distributed training
        system_name: System name for config
        cdu_lim: Maximum CDUs to load
        cache_dir: Cache directory
        use_mmap: Use memory-mapped storage
        config: Pre-loaded config
        verbose: Print progress
        persistent_workers: Keep workers alive
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, multi_chunk_manager)
    """
    tno_config = tno_config or TNOSequenceConfig()
    
    data_config = LazyDataConfig(
        cdu_lim=cdu_lim,
        cache_dir=cache_dir,
        num_workers=num_workers,
        use_mmap=use_mmap
    )
    
    # Create and prepare multi-chunk manager
    manager = MultiChunkDataManager(
        data_path=data_path,
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        test_chunks=test_chunks,
        data_config=data_config,
        tno_config=tno_config,
        system_name=system_name,
        config=config,
        verbose=verbose
    ).prepare()
    
    # Get views
    train_view = manager.get_train_view()
    val_view = manager.get_val_view()
    test_view = manager.get_test_view()
    
    if verbose:
        print(f"\nDataset splits (chunk-based):")
        print(f"  Train: {len(train_view)} sequences from chunks {train_chunks}")
        print(f"  Val: {len(val_view)} sequences from chunks {val_chunks}")
        print(f"  Test: {len(test_view)} sequences from chunks {test_chunks}")
    
    # Create samplers
    train_sampler = DistributedSampler(train_view, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_view, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_view, shuffle=False) if distributed else None
    
    use_persistent = persistent_workers and num_workers > 0
    
    # Create loaders
    train_loader = DataLoader(
        train_view,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_view,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_view,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=tno_collate_fn,
        persistent_workers=use_persistent
    )
    
    if verbose:
        print(f"\nDataLoader Summary:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, manager


# =============================================================================
# Auto Data Loader Factory - Intelligent split selection
# =============================================================================

def auto_create_data_loaders(
    data_path: str,
    tno_config: Optional[TNOSequenceConfig] = None,
    split_config: Optional[SplitConfig] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    system_name: str = 'summit',
    cdu_lim: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_mmap: bool = True,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Union[CDUDataManager, MultiChunkDataManager]]:
    """
    Automatically create data loaders with intelligent split strategy selection.
    
    Automatically determines whether to use:
    - Temporal splitting (single chunk or overlapping chunks)
    - Chunk-based splitting (multiple distinct chunks)
    
    Args:
        data_path: Path to data directory
        tno_config: TNO sequence configuration
        split_config: Split configuration (ratios and strategy)
        batch_size: Batch size
        num_workers: Data loading workers
        pin_memory: Pin memory for GPU
        distributed: Distributed training mode
        system_name: System name
        cdu_lim: Maximum CDUs
        cache_dir: Cache directory
        use_mmap: Use memory-mapped storage
        config: Pre-loaded config
        verbose: Print progress
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, data_manager)
    """
    data_path = Path(data_path)
    tno_config = tno_config or TNOSequenceConfig()
    split_config = split_config or SplitConfig()
    
    # Discover available chunks
    available_chunks = []
    for chunk_dir in sorted(data_path.glob("chunk_*")):
        try:
            chunk_idx = int(chunk_dir.name.split("_")[1])
            parquet_files = (list(chunk_dir.glob("*_output_*.parquet")) or 
                            list(chunk_dir.glob("*.parquet")))
            if parquet_files:
                available_chunks.append(chunk_idx)
        except (ValueError, IndexError):
            continue
    
    if not available_chunks:
        raise ValueError(f"No valid chunks found in {data_path}")
    
    if verbose:
        print(f"Found {len(available_chunks)} available chunks: {available_chunks}")
    
    # Determine split strategy
    explicit_chunks = any([
        split_config.train_chunks,
        split_config.val_chunks,
        split_config.test_chunks
    ])
    
    if split_config.split_by == 'temporal':
        use_temporal = True
    elif split_config.split_by == 'chunks':
        use_temporal = False
    else:  # auto
        # Use temporal if only 1 chunk or if explicit chunks overlap
        if len(available_chunks) == 1:
            use_temporal = True
        elif explicit_chunks:
            train_set = set(split_config.train_chunks or [])
            val_set = set(split_config.val_chunks or [])
            test_set = set(split_config.test_chunks or [])
            # Check for any overlap
            use_temporal = bool(
                train_set & val_set or 
                train_set & test_set or 
                val_set & test_set
            )
        else:
            # Enough chunks for chunk-based splitting
            use_temporal = len(available_chunks) < 3
    
    if use_temporal:
        if verbose:
            print("Using temporal splitting strategy")
        
        # Determine which chunks to load
        if explicit_chunks:
            all_chunks = list(set(
                (split_config.train_chunks or []) +
                (split_config.val_chunks or []) +
                (split_config.test_chunks or [])
            ))
            chunk_indices = [c for c in all_chunks if c in available_chunks]
        else:
            chunk_indices = available_chunks
        
        return create_tno_data_loaders(
            data_path=str(data_path),
            tno_config=tno_config,
            batch_size=batch_size,
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
            num_workers=num_workers,
            pin_memory=pin_memory,
            distributed=distributed,
            system_name=system_name,
            cdu_lim=cdu_lim,
            chunk_indices=chunk_indices,
            cache_dir=cache_dir,
            use_mmap=use_mmap,
            config=config,
            verbose=verbose
        )
    else:
        if verbose:
            print("Using chunk-based splitting strategy")
        
        # Determine chunk assignments
        if explicit_chunks:
            train_chunks = [c for c in (split_config.train_chunks or []) if c in available_chunks]
            val_chunks = [c for c in (split_config.val_chunks or []) if c in available_chunks]
            test_chunks = [c for c in (split_config.test_chunks or []) if c in available_chunks]
        else:
            # Auto-split chunks
            n = len(available_chunks)
            n_test = max(1, int(n * split_config.test_ratio))
            n_val = max(1, int(n * split_config.val_ratio))
            n_train = n - n_test - n_val
            
            train_chunks = available_chunks[:n_train]
            val_chunks = available_chunks[n_train:n_train + n_val]
            test_chunks = available_chunks[n_train + n_val:]
        
        return create_tno_data_loaders_multi_chunk(
            data_path=str(data_path),
            train_chunks=train_chunks,
            val_chunks=val_chunks,
            test_chunks=test_chunks,
            tno_config=tno_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            distributed=distributed,
            system_name=system_name,
            cdu_lim=cdu_lim,
            cache_dir=cache_dir,
            use_mmap=use_mmap,
            config=config,
            verbose=verbose
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_sample_batch(loader: DataLoader) -> Dict[str, torch.Tensor]:
    """Get a single batch from dataloader for inspection."""
    return next(iter(loader))


def print_batch_shapes(batch: Dict[str, torch.Tensor]) -> None:
    """Print shapes of all tensors in a batch."""
    print("Batch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape} (dtype: {tensor.dtype})")


def inspect_view(view: CDUDataView) -> Dict:
    """Get summary statistics about a data view."""
    return {
        'num_sequences': len(view),
        'num_cdus': view.num_cdus,
        'num_timesteps': view.num_timesteps,
        'time_range': (view.time_start, view.time_end),
        'cdu_range': (view.cdu_start + 1, view.cdu_end),  # 1-indexed
        'history_length': view.tno_config.history_length,
        'prediction_horizon': view.tno_config.prediction_horizon,
        'num_input_features': view.num_input_features,
        'num_output_features': view.num_output_features
    }


def inspect_manager(manager: Union[CDUDataManager, MultiChunkDataManager]) -> Dict:
    """Get summary statistics about a data manager."""
    if isinstance(manager, MultiChunkDataManager):
        # Handle MultiChunkDataManager
        train_manager = manager._train_manager
        return {
            'manager_type': 'MultiChunkDataManager',
            'num_cdus': train_manager.num_cdus if train_manager else None,
            'train_timesteps': train_manager.num_timesteps if train_manager else None,
            'val_timesteps': manager._val_manager.num_timesteps if manager._val_manager else None,
            'test_timesteps': manager._test_manager.num_timesteps if manager._test_manager else None,
            'train_chunks': manager.train_chunks,
            'val_chunks': manager.val_chunks,
            'test_chunks': manager.test_chunks,
            'num_input_features': train_manager.column_builder.num_input_features if train_manager else None,
            'num_output_features': train_manager.column_builder.num_output_features if train_manager else None,
            'cache_dir': str(manager.data_config.cache_dir) if manager.data_config.cache_dir else None,
            'use_mmap': manager.data_config.use_mmap,
            'is_prepared': manager._is_prepared
        }
    else: 
        # Handle CDUDataManager
        return {
            'manager_type': 'CDUDataManager',
            'num_cdus': manager.num_cdus,
            'num_timesteps': manager.num_timesteps,
            'chunks': manager.dask_backend.chunk_indices,
            'num_input_features': manager.column_builder.num_input_features,
            'num_output_features': manager.column_builder.num_output_features,
            'cache_dir': str(manager.cache_dir),
            'use_mmap': manager.data_config.use_mmap,
            'is_prepared': manager._is_prepared
        }


def benchmark_dataloader(
    loader: DataLoader,
    num_batches: int = 100,
    warmup_batches: int = 10
) -> Dict[str, float]:
    """
    Benchmark data loading performance.
    
    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches
        
    Returns:
        Dict with timing statistics
    """
    import time
    
    # Warmup
    loader_iter = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        try:
            _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            _ = next(loader_iter)
    
    # Benchmark
    times = []
    loader_iter = iter(loader)
    for _ in range(min(num_batches, len(loader))):
        start = time.perf_counter()
        try:
            _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            _ = next(loader_iter)
        times.append(time.perf_counter() - start)
    
    times = np.array(times)
    batch_size = loader.batch_size
    
    return {
        'mean_batch_time_ms': float(times.mean() * 1000),
        'std_batch_time_ms': float(times.std() * 1000),
        'min_batch_time_ms': float(times.min() * 1000),
        'max_batch_time_ms': float(times.max() * 1000),
        'samples_per_second': float(batch_size / times.mean()),
        'num_batches_measured': len(times)
    }


# =============================================================================
# Context Manager for Clean Resource Management
# =============================================================================

class DataLoaderContext:
    """
    Context manager for automatic cleanup of data loader resources.
    
    Usage:
        with DataLoaderContext(data_path, ...) as ctx:
            train_loader, val_loader, test_loader = ctx.loaders
            # Use loaders
        # Resources automatically cleaned up
    """
    
    def __init__(
        self,
        data_path: str,
        tno_config: Optional[TNOSequenceConfig] = None,
        split_config: Optional[SplitConfig] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        cdu_lim: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_mmap: bool = True,
        cleanup_on_exit: bool = True,
        **kwargs
    ):
        self.data_path = data_path
        self.tno_config = tno_config
        self.split_config = split_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cdu_lim = cdu_lim
        self.cache_dir = cache_dir
        self.use_mmap = use_mmap
        self.cleanup_on_exit = cleanup_on_exit
        self.kwargs = kwargs
        
        self._manager = None
        self._loaders = None
    
    def __enter__(self) -> 'DataLoaderContext':
        train_loader, val_loader, test_loader, self._manager = auto_create_data_loaders(
            data_path=self.data_path,
            tno_config=self.tno_config,
            split_config=self.split_config,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            cdu_lim=self.cdu_lim,
            cache_dir=self.cache_dir,
            use_mmap=self.use_mmap,
            **self.kwargs
        )
        self._loaders = (train_loader, val_loader, test_loader)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit and self._manager is not None:
            self._manager.cleanup()
        return False
    
    @property
    def loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get (train_loader, val_loader, test_loader) tuple."""
        return self._loaders
    
    @property
    def train_loader(self) -> DataLoader:
        return self._loaders[0]
    
    @property
    def val_loader(self) -> DataLoader:
        return self._loaders[1]
    
    @property
    def test_loader(self) -> DataLoader:
        return self._loaders[2]
    
    @property
    def manager(self) -> Union[CDUDataManager, MultiChunkDataManager]:
        return self._manager


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

class CDUPooledSequenceDataset(CDUDataView):
    """
    Backward-compatible wrapper for existing code.
    
    This class provides the same interface as the original
    CDUPooledSequenceDataset but uses the new optimized backend.
    """
    
    def __init__(
        self,
        data_path: str,
        chunk_indices: List[int],
        tno_config: Optional[TNOSequenceConfig] = None,
        norm_handler: Optional[NormalizationHandler] = None,
        normalize: bool = True,
        config: Optional[Dict] = None,
        cdu_lim: Optional[int] = 50,
        system_name: str = 'summit',
        verbose: bool = True
    ):
        """Initialize with backward-compatible parameters."""
        # Create data manager
        self._internal_manager = create_data_manager(
            data_path=data_path,
            cdu_lim=cdu_lim,
            chunk_indices=chunk_indices,
            num_workers=4,
            use_mmap=True,
            tno_config=tno_config,
            system_name=system_name,
            config=config,
            verbose=verbose
        )
        
        # Initialize as view of full data
        super().__init__(
            data_manager=self._internal_manager,
            start_ratio=0.0,
            end_ratio=1.0,
            cdu_range=None,
            tno_config=tno_config
        )
        
        # Expose for backward compatibility
        self.raw_data = None  # Not available in new implementation
        self.norm_handler = self._internal_manager.norm_handler
        self.chunk_indices = chunk_indices
        self.cdu_lim = cdu_lim
        self.num_cdus_total = self._internal_manager.num_cdus
    
    def _print_summary(self):
        """Print dataset summary (backward compatibility)."""
        print(f"Created CDUPooledSequenceDataset (optimized):")
        print(f"  - Total sequences: {len(self)}")
        print(f"  - History length (L): {self.tno_config.history_length}")
        print(f"  - Prediction horizon (K): {self.tno_config.prediction_horizon}")
        print(f"  - CDUs pooled: {self.num_cdus}")
        print(f"  - Input features per CDU: {self.num_input_features}")
        print(f"  - Output features per CDU: {self.num_output_features}")


class CDUPooledSingleChunkDataset(CDUDataView):
    """
    Backward-compatible wrapper for single chunk with temporal splitting.
    """
    
    def __init__(
        self,
        data_path: str,
        chunk_idx: int,
        tno_config: Optional[TNOSequenceConfig] = None,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        norm_handler: Optional[NormalizationHandler] = None,
        normalize: bool = True,
        config: Optional[Dict] = None,
        cdu_lim: Optional[int] = 50,
        system_name: str = 'summit',
        verbose: bool = True
    ):
        """Initialize with backward-compatible parameters."""
        # Create data manager
        self._internal_manager = create_data_manager(
            data_path=data_path,
            cdu_lim=cdu_lim,
            chunk_indices=[chunk_idx],
            num_workers=4,
            use_mmap=True,
            tno_config=tno_config,
            system_name=system_name,
            config=config,
            verbose=verbose
        )
        
        # Override norm handler if provided
        if norm_handler is not None:
            self._internal_manager._norm_handler = norm_handler
        
        # Initialize as view with temporal split
        super().__init__(
            data_manager=self._internal_manager,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            cdu_range=None,
            tno_config=tno_config
        )
        
        # Expose for backward compatibility
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.norm_handler = self._internal_manager.norm_handler
        self.chunk_indices = [chunk_idx]


# =============================================================================
# Main / Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example usage demonstrating the new optimized data loader
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized CDU data loader")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--cdu-lim", type=int, default=10, help="Number of CDUs to load")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Optimized CDU Data Loader Test")
    print("=" * 60)
    
    # Configuration
    tno_config = TNOSequenceConfig(
        history_length=30,
        prediction_horizon=10,
        stride=1
    )
    
    split_config = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_by='auto'
    )
    
    # Create data loaders using context manager
    with DataLoaderContext(
        data_path=args.data_path,
        tno_config=tno_config,
        split_config=split_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cdu_lim=args.cdu_lim,
        use_mmap=True,
        verbose=True
    ) as ctx:
        train_loader, val_loader, test_loader = ctx.loaders
        
        print("\n" + "=" * 60)
        print("Sample Batch")
        print("=" * 60)
        
        batch = get_sample_batch(train_loader)
        print_batch_shapes(batch)
        
        if args.benchmark:
            print("\n" + "=" * 60)
            print("Benchmark Results")
            print("=" * 60)
            
            results = benchmark_dataloader(train_loader, num_batches=100)
            for key, value in results.items():
                print(f"  {key}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)