import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import glob

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from fmu2ml.config import ModelConfig, TrainingConfig
from fmu2ml.data.processors.normalization import NormalizationHandler
from raps.config import ConfigManager

class DatacenterCoolingDataset(Dataset):
    """
    Dataset for datacenter cooling system data
    """
    
    def __init__(
        self,
        data_path: str,
        chunk_indices: List[int],
        sequence_length: int = 12,
        stride: int = 1,
        norm_handler: Optional[NormalizationHandler] = None,
        normalize: bool = True,
        config: Optional[Dict] = None,
        system_name: str = 'marconi100'
    ):
        """
        Initialize dataset
        
        Parameters:
        -----------
        data_path : str
            Path to data directory containing chunks
        chunk_indices : List[int]
            List of chunk indices to load
        sequence_length : int
            Number of timesteps in input sequence
        stride : int
            Stride for creating sequences
        norm_handler : NormalizationHandler, optional
            Existing normalization handler
        normalize : bool
            Whether to normalize data
        """
        self.data_path = Path(data_path)
        self.chunk_indices = chunk_indices
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        if config is None:
            self.config = ConfigManager(system_name=system_name).get_config()
        else:
            self.config = config
        self.num_cdus = self.config['NUM_CDUS']
        # Load and preprocess data
        self.data = self._load_chunks()
        
        # Get column names
        self.input_cols = self._get_input_columns()
        self.output_cols = self._get_output_columns()
        
        # Verify columns exist
        missing_input_cols = set(self.input_cols) - set(self.data.columns)
        missing_output_cols = set(self.output_cols) - set(self.data.columns)
        
        if missing_input_cols:
            print(f"Warning: Missing input columns: {missing_input_cols}")
            self.input_cols = [col for col in self.input_cols if col in self.data.columns]
        
        if missing_output_cols:
            print(f"Warning: Missing output columns: {missing_output_cols}")
            self.output_cols = [col for col in self.output_cols if col in self.data.columns]
        
        if not self.input_cols or not self.output_cols:
            raise ValueError("No valid input or output columns found in data")
        
        # Handle normalization
        if norm_handler is None:
            self.norm_handler = NormalizationHandler()
            self.mean_in, self.std_in, self.mean_out, self.std_out = \
                self.norm_handler.compute_stats(self.data, self.input_cols, self.output_cols)
        else:
            self.norm_handler = norm_handler
            self.mean_in = self.norm_handler.mean_in
            self.std_in = self.norm_handler.std_in
            self.mean_out = self.norm_handler.mean_out
            self.std_out = self.norm_handler.std_out

    def _load_chunks(self) -> pd.DataFrame:
        """Load specified chunks and concatenate"""
        dfs = []
        for idx in self.chunk_indices:
            chunk_dir = self.data_path / f"chunk_{idx}"
            if not chunk_dir.exists():
                print(f"Warning: Chunk directory missing: {chunk_dir}")
                continue

            candidate_files = sorted(chunk_dir.glob("*_output_*.parquet"))
            if not candidate_files:
                candidate_files = sorted(chunk_dir.glob("*.parquet"))

            if not candidate_files:
                print(f"Warning: No parquet files found in {chunk_dir}")
                continue

            for parquet_file in candidate_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                    print(f"Loaded chunk {idx} ({parquet_file.name}): {len(df)} samples")
                except Exception as e:
                    print(f"Error loading {parquet_file} for chunk {idx}: {e}")

        if not dfs:
            raise ValueError(f"No valid chunks found in {self.chunk_indices} at {self.data_path}")

        return pd.concat(dfs, ignore_index=True)
    
    def _get_input_columns(self) -> List[str]:
        """Get input column names"""
        cols = []
        # CDU inputs
        for i in range(1, self.num_cdus + 1):
            cols.append(f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total')
            cols.append(f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air')
        # Global input
        cols.append('simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext')
        return cols
    
    def _get_output_columns(self) -> List[str]:
        """Get output column names"""
        cols = []
        # CDU outputs
        output_vars = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                      'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C', 
                      'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        for i in range(1, self.num_cdus + 1):
            for var in output_vars:
                cols.append(f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{var}')
        
        # Datacenter level outputs
        cols.append('simulator[1].datacenter[1].summary.V_flow_prim_GPM')
        cols.append('pue')
        
        # HTC for each CDU
        for i in range(1, self.num_cdus + 1):
            cols.append(f'simulator[1].datacenter[1].computeBlock[{i}].cabinet[1].summary.htc')
        
        return cols
    
    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.stride
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Get sequence data
        input_seq = self.data[self.input_cols].iloc[start_idx:end_idx].values
        output_seq = self.data[self.output_cols].iloc[end_idx-1].values
        
        if self.normalize:
            input_seq = self.norm_handler.normalize_input(input_seq)
            output_seq = self.norm_handler.normalize_output(output_seq)
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(output_seq)


class SingleChunkSplitDataset(DatacenterCoolingDataset):
    """Dataset that handles internal splitting for single chunk scenarios"""
    
    def __init__(
        self,
        data_path: str,
        chunk_idx: int,
        sequence_length: int,
        stride: int = 1,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        norm_handler: Optional[NormalizationHandler] = None
    ):
        super().__init__(data_path, [chunk_idx], sequence_length, stride, norm_handler=norm_handler)
        total_samples = len(self.data) - sequence_length
        self.start_idx = int(total_samples * start_ratio)
        self.end_idx = int(total_samples * end_ratio)
    
    def __len__(self):
        return (self.end_idx - self.start_idx) // self.stride
    
    def __getitem__(self, idx):
        idx = self.start_idx + idx * self.stride
        return super().__getitem__(idx)


def create_data_loaders(
    model_config: ModelConfig,
    train_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders
    
    Parameters:
    -----------
    model_config : ModelConfig
        Model configuration
    train_config : TrainingConfig
        Training configuration
    
    Returns:
    --------
    Tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, and test data loaders
    """
    data_path = Path(train_config.data_path)
    available_chunks = []
    chunk_info = {}
    
    # Scan for available chunks
    for chunk_dir in sorted(data_path.glob("chunk_*")):
        
        try:
            parquet_file = next(chunk_dir.glob("*_output_*.parquet"), None)
        except StopIteration:
            print(f"Warning: No parquet files found in {chunk_dir}")
            continue

        if parquet_file.exists():
            try:
                chunk_idx = int(chunk_dir.name.split("_")[1])
                available_chunks.append(chunk_idx)
                
                # Get row count
                if PYARROW_AVAILABLE:
                    try:
                        metadata = pq.read_metadata(parquet_file)
                        num_rows = metadata.num_rows
                    except:
                        num_rows = len(pd.read_parquet(parquet_file))
                else:
                    num_rows = len(pd.read_parquet(parquet_file))
                
                num_sequences = max(0, (num_rows - model_config.sequence_length) // 1)
                chunk_info[chunk_idx] = {
                    'num_rows': num_rows,
                    'num_sequences': num_sequences,
                    'file_path': parquet_file
                }
            except Exception as e:
                print(f"Warning: Failed to read chunk {chunk_dir}: {e}")
                continue
    
    if not available_chunks:
        raise ValueError(f"No valid data chunks found in {train_config.data_path}")
    
    print(f"Found {len(available_chunks)} available chunks")
    
    # Handle specific chunks if requested
    if hasattr(train_config, 'specific_chunks') and train_config.specific_chunks:
        filtered_chunks = [c for c in train_config.specific_chunks if c in available_chunks]
        if filtered_chunks:
            available_chunks = filtered_chunks
            print(f"Using specific chunks: {available_chunks}")
    elif hasattr(train_config, 'num_chunks') and train_config.num_chunks > 0 and train_config.num_chunks < len(available_chunks):
        available_chunks = available_chunks[:train_config.num_chunks]
    
    # Get split ratios
    val_split = getattr(train_config, 'val_split', 0.15)
    test_split = getattr(train_config, 'test_split', 0.15)
    sample_based_split = getattr(train_config, 'sample_based_split', False)
    
    # Split chunks
    if sample_based_split and len(available_chunks) == 1:
        # Single chunk: split samples
        chunk_idx = available_chunks[0]
        
        train_dataset = SingleChunkSplitDataset(
            str(data_path), chunk_idx, model_config.sequence_length,
            start_ratio=0.0,
            end_ratio=1.0 - val_split - test_split
        )
        
        val_dataset = SingleChunkSplitDataset(
            str(data_path), chunk_idx, model_config.sequence_length,
            start_ratio=1.0 - val_split - test_split,
            end_ratio=1.0 - test_split,
            norm_handler=train_dataset.norm_handler
        )
        
        test_dataset = SingleChunkSplitDataset(
            str(data_path), chunk_idx, model_config.sequence_length,
            start_ratio=1.0 - test_split,
            end_ratio=1.0,
            norm_handler=train_dataset.norm_handler
        )
        
    else:
        # Multiple chunks: split by chunks
        n_chunks = len(available_chunks)
        n_test = max(1, int(n_chunks * test_split))
        n_val = max(1, int(n_chunks * val_split))
        n_train = n_chunks - n_test - n_val
        
        train_chunks = available_chunks[:n_train]
        val_chunks = available_chunks[n_train:n_train+n_val]
        test_chunks = available_chunks[n_train+n_val:]
        
        print(f"Train chunks: {train_chunks}")
        print(f"Val chunks: {val_chunks}")
        print(f"Test chunks: {test_chunks}")
        
        train_dataset = DatacenterCoolingDataset(
            str(data_path), train_chunks, model_config.sequence_length
        )
        
        val_dataset = DatacenterCoolingDataset(
            str(data_path), val_chunks, model_config.sequence_length,
            norm_handler=train_dataset.norm_handler
        )
        
        test_dataset = DatacenterCoolingDataset(
            str(data_path), test_chunks, model_config.sequence_length,
            norm_handler=train_dataset.norm_handler
        )
    
    # Create data loaders
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    distributed = getattr(train_config, 'distributed', False)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    num_workers = getattr(train_config, 'cpu', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader