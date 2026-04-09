"""
CDU Sequence Dataset for TNO-DeepM&Mnet Training
================================================

Extension to the existing DatacenterCoolingDataset that provides:
1. (u_hist, y_hist, y_future) sequence structure
2. CDU data pooling as independent observations
3. Temporal bundling support (L history, K prediction horizon)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
import logging

from .normalization import NormalizationHandler
from raps.config import ConfigManager


@dataclass
class TNOSequenceConfig:
    """
    Configuration for TNO-DeepM&Mnet sequence dataset.
    
    Attributes:
        history_length (int): L - Number of past timesteps for input/output history
        prediction_horizon (int): K - Number of future timesteps to predict
        stride (int): Stride for creating sequences (default: 1)
        pool_cdus (bool): Whether to pool CDU data as independent observations
        include_cdu_id (bool): Whether to include CDU ID in the output
    """
    history_length: int = 30
    prediction_horizon: int = 10
    stride: int = 1
    pool_cdus: bool = True
    include_cdu_id: bool = True


class CDUPooledSequenceDataset(Dataset):
    """
    Dataset for TNO-DeepM&Mnet training with CDU pooling.
    
    Creates sequences with:
    - u_hist: Input history (L, num_input_features_per_cdu)
    - y_hist: Output history (L, num_output_features_per_cdu)
    - y_future: Future outputs to predict (K, num_output_features_per_cdu)
    
    CDU data is pooled as independent observations, meaning each CDU
    generates its own set of sequences.
    """
    
    # CDU output variable names (consistent with existing code)
    CDU_OUTPUT_VARS = [
        'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
        'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
        'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
    ]
    
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
        self.data_path = Path(data_path)
        self.chunk_indices = chunk_indices
        self.tno_config = tno_config or TNOSequenceConfig()
        self.normalize = normalize
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.cdu_lim = cdu_lim if cdu_lim is not None else 50

        # Get system configuration
        if config is None:
            self.config = ConfigManager(system_name=system_name).get_config()
        else:
            self.config = config
        self.num_cdus = min(self.config['NUM_CDUS'], self.cdu_lim)
        
        # Load data
        self.raw_data = self._load_chunks()
        
        # Get column names (per-CDU structure)
        self.input_cols_per_cdu = self._get_input_columns_per_cdu()
        self.output_cols_per_cdu = self._get_output_columns_per_cdu()
        self.global_input_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        
        # Validate columns exist
        self._validate_columns()
        
        # Setup normalization using existing NormalizationHandler
        if norm_handler is None:
            self.norm_handler = NormalizationHandler()
            all_input_cols = self._get_all_input_columns_flat()
            all_output_cols = self._get_all_output_columns_flat()
            self.norm_handler.compute_stats(
                self.raw_data, all_input_cols, all_output_cols
            )
        else:
            self.norm_handler = norm_handler
        
        # Pre-extract and normalize CDU data for efficiency
        self.cdu_data = self._extract_cdu_data()
        
        # Create sequence indices
        self.sequences = self._create_sequence_indices()
        
        if self.verbose:
            self._print_summary()
    
    def _print_summary(self):
        """Print dataset summary."""
        print(f"Created CDUPooledSequenceDataset:")
        print(f"  - Total sequences: {len(self.sequences)}")
        print(f"  - History length (L): {self.tno_config.history_length}")
        print(f"  - Prediction horizon (K): {self.tno_config.prediction_horizon}")
        print(f"  - CDUs pooled: {self.num_cdus}")
        print(f"  - Input features per CDU: {self.num_input_features}")
        print(f"  - Output features per CDU: {self.num_output_features}")
        print(f"  - Raw data rows: {len(self.raw_data)}")
    
    def _load_chunks(self) -> pd.DataFrame:
        """Load specified chunks and concatenate."""
        dfs = []
        for idx in self.chunk_indices:
            chunk_dir = self.data_path / f"chunk_{idx}"
            if not chunk_dir.exists():
                if self.verbose:
                    print(f"Warning: Chunk directory missing: {chunk_dir}")
                continue

            # Try to find output parquet files
            candidate_files = sorted(chunk_dir.glob("*_output_*.parquet"))
            if not candidate_files:
                candidate_files = sorted(chunk_dir.glob("*.parquet"))

            if not candidate_files:
                if self.verbose:
                    print(f"Warning: No parquet files found in {chunk_dir}")
                continue

            for parquet_file in candidate_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    df['_chunk_idx'] = idx
                    dfs.append(df)
                    if self.verbose:
                        print(f"Loaded chunk {idx} ({parquet_file.name}): {len(df)} samples")
                except Exception as e:
                    print(f"Error loading {parquet_file}: {e}")

        if not dfs:
            raise ValueError(f"No valid chunks found in {self.chunk_indices} at {self.data_path}")

        return pd.concat(dfs, ignore_index=True)
    
    def _get_input_columns_per_cdu(self) -> Dict[int, List[str]]:
        """Get input column names organized by CDU: {cdu_id: [Q_flow_col, T_Air_col]}"""
        cols_per_cdu = {}
        for i in range(1, self.num_cdus + 1):
            cols_per_cdu[i] = [
                f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total',
                f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air'
            ]
        return cols_per_cdu
    
    def _get_output_columns_per_cdu(self) -> Dict[int, List[str]]:
        """Get output column names organized by CDU."""
        cols_per_cdu = {}
        for i in range(1, self.num_cdus + 1):
            cdu_cols = []
            for var in self.CDU_OUTPUT_VARS:
                cdu_cols.append(f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{var}')
            # Add HTC
            cdu_cols.append(f'simulator[1].datacenter[1].computeBlock[{i}].cabinet[1].summary.htc')
            cols_per_cdu[i] = cdu_cols
        return cols_per_cdu
    
    def _get_all_input_columns_flat(self) -> List[str]:
        """Get flat list of all input columns for normalization."""
        cols = []
        for cdu_cols in self.input_cols_per_cdu.values():
            cols.extend(cdu_cols)
        cols.append(self.global_input_col)
        return [c for c in cols if c in self.raw_data.columns]
    
    def _get_all_output_columns_flat(self) -> List[str]:
        """Get flat list of all output columns for normalization."""
        cols = []
        for cdu_cols in self.output_cols_per_cdu.values():
            cols.extend(cdu_cols)
        # Add datacenter-level outputs
        dc_flow = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if dc_flow in self.raw_data.columns:
            cols.append(dc_flow)
        if 'pue' in self.raw_data.columns:
            cols.append('pue')
        return [c for c in cols if c in self.raw_data.columns]
    
    def _validate_columns(self):
        """Validate that required columns exist in data."""
        available = set(self.raw_data.columns)
        
        if self.global_input_col not in available:
            raise ValueError(f"Required global input column not found: {self.global_input_col}")
        
        # Check for at least some CDU columns
        found_cdus = 0
        for cdu_id, cols in self.input_cols_per_cdu.items():
            if all(c in available for c in cols):
                found_cdus += 1
        
        if found_cdus == 0:
            raise ValueError("No complete CDU input columns found in data")
        
        if found_cdus < self.num_cdus:
            warnings.warn(f"Only {found_cdus}/{self.num_cdus} CDUs have complete input data")
    
    def _extract_cdu_data(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Extract and optionally normalize data for each CDU.
        
        Returns dict with structure:
        {
            cdu_id: {
                'inputs': np.ndarray (T, 3) - [Q_flow, T_Air, T_ext]
                'outputs': np.ndarray (T, num_outputs)
                'inputs_raw': np.ndarray (T, 3) - unnormalized for physics
                'outputs_raw': np.ndarray (T, num_outputs) - unnormalized
            }
        }
        """
        T_ext = self.raw_data[self.global_input_col].values.astype(np.float32)
        cdu_data = {}
        
        for cdu_id in range(1, self.num_cdus + 1):
            input_cols = self.input_cols_per_cdu[cdu_id]
            output_cols = self.output_cols_per_cdu[cdu_id]
            
            # Extract inputs: [Q_flow, T_Air, T_ext]
            Q_flow = self._safe_get_column(input_cols[0])
            T_Air = self._safe_get_column(input_cols[1])
            inputs_raw = np.column_stack([Q_flow, T_Air, T_ext]).astype(np.float32)
            
            # Extract outputs
            outputs_list = [self._safe_get_column(col) for col in output_cols]
            outputs_raw = np.column_stack(outputs_list).astype(np.float32)
            
            # Normalize if needed
            if self.normalize:
                inputs_norm = self._normalize_cdu_inputs(inputs_raw, cdu_id)
                outputs_norm = self._normalize_cdu_outputs(outputs_raw, cdu_id)
            else:
                inputs_norm = inputs_raw.copy()
                outputs_norm = outputs_raw.copy()
            
            cdu_data[cdu_id] = {
                'inputs': inputs_norm,
                'outputs': outputs_norm,
                'inputs_raw': inputs_raw,
                'outputs_raw': outputs_raw
            }
        
        return cdu_data
    
    def _safe_get_column(self, col: str) -> np.ndarray:
        """Safely get column data, returning zeros if not found."""
        if col in self.raw_data.columns:
            return self.raw_data[col].values.astype(np.float32)
        return np.zeros(len(self.raw_data), dtype=np.float32)
    
    def _normalize_cdu_inputs(self, inputs: np.ndarray, cdu_id: int) -> np.ndarray:
        """Normalize inputs for a specific CDU using stored stats."""
        all_input_cols = self._get_all_input_columns_flat()
        cdu_input_cols = self.input_cols_per_cdu[cdu_id] + [self.global_input_col]
        
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
        
        for i, col in enumerate(cdu_input_cols):
            if col in all_input_cols:
                idx = all_input_cols.index(col)
                if self.norm_handler.mean_in is not None and idx < len(self.norm_handler.mean_in):
                    mean[i] = self.norm_handler.mean_in[idx]
                    std[i] = self.norm_handler.std_in[idx]
        
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return ((inputs - mean) / std).astype(np.float32)
    
    def _normalize_cdu_outputs(self, outputs: np.ndarray, cdu_id: int) -> np.ndarray:
        """Normalize outputs for a specific CDU using stored stats."""
        all_output_cols = self._get_all_output_columns_flat()
        cdu_output_cols = self.output_cols_per_cdu[cdu_id]
        
        n_out = outputs.shape[1]
        mean = np.zeros(n_out, dtype=np.float32)
        std = np.ones(n_out, dtype=np.float32)
        
        for i, col in enumerate(cdu_output_cols):
            if col in all_output_cols:
                idx = all_output_cols.index(col)
                if self.norm_handler.mean_out is not None and idx < len(self.norm_handler.mean_out):
                    mean[i] = self.norm_handler.mean_out[idx]
                    std[i] = self.norm_handler.std_out[idx]
        
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return ((outputs - mean) / std).astype(np.float32)
    
    def _create_sequence_indices(self) -> List[Dict]:
        """Create list of (cdu_id, timestep) pairs for all valid sequences."""
        L = self.tno_config.history_length
        K = self.tno_config.prediction_horizon
        stride = self.tno_config.stride
        T = len(self.raw_data)
        
        sequences = []
        
        for cdu_id in range(1, self.num_cdus + 1):
            # Valid range: need L history and K future steps
            for t in range(L - 1, T - K, stride):
                sequences.append({
                    'cdu_id': cdu_id,
                    'timestep': t,
                    'hist_start': t - L + 1,
                    'hist_end': t + 1,
                    'future_start': t + 1,
                    'future_end': t + 1 + K
                })
        
        return sequences
    
    @property
    def num_input_features(self) -> int:
        """Number of input features per CDU: [Q_flow, T_Air, T_ext]"""
        return 3
    
    @property
    def num_output_features(self) -> int:
        """Number of output features per CDU (11 CDU vars + 1 HTC = 12)"""
        return len(self.CDU_OUTPUT_VARS) + 1  # +1 for HTC
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        
        Returns dict with:
            - u_hist: (L, 3) input history [Q_flow, T_Air, T_ext]
            - y_hist: (L, 12) output history
            - y_future: (K, 12) future targets
            - Q_flow_raw: (L,) unnormalized Q_flow for physics losses
            - cdu_id: CDU identifier
            - timestep: timestep index
        """
        seq_info = self.sequences[idx]
        cdu_id = seq_info['cdu_id']
        cdu = self.cdu_data[cdu_id]
        
        # Extract sequences
        u_hist = cdu['inputs'][seq_info['hist_start']:seq_info['hist_end']]
        y_hist = cdu['outputs'][seq_info['hist_start']:seq_info['hist_end']]
        y_future = cdu['outputs'][seq_info['future_start']:seq_info['future_end']]
        
        # Raw Q_flow for physics losses
        Q_flow_raw = cdu['inputs_raw'][seq_info['hist_start']:seq_info['hist_end'], 0]
        
        return {
            'u_hist': torch.from_numpy(u_hist),
            'y_hist': torch.from_numpy(y_hist),
            'y_future': torch.from_numpy(y_future),
            'Q_flow_raw': torch.from_numpy(Q_flow_raw),
            'cdu_id': torch.tensor(cdu_id, dtype=torch.long),
            'timestep': torch.tensor(seq_info['timestep'], dtype=torch.long)
        }
    
    def get_denormalization_params(self, cdu_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (mean, std) for denormalizing a specific CDU's outputs."""
        all_output_cols = self._get_all_output_columns_flat()
        cdu_output_cols = self.output_cols_per_cdu[cdu_id]
        
        n_out = len(cdu_output_cols)
        mean = np.zeros(n_out, dtype=np.float32)
        std = np.ones(n_out, dtype=np.float32)
        
        for i, col in enumerate(cdu_output_cols):
            if col in all_output_cols:
                idx = all_output_cols.index(col)
                if self.norm_handler.mean_out is not None and idx < len(self.norm_handler.mean_out):
                    mean[i] = self.norm_handler.mean_out[idx]
                    std[i] = self.norm_handler.std_out[idx]
        
        return mean, std


class CDUPooledSingleChunkDataset(CDUPooledSequenceDataset):
    """CDU Pooled Sequence Dataset for single chunk with temporal splitting."""
    
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
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        
        super().__init__(
            data_path=data_path,
            chunk_indices=[chunk_idx],
            tno_config=tno_config,
            norm_handler=norm_handler,
            normalize=normalize,
            config=config,
            cdu_lim=cdu_lim,
            system_name=system_name,
            verbose=verbose
        )
    
    def _create_sequence_indices(self) -> List[Dict]:
        """Create sequence indices with temporal split applied."""
        L = self.tno_config.history_length
        K = self.tno_config.prediction_horizon
        stride = self.tno_config.stride
        T = len(self.raw_data)
        
        valid_start = L - 1
        valid_end = T - K
        total_valid = valid_end - valid_start
        
        split_start = valid_start + int(total_valid * self.start_ratio)
        split_end = valid_start + int(total_valid * self.end_ratio)
        
        sequences = []
        for cdu_id in range(1, self.num_cdus + 1):
            for t in range(split_start, split_end, stride):
                sequences.append({
                    'cdu_id': cdu_id,
                    'timestep': t,
                    'hist_start': t - L + 1,
                    'hist_end': t + 1,
                    'future_start': t + 1,
                    'future_end': t + 1 + K
                })
        
        return sequences


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


def create_tno_data_loaders(
    data_path: str,
    tno_config: TNOSequenceConfig,
    batch_size: int = 128,
    train_chunks: Optional[List[int]] = None,
    val_chunks: Optional[List[int]] = None,
    test_chunks: Optional[List[int]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    system_name: str = 'summit',
    cdu_lim: Optional[int] = 50,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, NormalizationHandler]:
    """
    Create train, validation, and test data loaders for TNO training.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, norm_handler)
    """
    data_path = Path(data_path)
    
    # Discover available chunks
    available_chunks = []
    for chunk_dir in sorted(data_path.glob("chunk_*")):
        try:
            chunk_idx = int(chunk_dir.name.split("_")[1])
            parquet_files = list(chunk_dir.glob("*_output_*.parquet")) or list(chunk_dir.glob("*.parquet"))
            if parquet_files:
                available_chunks.append(chunk_idx)
        except (ValueError, IndexError):
            continue
    
    if not available_chunks:
        raise ValueError(f"No valid chunks found in {data_path}")
    
    if verbose:
        print(f"Found {len(available_chunks)} available chunks: {available_chunks}")
    
    # Determine split strategy
    explicit_chunks = any(x is not None for x in [train_chunks, val_chunks, test_chunks])
    
    if explicit_chunks:
        train_chunks = [c for c in (train_chunks or []) if c in available_chunks]
        val_chunks = [c for c in (val_chunks or []) if c in available_chunks]
        test_chunks = [c for c in (test_chunks or []) if c in available_chunks]
    else:
        n_chunks = len(available_chunks)
        if n_chunks == 1:
            train_chunks = val_chunks = test_chunks = available_chunks
        else:
            n_test = max(1, int(n_chunks * (1 - train_ratio - val_ratio)))
            n_val = max(1, int(n_chunks * val_ratio))
            n_train = n_chunks - n_test - n_val
            train_chunks = available_chunks[:n_train]
            val_chunks = available_chunks[n_train:n_train + n_val]
            test_chunks = available_chunks[n_train + n_val:]
    
    if verbose:
        print(f"Train chunks: {train_chunks}, Val chunks: {val_chunks}, Test chunks: {test_chunks}")
    
    # Check if temporal split needed
    use_temporal_split = (len(available_chunks) == 1 or 
                          set(train_chunks) == set(val_chunks) == set(test_chunks))
    
    if use_temporal_split:
        chunk_idx = train_chunks[0] if train_chunks else available_chunks[0]
        
        if verbose:
            print(f"Using temporal split within chunk {chunk_idx}")
        
        train_dataset = CDUPooledSingleChunkDataset(
            data_path=str(data_path), chunk_idx=chunk_idx, tno_config=tno_config,
            start_ratio=0.0, end_ratio=train_ratio, normalize=True,
            config=config, cdu_lim=cdu_lim, system_name=system_name, verbose=verbose
        )
        
        val_dataset = CDUPooledSingleChunkDataset(
            data_path=str(data_path), chunk_idx=chunk_idx, tno_config=tno_config,
            start_ratio=train_ratio, end_ratio=train_ratio + val_ratio,
            norm_handler=train_dataset.norm_handler, normalize=True,
            config=config, cdu_lim=cdu_lim, system_name=system_name, verbose=verbose
        )
        
        test_dataset = CDUPooledSingleChunkDataset(
            data_path=str(data_path), chunk_idx=chunk_idx, tno_config=tno_config,
            start_ratio=train_ratio + val_ratio, end_ratio=1.0,
            norm_handler=train_dataset.norm_handler, normalize=True,
            config=config, cdu_lim=cdu_lim, system_name=system_name, verbose=verbose
        )
    else:
        train_dataset = CDUPooledSequenceDataset(
            data_path=str(data_path), chunk_indices=train_chunks, tno_config=tno_config,
            normalize=True, config=config, cdu_lim=cdu_lim,  system_name=system_name, verbose=verbose
        )
        
        val_dataset = CDUPooledSequenceDataset(
            data_path=str(data_path), chunk_indices=val_chunks, tno_config=tno_config,
            norm_handler=train_dataset.norm_handler, normalize=True,
            config=config, cdu_lim=cdu_lim, system_name=system_name, verbose=verbose
        )
        
        test_dataset = CDUPooledSequenceDataset(
            data_path=str(data_path), chunk_indices=test_chunks, tno_config=tno_config,
            norm_handler=train_dataset.norm_handler, normalize=True,
            config=config, cdu_lim=cdu_lim, system_name=system_name, verbose=verbose
        )
    
    # Create samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, collate_fn=tno_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=tno_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=tno_collate_fn
    )
    
    if verbose:
        print(f"\nDataloader Summary:")
        print(f"  Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} sequences, {len(val_loader)} batches")
        print(f"  Test: {len(test_dataset)} sequences, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset.norm_handler


def get_sample_batch(loader: DataLoader) -> Dict[str, torch.Tensor]:
    """Get a single batch from dataloader for inspection."""
    return next(iter(loader))


def print_batch_shapes(batch: Dict[str, torch.Tensor]) -> None:
    """Print shapes of all tensors in a batch."""
    print("Batch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape} (dtype: {tensor.dtype})")


def inspect_dataset(dataset: CDUPooledSequenceDataset) -> Dict:
    """Get summary statistics about a dataset."""
    return {
        'num_sequences': len(dataset),
        'num_cdus': min(dataset.num_cdus, dataset.cdu_lim),
        'history_length': dataset.tno_config.history_length,
        'prediction_horizon': dataset.tno_config.prediction_horizon,
        'num_input_features': dataset.num_input_features,
        'num_output_features': dataset.num_output_features,
        'chunk_indices': dataset.chunk_indices,
        'raw_data_rows': len(dataset.raw_data)
    }