"""I/O utilities for data loading and saving."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
    
    

logger = logging.getLogger(__name__)


def save_data(data: Any, filepath: Union[str, Path], format: str = 'auto') -> None:
    """
    Save data to file in various formats.
    
    Args:
        data: Data to save
        filepath: Path to save the data
        format: Format to save ('auto', 'npy', 'npz', 'pickle', 'json', 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove the dot
    
    if format in ['npy', 'numpy']:
        np.save(filepath, data)
    elif format == 'npz':
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            np.savez(filepath, data=data)
    elif format in ['pkl', 'pickle']:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(filepath, index=False)
        else:
            raise ValueError(f"Cannot save {type(data)} as CSV")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to {filepath}")


def load_data(filepath: Union[str, Path], format: str = 'auto') -> Any:
    """
    Load data from file in various formats.
    
    Args:
        filepath: Path to load the data from
        format: Format to load ('auto', 'npy', 'npz', 'pickle', 'json', 'csv')
    
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove the dot
    
    if format in ['npy', 'numpy']:
        return np.load(filepath, allow_pickle=True)
    elif format == 'npz':
        return np.load(filepath, allow_pickle=True)
    elif format in ['pkl', 'pickle']:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save a trained model.
    
    Args:
        model: Model to save
        filepath: Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    import torch
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(model: Any, filepath: Union[str, Path]) -> Any:
    """
    Load a trained model.
    
    Args:
        model: Model instance to load weights into
        filepath: Path to load the model from
    
    Returns:
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    import torch
    if isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        return model
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_config(config: Dict, filepath: Union[str, Path]) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to load the config from
    
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(data: Any, filepath: str, format: str = 'auto'):
    """
    Save results to file
    
    Parameters:
    -----------
    data : Any
        Data to save
    filepath : str
        Output file path
    format : str
        Format ('json', 'pickle', 'csv', 'parquet', 'auto')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove dot
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'pickle' or format == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    elif format == 'parquet':
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath)
        else:
            pd.DataFrame(data).to_parquet(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"✓ Saved to {filepath}")


def load_results(filepath: str, format: str = 'auto') -> Any:
    """
    Load results from file
    
    Parameters:
    -----------
    filepath : str
        Input file path
    format : str
        Format ('json', 'pickle', 'csv', 'parquet', 'auto')
    
    Returns:
    --------
    Any : Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'auto':
        format = filepath.suffix[1:]
    
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'pickle' or format == 'pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
