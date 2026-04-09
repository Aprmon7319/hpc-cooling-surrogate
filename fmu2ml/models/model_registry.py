"""
Model registry for creating model instances
"""

from typing import Dict, Type, Optional
import torch.nn as nn

from fmu2ml.config import ModelConfig, FNOConfig, HybridFNOConfig, DeepONetConfig
from fmu2ml.models.base_model import BaseModel
from fmu2ml.models.fno import FNOCooling
from fmu2ml.models.hybrid_fno import HybridFNOCooling
from fmu2ml.models.deeponet import DeepONetCooling


class ModelRegistry:
    """
    Registry for creating model instances by name
    """
    
    _models: Dict[str, Type[BaseModel]] = {
        'fno': FNOCooling,
        'hybrid_fno': HybridFNOCooling,
        'deeponet': DeepONetCooling,
    }
    
    # Map model names to their config classes
    _config_classes: Dict[str, Type[ModelConfig]] = {
        'fno': FNOConfig,
        'hybrid_fno': HybridFNOConfig,
        'deeponet': DeepONetConfig,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel], 
                      config_class: Type[ModelConfig] = ModelConfig):
        """
        Register a new model class
        
        Parameters:
        -----------
        name : str
            Model name
        model_class : Type[BaseModel]
            Model class
        config_class : Type[ModelConfig]
            Config class for the model
        """
        cls._models[name] = model_class
        cls._config_classes[name] = config_class
    
    @classmethod
    def _convert_config(cls, name: str, config: ModelConfig) -> ModelConfig:
        """
        Convert base ModelConfig to model-specific config if needed
        
        Parameters:
        -----------
        name : str
            Model name
        config : ModelConfig
            Input configuration
        
        Returns:
        --------
        ModelConfig
            Model-specific configuration
        """
        target_config_class = cls._config_classes.get(name, ModelConfig)
        
        # If already the correct type, return as is
        if isinstance(config, target_config_class):
            return config
        
        # Convert base ModelConfig to specific config
        config_dict = {
            'num_cdus': config.num_cdus,
            'input_dim': config.input_dim,
            'output_dim_per_cdu': config.output_dim_per_cdu,
            'sequence_length': config.sequence_length,
        }
        
        # Add model-specific defaults
        if name == 'fno':
            config_dict.update({
                'fno_modes': getattr(config, 'fno_modes', 16),
                'fno_width': getattr(config, 'fno_width', 64),
                'fno_layers': getattr(config, 'fno_layers', 4),
            })
        elif name == 'hybrid_fno':
            config_dict.update({
                'hidden_dim': getattr(config, 'hidden_dim', 128),
                'num_gru_layers': getattr(config, 'num_gru_layers', 2),
                'fno_modes': getattr(config, 'fno_modes', 3),
                'fno_width': getattr(config, 'fno_width', 32),
                'fno_layers': getattr(config, 'fno_layers', 2),
            })
        elif name == 'deeponet':
            config_dict.update({
                'deeponet_branch_layers': getattr(config, 'deeponet_branch_layers', None),
                'deeponet_trunk_layers': getattr(config, 'deeponet_trunk_layers', None),
                'deeponet_basis_dim': getattr(config, 'deeponet_basis_dim', 100),
            })
        
        return target_config_class(**config_dict)
    
    @classmethod
    def create_model(cls, name: str, config: ModelConfig) -> BaseModel:
        """
        Create a model instance
        
        Parameters:
        -----------
        name : str
            Model name
        config : ModelConfig
            Model configuration
        
        Returns:
        --------
        BaseModel
            Model instance
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        
        # Convert config to appropriate type
        specific_config = cls._convert_config(name, config)
        
        model_class = cls._models[name]
        return model_class(specific_config)
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all registered models
        
        Returns:
        --------
        list
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """
        Get model class by name
        
        Parameters:
        -----------
        name : str
            Model name
        
        Returns:
        --------
        Type[BaseModel]
            Model class
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        
        return cls._models[name]


# Convenience function
def create_model(name: str, config: ModelConfig) -> BaseModel:
    """
    Create a model instance
    
    Parameters:
    -----------
    name : str
        Model name ('fno', 'deeponet', 'hybrid_fno')
    config : ModelConfig
        Model configuration (will be converted to specific type if needed)
    
    Returns:
    --------
    BaseModel
        Model instance
    """
    return ModelRegistry.create_model(name, config)