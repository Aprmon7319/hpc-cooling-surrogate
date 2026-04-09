"""
FMU2ML: Physics-informed Machine Learning for Datacenter Cooling Systems

A modular package for datacenter cooling simulation and machine learning across
different HPC systems (Summit, Marconi100, Frontier, Fugaku, Lassen).

Features:
- System-agnostic configuration with RAPS integration
- Data generation (power scenarios, temperature modeling)
- Neural operator models (FNO, Hybrid-FNO, DeepONet)
- Physics-informed training losses
- FMU simulation integration
- Model evaluation and comparison
"""


__author__ = "HPCI Lab"

# Configuration
from fmu2ml.config import SystemConfig, ModelConfig, TrainingConfig, get_system_config

# Data
from fmu2ml.data.generators import PowerGenerator, TemperatureGenerator, ScenarioGenerator
from fmu2ml.data.processors import NormalizationHandler, create_data_loaders

# Models
from fmu2ml.models import ModelRegistry, create_model, FNOCooling, HybridFNOCooling, DeepONetCooling

# Training
from fmu2ml.training import Trainer

# Inference
from fmu2ml.inference import CoolingModelPredictor, BatchProcessor

# Evaluation
from fmu2ml.evaluation import calculate_metrics, MetricsCalculator, PhysicsValidator, ModelComparator

# Utils
from fmu2ml.utils import save_results, load_results, setup_logger

__all__ = [
    # Config
    'SystemConfig',
    'ModelConfig',
    'TrainingConfig',
    'get_system_config',

    # Data
    'PowerGenerator',
    'TemperatureGenerator',
    'ScenarioGenerator',
    'NormalizationHandler',
    'create_data_loaders',

    # Models
    'ModelRegistry',
    'create_model',
    'FNOCooling',
    'HybridFNOCooling',
    'DeepONetCooling',
    
    # Training
    'Trainer',
    
    # Inference
    'CoolingModelPredictor',
    'BatchProcessor',
    
    # Evaluation
    'calculate_metrics',
    'MetricsCalculator',
    'PhysicsValidator',
    'ModelComparator',
    
    # Utils
    'save_results',
    'load_results',
    'setup_logger',
]
