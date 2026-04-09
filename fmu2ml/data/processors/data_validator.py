import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"      # Critical issues that prevent processing
    WARNING = "warning"  # Issues that should be reviewed
    INFO = "info"        # Informational messages


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    message: str
    column: Optional[str] = None
    count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class DataValidator:
    """
    Comprehensive data validator for FMU2ML datacenter cooling pipeline
    
    Validates:
    - Missing values
    - Data types
    - Value ranges (physical constraints)
    - Outliers
    - Temporal consistency
    - Data quality metrics
    - Physical relationships (e.g., temperature differentials)
    """
    
    def __init__(self, config: Optional[Any] = None, strict: bool = False):
        """
        Initialize data validator
        
        Args:
            config: System configuration with physical constraints
            strict: If True, warnings are treated as errors
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.strict = strict
        
        # Number of CDUs
        self.num_cdus = config['NUM_CDUS']
        
        # Default physical constraints (updated for datacenter cooling)
        self.constraints = self._get_default_constraints()
        
        # Override with config if provided
        if config is not None:
            self._update_constraints_from_config(config)
        
        self.logger.info("Data validator initialized for datacenter cooling system")
    
    def _get_default_constraints(self) -> Dict[str, Dict[str, float]]:
        """Get default physical constraints for datacenter cooling"""
        return {
            # Power constraints (Watts)
            'power': {
                'min': 0.0,
                'max': np.ceil(self.config['MAX_POWER'])*1000 if self.config and 'MAX_POWER' in self.config else 50000.0,  # 50 kW per CDU
                'idle_min': 10000.0,
                'idle_max': np.floor(self.config['MIN_POWER'])*1000 if self.config and 'MIN_POWER' in self.config else 15000.0
            },
            # Temperature constraints (Celsius for outputs, Kelvin for inputs)
            'temperature_kelvin': {
                'min': 263.15,  # -10°C
                'max': 373.15,  # 100°C
                'ambient_min': 263.15,  # -10°C
                'ambient_max': 323.15,  # 50°C
                'air_min': 288.15,  # 15°C
                'air_max': 303.15   # 30°C
            },
            'temperature_celsius': {
                'min': -10.0,
                'max': 100.0,
                'supply_min': 5.0,
                'supply_max': 25.0,
                'return_min': 15.0,
                'return_max': 40.0
            },
            # Flow rate constraints (GPM)
            'flow_gpm': {
                'min': 0.0,
                'max': 500.0,  # Per CDU
                'datacenter_min': 100.0,
                'datacenter_max': 25000.0  # Sum of all CDUs
            },
            # Pump power constraints (kW)
            'pump_power_kw': {
                'min': 0.0,
                'max': 100.0  # Per CDU pump
            },
            # Pressure constraints (psig)
            'pressure_psig': {
                'min': -5.0,
                'max': 150.0,
                'typical_min': 10.0,
                'typical_max': 100.0
            },
            # PUE constraints
            'pue': {
                'min': 1.0,
                'max': 3.0,
                'ideal_min': 1.0,
                'ideal_max': 1.5
            },
            # Heat Transfer Coefficient (W/(m²·K))
            'htc': {
                'min': 10.0,
                'max': 10000.0,
                'typical_min': 100.0,
                'typical_max': 5000.0
            },
            # Temperature differentials (°C)
            'delta_T': {
                'min': 0.0,
                'max': 30.0,
                'typical_min': 2.0,
                'typical_max': 15.0
            }
        }
    
    def _update_constraints_from_config(self, config: Any) -> None:
        """Update constraints from system configuration"""
        if hasattr(config, 'MIN_POWER'):
            self.constraints['power']['min'] = config.MIN_POWER * 1000  # kW to W
        if hasattr(config, 'MAX_POWER'):
            self.constraints['power']['max'] = config.MAX_POWER * 1000
    
    def validate(
        self,
        data: pd.DataFrame,
        data_type: str = 'input'
    ) -> Tuple[List[ValidationResult], Optional[pd.DataFrame]]:
        """
        Validate data comprehensively
        
        Args:
            data: DataFrame to validate
            data_type: 'input' or 'output'
            return_clean: If True, return cleaned data
            
        Returns:
            Tuple of (validation_results, clean_data or None)
        """
        self.logger.info(f"Validating {data_type} data: {len(data)} samples, {len(data.columns)} columns")
        
        results = []
        
        # 1. Basic structure validation
        results.extend(self._validate_structure(data, data_type))
        
        # 2. Missing values
        results.extend(self._validate_missing_values(data))
        
        # 3. Data types
        results.extend(self._validate_data_types(data))
        
        # 4. Value ranges
        results.extend(self._validate_value_ranges(data, data_type))
        
        # 5. Physical constraints
        results.extend(self._validate_physical_constraints(data, data_type))
        
        # 6. Outliers
        results.extend(self._validate_outliers(data))
        
        # 7. Temporal consistency
        if 'time' in data.columns:
            results.extend(self._validate_temporal_consistency(data))
        
        # 8. Data quality metrics
        results.extend(self._validate_data_quality(data))
        
        # Log summary
        self._log_validation_summary(results)
        
        # Raise error if strict mode and errors found
        if self.strict:
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            if errors:
                raise ValueError(f"Validation failed with {len(errors)} errors")
        
        return results
    
    def _validate_structure(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> List[ValidationResult]:
        """Validate DataFrame structure for datacenter cooling system"""
        results = []
        
        # Check if DataFrame is empty
        if len(data) == 0:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="DataFrame is empty"
            ))
            return results
        
        # Get expected columns
        if data_type == 'input':
            required_cols = self._get_required_input_columns()
        else:
            required_cols = self._get_required_output_columns()
        
        # Check for missing required columns
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Missing {len(missing_cols)} required columns",
                count=len(missing_cols),
                details={'missing_columns': list(missing_cols)[:10]}  # Show first 10
            ))
        
        # Check for CDU coverage
        if data_type == 'input':
            # Check Q_flow_total columns
            power_cols = [col for col in data.columns 
                         if 'Q_flow_total' in col and 'computeBlock' in col]
            expected_power_cols = self.num_cdus
            if len(power_cols) < expected_power_cols:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Incomplete CDU power data: {len(power_cols)}/{expected_power_cols} CDUs",
                    count=expected_power_cols - len(power_cols)
                ))
            
            # Check T_Air columns
            temp_cols = [col for col in data.columns 
                        if 'T_Air' in col and 'computeBlock' in col]
            if len(temp_cols) < expected_power_cols:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Incomplete CDU temperature data: {len(temp_cols)}/{expected_power_cols} CDUs",
                    count=expected_power_cols - len(temp_cols)
                ))
        
        elif data_type == 'output':
            # Check CDU output parameters
            cdu_params = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                         'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                         'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
            
            for param in cdu_params:
                param_cols = [col for col in data.columns if param in col and 'computeBlock' in col]
                if len(param_cols) < self.num_cdus:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Incomplete {param} data: {len(param_cols)}/{self.num_cdus} CDUs",
                        details={'parameter': param}
                    ))
        
        return results
    
    def _validate_missing_values(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate missing values"""
        results = []
        
        missing_counts = data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                pct = (count / len(data)) * 100
                
                # More strict for critical columns
                if 'T_ext' in col or 'Q_flow_total' in col:
                    level = ValidationLevel.ERROR if pct > 1 else ValidationLevel.WARNING
                else:
                    level = ValidationLevel.ERROR if pct > 10 else ValidationLevel.WARNING
                
                results.append(ValidationResult(
                    level=level,
                    message=f"Column has {pct:.2f}% missing values",
                    column=col,
                    count=int(count),
                    details={'percentage': pct}
                ))
        
        return results
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data types"""
        results = []
        
        for col in data.columns:
            if col == 'time':
                continue
            
            # All columns except time should be numeric
            if not pd.api.types.is_numeric_dtype(data[col]):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Column is not numeric: {data[col].dtype}",
                    column=col
                ))
        
        return results
    
    def _validate_value_ranges(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> List[ValidationResult]:
        """Validate value ranges based on physical constraints"""
        results = []
        
        for col in data.columns:
            if col == 'time':
                continue
            
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Get constraints for this column
            constraints = self._get_column_constraints(col, data_type)
            if constraints is None:
                continue
            
            min_val, max_val = constraints
            
            # Check minimum
            below_min = data[col] < min_val
            if below_min.any():
                count = below_min.sum()
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Values below minimum ({min_val:.2f})",
                    column=col,
                    count=int(count),
                    details={
                        'min_value': float(data[col].min()),
                        'threshold': min_val
                    }
                ))
            
            # Check maximum
            above_max = data[col] > max_val
            if above_max.any():
                count = above_max.sum()
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Values above maximum ({max_val:.2f})",
                    column=col,
                    count=int(count),
                    details={
                        'max_value': float(data[col].max()),
                        'threshold': max_val
                    }
                ))
        
        return results
    
    def _validate_physical_constraints(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> List[ValidationResult]:
        """Validate physical constraints specific to datacenter cooling"""
        results = []
        
        if data_type == 'output':
            # 1. Validate temperature differentials for each CDU
            for cdu_id in range(1, self.num_cdus + 1):
                # Primary loop: T_prim_r should > T_prim_s
                t_prim_s_col = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.T_prim_s_C'
                t_prim_r_col = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.T_prim_r_C'
                
                if t_prim_s_col in data.columns and t_prim_r_col in data.columns:
                    delta_t_prim = data[t_prim_r_col] - data[t_prim_s_col]
                    
                    # Check for negative delta T
                    negative = delta_t_prim < 0
                    if negative.any():
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"CDU {cdu_id}: Negative primary ΔT (return < supply)",
                            count=int(negative.sum()),
                            details={
                                'min_delta_T': float(delta_t_prim.min()),
                                'cdu_id': cdu_id
                            }
                        ))
                    
                    # Check for excessive delta T
                    max_delta = self.constraints['delta_T']['max']
                    excessive = delta_t_prim > max_delta
                    if excessive.any():
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"CDU {cdu_id}: Excessive primary ΔT (>{max_delta}°C)",
                            count=int(excessive.sum()),
                            details={
                                'max_delta_T': float(delta_t_prim.max()),
                                'cdu_id': cdu_id
                            }
                        ))
                
                # Secondary loop: T_sec_r should > T_sec_s
                t_sec_s_col = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.T_sec_s_C'
                t_sec_r_col = f'simulator[1].datacenter[1].computeBlock[{cdu_id}].cdu[1].summary.T_sec_r_C'
                
                if t_sec_s_col in data.columns and t_sec_r_col in data.columns:
                    delta_t_sec = data[t_sec_r_col] - data[t_sec_s_col]
                    
                    negative = delta_t_sec < 0
                    if negative.any():
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"CDU {cdu_id}: Negative secondary ΔT (return < supply)",
                            count=int(negative.sum()),
                            details={
                                'min_delta_T': float(delta_t_sec.min()),
                                'cdu_id': cdu_id
                            }
                        ))
            
            # 2. Validate PUE
            if 'pue' in data.columns:
                pue_min = self.constraints['pue']['min']
                pue_max = self.constraints['pue']['max']
                
                below_min = data['pue'] < pue_min
                if below_min.any():
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"PUE below physical minimum ({pue_min})",
                        column='pue',
                        count=int(below_min.sum()),
                        details={'min_pue': float(data['pue'].min())}
                    ))
                
                above_max = data['pue'] > pue_max
                if above_max.any():
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"PUE above expected maximum ({pue_max})",
                        column='pue',
                        count=int(above_max.sum()),
                        details={'max_pue': float(data['pue'].max())}
                    ))
            
            # 3. Validate datacenter total flow vs CDU flows
            dc_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
            if dc_flow_col in data.columns:
                # Sum individual CDU flows
                cdu_flow_cols = [f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.V_flow_prim_GPM'
                               for i in range(1, self.num_cdus + 1)
                               if f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.V_flow_prim_GPM' in data.columns]
                
                if cdu_flow_cols:
                    cdu_flow_sum = data[cdu_flow_cols].sum(axis=1)
                    dc_flow = data[dc_flow_col]
                    
                    # Check for large discrepancies (>5%)
                    rel_error = np.abs(cdu_flow_sum - dc_flow) / (dc_flow + 1e-8)
                    large_error = rel_error > 0.05
                    
                    if large_error.any():
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message="Large discrepancy between datacenter total flow and sum of CDU flows",
                            count=int(large_error.sum()),
                            details={
                                'max_relative_error': float(rel_error.max()),
                                'mean_relative_error': float(rel_error.mean())
                            }
                        ))
        
        return results
    
    def _validate_outliers(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate outliers using IQR method"""
        results = []
        
        for col in data.columns:
            if col == 'time':
                continue
            
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                pct = (outlier_count / len(data)) * 100
                
                # Be more lenient for outputs (cooling system can have variability)
                level = ValidationLevel.WARNING if pct < 5 else ValidationLevel.ERROR
                
                results.append(ValidationResult(
                    level=level,
                    message=f"Outliers detected ({pct:.2f}%)",
                    column=col,
                    count=int(outlier_count),
                    details={
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'percentage': pct
                    }
                ))
        
        return results
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate temporal consistency"""
        results = []
        
        time_col = 'time'
        
        # Check if time is monotonically increasing
        time_diff = data[time_col].diff()
        
        # Skip first value (NaN)
        non_positive = time_diff[1:] <= 0
        if non_positive.any():
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="Time is not monotonically increasing",
                column=time_col,
                count=int(non_positive.sum())
            ))
        
        # Check for consistent timestep
        if len(time_diff) > 1:
            expected_dt = time_diff[1:].mode()[0] if len(time_diff[1:]) > 0 else None
            if expected_dt is not None and expected_dt > 0:
                inconsistent = np.abs(time_diff[1:] - expected_dt) > 0.1
                if inconsistent.any():
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Inconsistent timestep detected (expected: {expected_dt:.1f}s)",
                        column=time_col,
                        count=int(inconsistent.sum()),
                        details={
                            'expected_dt': float(expected_dt),
                            'min_dt': float(time_diff[1:].min()),
                            'max_dt': float(time_diff[1:].max())
                        }
                    ))
        
        return results
    
    def _validate_data_quality(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Calculate and validate data quality metrics"""
        results = []
        
        # Overall completeness
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        if completeness < 95:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Low data completeness: {completeness:.2f}%",
                details={'completeness': completeness}
            ))
        else:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message=f"Data completeness: {completeness:.2f}%",
                details={'completeness': completeness}
            ))
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            pct = (duplicates / len(data)) * 100
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Duplicate rows found: {pct:.2f}%",
                count=int(duplicates),
                details={'percentage': pct}
            ))
        
        # Check for constant columns (no variation)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'time':
                continue
            
            if data[col].std() == 0:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="Column has no variation (constant values)",
                    column=col,
                    details={'value': float(data[col].iloc[0])}
                ))
        
        return results
    
    def _get_column_constraints(
        self,
        col: str,
        data_type: str
    ) -> Optional[Tuple[float, float]]:
        """Get min/max constraints for a column based on naming"""
        
        # Input columns
        if data_type == 'input':
            # Power (Q_flow_total) - in Watts
            if 'Q_flow_total' in col:
                return (self.constraints['power']['min'], 
                       self.constraints['power']['max'])
            
            # Air temperature (T_Air) - in Kelvin
            if 'T_Air' in col:
                return (self.constraints['temperature_kelvin']['air_min'],
                       self.constraints['temperature_kelvin']['air_max'])
            
            # External temperature (T_ext) - in Kelvin
            if 'T_ext' in col:
                return (self.constraints['temperature_kelvin']['ambient_min'],
                       self.constraints['temperature_kelvin']['ambient_max'])
        
        # Output columns
        elif data_type == 'output':
            # Flow rates (GPM)
            if 'V_flow_prim_GPM' in col or 'V_flow_sec_GPM' in col:
                if 'datacenter[1].summary' in col:
                    return (self.constraints['flow_gpm']['datacenter_min'],
                           self.constraints['flow_gpm']['datacenter_max'])
                else:
                    return (self.constraints['flow_gpm']['min'],
                           self.constraints['flow_gpm']['max'])
            
            # Pump power (kW)
            if 'W_flow_CDUP_kW' in col:
                return (self.constraints['pump_power_kw']['min'],
                       self.constraints['pump_power_kw']['max'])
            
            # Temperatures (Celsius)
            if any(t in col for t in ['T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C']):
                if '_s_C' in col:  # Supply
                    return (self.constraints['temperature_celsius']['supply_min'],
                           self.constraints['temperature_celsius']['supply_max'])
                else:  # Return
                    return (self.constraints['temperature_celsius']['return_min'],
                           self.constraints['temperature_celsius']['return_max'])
            
            # Pressures (psig)
            if 'psig' in col:
                return (self.constraints['pressure_psig']['min'],
                       self.constraints['pressure_psig']['max'])
            
            # PUE
            if col == 'pue':
                return (self.constraints['pue']['min'],
                       self.constraints['pue']['max'])
            
            # Heat Transfer Coefficient
            if 'htc' in col:
                return (self.constraints['htc']['min'],
                       self.constraints['htc']['max'])
        
        return None
    
    def _get_required_input_columns(self) -> List[str]:
        """Get required input columns for datacenter cooling"""
        required = []
        
        # CDU power inputs
        for i in range(1, self.num_cdus + 1):
            required.append(f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_Q_flow_total')
        
        # CDU air temperature inputs
        for i in range(1, self.num_cdus + 1):
            required.append(f'simulator_1_datacenter_1_computeBlock_{i}_cabinet_1_sources_T_Air')
        
        # External temperature
        required.append('simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext')
        
        return required
    
    def _get_required_output_columns(self) -> List[str]:
        """Get required output columns for datacenter cooling"""
        required = []
        
        # CDU outputs
        output_vars = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                      'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                      'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        
        for i in range(1, self.num_cdus + 1):
            for var in output_vars:
                required.append(f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{var}')
        
        # Datacenter-level outputs
        required.append('simulator[1].datacenter[1].summary.V_flow_prim_GPM')
        required.append('pue')
        
        # Heat transfer coefficients
        for i in range(1, self.num_cdus + 1):
            required.append(f'simulator[1].datacenter[1].computeBlock[{i}].cabinet[1].summary.htc')
        
        return required
    
    
    def _log_validation_summary(self, results: List[ValidationResult]) -> None:
        """Log validation summary"""
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        self.logger.info(f"Validation complete: {len(errors)} errors, "
                        f"{len(warnings)} warnings, {len(infos)} info")
        
        if errors:
            self.logger.error("Validation errors:")
            for result in errors[:10]:  # Show first 10
                msg = f"  - {result.message}"
                if result.column:
                    msg += f" [{result.column}]"
                if result.count:
                    msg += f" (count: {result.count})"
                self.logger.error(msg)
            
            if len(errors) > 10:
                self.logger.error(f"  ... and {len(errors) - 10} more errors")
        
        if warnings:
            self.logger.warning("Validation warnings:")
            for result in warnings[:10]:  # Show first 10
                msg = f"  - {result.message}"
                if result.column:
                    msg += f" [{result.column}]"
                if result.count:
                    msg += f" (count: {result.count})"
                self.logger.warning(msg)
            
            if len(warnings) > 10:
                self.logger.warning(f"  ... and {len(warnings) - 10} more warnings")
    
    def validate_batch(
        self,
        data_chunks: List[pd.DataFrame],
        data_type: str = 'input'
    ) -> Dict[int, List[ValidationResult]]:
        """
        Validate multiple data chunks
        
        Args:
            data_chunks: List of DataFrames to validate
            data_type: 'input' or 'output'
            
        Returns:
            Dictionary mapping chunk_id to validation results
        """
        all_results = {}
        
        for chunk_id, chunk in enumerate(data_chunks):
            self.logger.info(f"Validating chunk {chunk_id + 1}/{len(data_chunks)}")
            results, _ = self.validate(chunk, data_type, return_clean=False)
            all_results[chunk_id] = results
        
        return all_results
    
    def get_summary_report(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary report from validation results
        
        Args:
            results: List of validation results
            
        Returns:
            Dictionary with summary statistics
        """
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        return {
            'total_checks': len(results),
            'errors': len(errors),
            'warnings': len(warnings),
            'infos': len(infos),
            'error_details': [
                {
                    'message': r.message,
                    'column': r.column,
                    'count': r.count
                } for r in errors
            ],
            'warning_details': [
                {
                    'message': r.message,
                    'column': r.column,
                    'count': r.count
                } for r in warnings
            ]
        }