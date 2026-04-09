"""
CDU Response Analyzer for Per-CDU Comparative Analysis.

Analyzes input-output response characteristics at the CDU level
across different cooling models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional, Any
import logging

from raps.config import ConfigManager

logger = logging.getLogger(__name__)


class CDUResponseAnalyzer:
    """
    Analyzes CDU-level input-output response characteristics.
    
    Compares how individual CDUs respond to the same inputs across
    different cooling models (systems).
    """
    
    INPUT_VARS = ['Q_flow', 'T_Air', 'T_ext']
    OUTPUT_VARS = [
        'V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
        'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
        'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig'
    ]
    
    def __init__(self, system_name: str):
        """Initialize the CDU response analyzer."""
        self.system_name = system_name
        self.config = ConfigManager(system_name=system_name).get_config()
        self.num_cdus = self.config.get('NUM_CDUS', 1)
        
        logger.info(f"CDUResponseAnalyzer initialized for {system_name} ({self.num_cdus} CDUs)")
    
    def extract_cdu_data(self, data: pd.DataFrame, cdu_idx: int) -> pd.DataFrame:
        """Extract normalized per-CDU data from full simulation DataFrame."""
        extracted = {}
        
        # Input columns
        q_flow_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_Q_flow_total'
        t_air_col = f'simulator_1_datacenter_1_computeBlock_{cdu_idx}_cabinet_1_sources_T_Air'
        t_ext_col = 'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'
        
        if q_flow_col in data.columns:
            extracted['Q_flow'] = data[q_flow_col].values
        if t_air_col in data.columns:
            extracted['T_Air'] = data[t_air_col].values
        if t_ext_col in data.columns:
            extracted['T_ext'] = data[t_ext_col].values
        
        # Output columns
        for var in self.OUTPUT_VARS:
            col = f'simulator[1].datacenter[1].computeBlock[{cdu_idx}].cdu[1].summary.{var}'
            if col in data.columns:
                extracted[var] = data[col].values
        
        return pd.DataFrame(extracted)
    
    def compute_response_statistics(
        self,
        data: pd.DataFrame,
        cdu_idx: int = 1
    ) -> Dict[str, Any]:
        """
        Compute statistical summary of CDU responses.
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index to analyze
            
        Returns:
            Dictionary of response statistics
        """
        cdu_data = self.extract_cdu_data(data, cdu_idx)
        
        stats_dict = {
            'system': self.system_name,
            'cdu_idx': cdu_idx,
            'n_samples': len(cdu_data),
            'inputs': {},
            'outputs': {}
        }
        
        # Input statistics
        for var in self.INPUT_VARS:
            if var in cdu_data.columns:
                stats_dict['inputs'][var] = {
                    'mean': float(cdu_data[var].mean()),
                    'std': float(cdu_data[var].std()),
                    'min': float(cdu_data[var].min()),
                    'max': float(cdu_data[var].max()),
                    'range': float(cdu_data[var].max() - cdu_data[var].min())
                }
        
        # Output statistics
        for var in self.OUTPUT_VARS:
            if var in cdu_data.columns:
                stats_dict['outputs'][var] = {
                    'mean': float(cdu_data[var].mean()),
                    'std': float(cdu_data[var].std()),
                    'min': float(cdu_data[var].min()),
                    'max': float(cdu_data[var].max()),
                    'range': float(cdu_data[var].max() - cdu_data[var].min()),
                    'cv': float(cdu_data[var].std() / cdu_data[var].mean()) if cdu_data[var].mean() != 0 else 0
                }
        
        return stats_dict
    
    def compute_io_correlations(
        self,
        data: pd.DataFrame,
        cdu_idx: int = 1
    ) -> pd.DataFrame:
        """
        Compute input-output correlation matrix for a CDU.
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index to analyze
            
        Returns:
            DataFrame with correlation coefficients
        """
        cdu_data = self.extract_cdu_data(data, cdu_idx)
        
        available_inputs = [v for v in self.INPUT_VARS if v in cdu_data.columns]
        available_outputs = [v for v in self.OUTPUT_VARS if v in cdu_data.columns]
        
        correlations = []
        for inp in available_inputs:
            for out in available_outputs:
                try:
                    corr, pval = stats.pearsonr(cdu_data[inp], cdu_data[out])
                    correlations.append({
                        'input': inp,
                        'output': out,
                        'correlation': corr,
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
                except Exception:
                    pass
        
        return pd.DataFrame(correlations)
    
    def compute_static_gains(
        self,
        data: pd.DataFrame,
        cdu_idx: int = 1
    ) -> pd.DataFrame:
        """
        Compute static gain (sensitivity) for each input-output pair.
        
        Gain = delta_output / delta_input (linearized)
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index to analyze
            
        Returns:
            DataFrame with gain values
        """
        cdu_data = self.extract_cdu_data(data, cdu_idx)
        
        available_inputs = [v for v in self.INPUT_VARS if v in cdu_data.columns]
        available_outputs = [v for v in self.OUTPUT_VARS if v in cdu_data.columns]
        
        gains = []
        for inp in available_inputs:
            X = cdu_data[inp].values.reshape(-1, 1)
            
            for out in available_outputs:
                y = cdu_data[out].values
                
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    gain = model.coef_[0]
                    r2 = r2_score(y, model.predict(X))
                    
                    # Compute normalized gain
                    inp_range = cdu_data[inp].max() - cdu_data[inp].min()
                    out_range = cdu_data[out].max() - cdu_data[out].min()
                    norm_gain = (gain * inp_range / out_range) if out_range > 0 else 0
                    
                    gains.append({
                        'input': inp,
                        'output': out,
                        'gain': gain,
                        'normalized_gain': norm_gain,
                        'intercept': model.intercept_,
                        'r2': r2,
                        'linear_fit_quality': 'good' if r2 > 0.8 else 'moderate' if r2 > 0.5 else 'poor'
                    })
                except Exception as e:
                    logger.warning(f"Failed to compute gain for {inp}->{out}: {e}")
        
        return pd.DataFrame(gains)
    
    def compute_nonlinearity_index(
        self,
        data: pd.DataFrame,
        cdu_idx: int = 1,
        max_degree: int = 3
    ) -> pd.DataFrame:
        """
        Compute nonlinearity index for each input-output pair.
        
        Compares linear vs polynomial fit quality.
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index to analyze
            max_degree: Maximum polynomial degree to test
            
        Returns:
            DataFrame with nonlinearity indices
        """
        cdu_data = self.extract_cdu_data(data, cdu_idx)
        
        available_inputs = [v for v in self.INPUT_VARS if v in cdu_data.columns]
        available_outputs = [v for v in self.OUTPUT_VARS if v in cdu_data.columns]
        
        nonlinearity = []
        for inp in available_inputs:
            X = cdu_data[inp].values.reshape(-1, 1)
            
            for out in available_outputs:
                y = cdu_data[out].values
                
                try:
                    # Linear fit
                    lin_model = LinearRegression()
                    lin_model.fit(X, y)
                    lin_r2 = r2_score(y, lin_model.predict(X))
                    
                    # Polynomial fits
                    best_poly_r2 = lin_r2
                    best_degree = 1
                    
                    for degree in range(2, max_degree + 1):
                        poly = PolynomialFeatures(degree=degree)
                        X_poly = poly.fit_transform(X)
                        poly_model = LinearRegression()
                        poly_model.fit(X_poly, y)
                        poly_r2 = r2_score(y, poly_model.predict(X_poly))
                        
                        if poly_r2 > best_poly_r2:
                            best_poly_r2 = poly_r2
                            best_degree = degree
                    
                    # Nonlinearity index: improvement from polynomial fit
                    nonlin_index = (best_poly_r2 - lin_r2) / (1 - lin_r2 + 1e-10)
                    
                    nonlinearity.append({
                        'input': inp,
                        'output': out,
                        'linear_r2': lin_r2,
                        'best_poly_r2': best_poly_r2,
                        'best_poly_degree': best_degree,
                        'nonlinearity_index': nonlin_index,
                        'relationship': 'linear' if nonlin_index < 0.1 else 'mildly_nonlinear' if nonlin_index < 0.3 else 'nonlinear'
                    })
                except Exception as e:
                    logger.warning(f"Failed to compute nonlinearity for {inp}->{out}: {e}")
        
        return pd.DataFrame(nonlinearity)
    
    def compute_response_surface_samples(
        self,
        data: pd.DataFrame,
        cdu_idx: int = 1,
        input_pair: Tuple[str, str] = ('Q_flow', 'T_Air'),
        output_var: str = 'T_sec_r_C',
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute binned response surface for visualization.
        
        Args:
            data: Full simulation DataFrame
            cdu_idx: CDU index
            input_pair: Tuple of two input variable names
            output_var: Output variable to analyze
            n_bins: Number of bins per dimension
            
        Returns:
            Dictionary with grid coordinates and response values
        """
        cdu_data = self.extract_cdu_data(data, cdu_idx)
        
        inp1, inp2 = input_pair
        
        if inp1 not in cdu_data.columns or inp2 not in cdu_data.columns:
            return {}
        if output_var not in cdu_data.columns:
            return {}
        
        # Create bins
        x1_bins = np.linspace(cdu_data[inp1].min(), cdu_data[inp1].max(), n_bins + 1)
        x2_bins = np.linspace(cdu_data[inp2].min(), cdu_data[inp2].max(), n_bins + 1)
        
        # Compute mean response in each bin
        response_grid = np.full((n_bins, n_bins), np.nan)
        count_grid = np.zeros((n_bins, n_bins))
        
        for i in range(n_bins):
            for j in range(n_bins):
                mask = (
                    (cdu_data[inp1] >= x1_bins[i]) & (cdu_data[inp1] < x1_bins[i+1]) &
                    (cdu_data[inp2] >= x2_bins[j]) & (cdu_data[inp2] < x2_bins[j+1])
                )
                if mask.sum() > 0:
                    response_grid[i, j] = cdu_data.loc[mask, output_var].mean()
                    count_grid[i, j] = mask.sum()
        
        # Grid centers
        x1_centers = (x1_bins[:-1] + x1_bins[1:]) / 2
        x2_centers = (x2_bins[:-1] + x2_bins[1:]) / 2
        
        return {
            'x1_centers': x1_centers,
            'x2_centers': x2_centers,
            'x1_name': inp1,
            'x2_name': inp2,
            'output_name': output_var,
            'response_grid': response_grid,
            'count_grid': count_grid
        }
    
    def analyze_all_cdus(
        self,
        data: pd.DataFrame,
        max_cdus: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze response characteristics for all CDUs.
        
        Args:
            data: Full simulation DataFrame
            max_cdus: Maximum CDUs to analyze (None = all)
            
        Returns:
            Dictionary with analysis results for all CDUs
        """
        n_cdus = min(self.num_cdus, max_cdus) if max_cdus else self.num_cdus
        
        results = {
            'system': self.system_name,
            'num_cdus_analyzed': n_cdus,
            'statistics': [],
            'correlations': [],
            'gains': [],
            'nonlinearity': []
        }
        
        for cdu_idx in range(1, n_cdus + 1):
            logger.info(f"Analyzing CDU {cdu_idx}/{n_cdus}")
            
            stats = self.compute_response_statistics(data, cdu_idx)
            stats['cdu_idx'] = cdu_idx
            results['statistics'].append(stats)
            
            corr_df = self.compute_io_correlations(data, cdu_idx)
            corr_df['cdu_idx'] = cdu_idx
            results['correlations'].append(corr_df)
            
            gains_df = self.compute_static_gains(data, cdu_idx)
            gains_df['cdu_idx'] = cdu_idx
            results['gains'].append(gains_df)
            
            nonlin_df = self.compute_nonlinearity_index(data, cdu_idx)
            nonlin_df['cdu_idx'] = cdu_idx
            results['nonlinearity'].append(nonlin_df)
        
        # Aggregate DataFrames
        if results['correlations']:
            results['correlations'] = pd.concat(results['correlations'], ignore_index=True)
        if results['gains']:
            results['gains'] = pd.concat(results['gains'], ignore_index=True)
        if results['nonlinearity']:
            results['nonlinearity'] = pd.concat(results['nonlinearity'], ignore_index=True)
        
        return results
