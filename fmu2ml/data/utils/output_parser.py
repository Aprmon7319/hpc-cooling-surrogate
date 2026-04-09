import pandas as pd
from typing import  List, Optional
import logging


class OutputParser:
    """
    Parses FMU simulation output data
    
    Handles:
    - Column extraction
    - Derived quantity computation
    - Data aggregation
    - Feature engineering
    - Format conversion
    """
    
    def __init__(self, num_cdus: int = 49):
        """
        Initialize output parser
        
        Args:
            num_cdus: Number of CDUs in the system
        """
        self.logger = logging.getLogger(__name__)
        self.num_cdus = num_cdus
        
        self.logger.info(f"Output parser initialized: {num_cdus} CDUs")
    
    def parse_simulation_output(
        self,
        output_df: pd.DataFrame,
        extract_features: bool = True
    ) -> pd.DataFrame:
        """
        Parse raw FMU output
        
        Args:
            output_df: Raw output DataFrame from FMU
            extract_features: Whether to extract additional features
            
        Returns:
            Parsed DataFrame with selected columns
        """
        self.logger.debug(f"Parsing simulation output: {len(output_df)} samples")
        
        # Extract base outputs
        parsed = self._extract_base_outputs(output_df)
        
        
        return parsed
    
    def _extract_base_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract base output variables"""
        base_outputs = {}
        
        # Time
        if 'time' in df.columns:
            base_outputs['time'] = df['time']
        
        # Datacenter level outputs
        dc_flow_col = 'simulator[1].datacenter[1].summary.V_flow_prim_GPM'
        if dc_flow_col in df.columns:
            base_outputs['datacenter_V_flow_prim_GPM'] = df[dc_flow_col]
        
        if 'pue' in df.columns:
            base_outputs['pue'] = df['pue']
        
        # CDU outputs - all parameters
        cdu_params = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                      'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                      'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig']
        
        for i in range(1, self.num_cdus + 1):
            for param in cdu_params:
                col_name = f'simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary.{param}'
                if col_name in df.columns:
                    base_outputs[f'CDU_{i}_{param}'] = df[col_name]
        
        # HTC for each CDU
        for i in range(1, self.num_cdus + 1):
            htc_col = f'simulator[1].datacenter[1].computeBlock[{i}].cabinet[1].summary.htc'
            if htc_col in df.columns:
                base_outputs[f'CDU_{i}_htc'] = df[htc_col]
        
        return pd.DataFrame(base_outputs)
    
    
    def extract_cdu_outputs(
        self,
        df: pd.DataFrame,
        cdu_indices: Optional[List[int]] = None,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract outputs for specific CDUs and parameters
        
        Args:
            df: Output DataFrame
            cdu_indices: List of CDU indices (1-based), None for all
            parameters: List of parameters to extract, None for all
            
        Returns:
            DataFrame with selected CDU outputs
        """
        if cdu_indices is None:
            cdu_indices = list(range(1, self.num_cdus + 1))
        
        if parameters is None:
            parameters = ['V_flow_prim_GPM', 'V_flow_sec_GPM', 'W_flow_CDUP_kW',
                         'T_prim_s_C', 'T_prim_r_C', 'T_sec_s_C', 'T_sec_r_C',
                         'p_prim_s_psig', 'p_prim_r_psig', 'p_sec_s_psig', 'p_sec_r_psig',
                         'htc']
        
        selected_cols = ['time'] if 'time' in df.columns else []
        
        for idx in cdu_indices:
            for param in parameters:
                col_name = f'CDU_{idx}_{param}'
                if col_name in df.columns:
                    selected_cols.append(col_name)
        
        return df[selected_cols].copy()
    
    def save_parsed_output(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save parsed output to file
        
        Args:
            df: Parsed DataFrame
            output_path: Output file path
            format: File format ('parquet', 'csv', 'hdf5')
        """
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'hdf5':
            df.to_hdf(output_path, key='outputs', mode='w')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved parsed output to: {output_path}")