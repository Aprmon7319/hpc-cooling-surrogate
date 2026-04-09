"""
Systematic FMU Input Generator

Generates comprehensive input data covering the full state space
using structured scenarios: steady-state grid, step responses,
ramps, sinusoids, and realistic random patterns.

Supports independent per-CDU scenario generation using LHS sampling.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scenario_definitions import (
    generate_all_scenarios,
    ScenarioSpec,
    ScenarioType,
    OperatingPoint
)
from .scenario_sequencer import ScenarioSequencer, SequencedScenario
from .input_sequence_builder import InputSequenceBuilder, InputConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystematicInputGenerator:
    """
    Main orchestrator for systematic FMU input generation.
    
    Generates structured input data covering:
    - Steady-state factorial grid points
    - Step responses
    - Ramp sweeps
    - Sinusoidal/diurnal patterns
    - Random realistic scenarios
    
    Supports two modes:
    - Shared: All CDUs follow the same scenario sequence (with jitter)
    - Independent: Each CDU has independently sampled scenarios (LHS)
    """
    
    def __init__(
        self,
        n_cdus: int = 49,
        timestep_seconds: int = 1,
        transition_ramp_seconds: int = 60,
        jitter_fraction: float = 0.02,
        timing_offset_fraction: float = 0.1,
        seed: int = 42,
        config: Optional[Dict] = None
    ):
        self.n_cdus = n_cdus
        self.timestep_seconds = timestep_seconds
        self.seed = seed
        
        # Load config if not provided
        if config is None:
            try:
                from raps.config import ConfigManager
                config = ConfigManager(system_name="marconi100").get_config()
                logger.info("Using Marconi100 configuration")
            except ImportError:
                config = {
                    'MIN_POWER': 12.3652,
                    'MAX_POWER': 38.5552,
                }
                logger.info("Using default configuration")
        
        self.config = config
        
        # Get power range from config
        q_flow_min_kw = config.get('MIN_POWER', 10.0)
        q_flow_max_kw = config.get('MAX_POWER', 100.0)
        
        self.input_config = InputConfig(
            n_cdus=n_cdus,
            q_flow_max_kw=q_flow_max_kw,
            q_flow_min_kw=q_flow_min_kw,
            timestep_seconds=timestep_seconds,
            transition_ramp_seconds=transition_ramp_seconds,
            jitter_fraction=jitter_fraction,
            timing_offset_fraction=timing_offset_fraction,
            seed=seed
        )
        
        self.sequencer = ScenarioSequencer(
            transition_buffer_seconds=transition_ramp_seconds
        )
        
        self.builder = InputSequenceBuilder(self.input_config)
    
    def generate(
        self,
        duration_hours: Optional[int] = None,
        steady_state_duration: int = 600,
        step_ramp_duration: int = 30,
        include_random: bool = True,
        sequence_method: str = "monotonic",
        apply_jitter: bool = True,
        independent_cdus: bool = True,  # NEW: Enable independent CDU scenarios
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[ScenarioSpec]]:
        """
        Generate complete systematic input dataset.
        
        Parameters
        ----------
        duration_hours : int, optional
            If provided, limits total duration. Otherwise generates all scenarios.
        steady_state_duration : int
            Duration of each steady-state scenario in seconds (default 600 = 10 min)
        step_ramp_duration : int
            Duration of ramps in step scenarios in seconds (default 30)
        include_random : bool
            Whether to include random realistic scenarios
        sequence_method : str
            "monotonic" for sweep-based ordering, "greedy" for nearest-neighbor
        apply_jitter : bool
            Whether to apply per-CDU timing jitter (only for shared mode)
        independent_cdus : bool
            If True, each CDU gets independently sampled scenarios (LHS).
            If False, all CDUs share the same scenario sequence.
        
        Returns
        -------
        Tuple containing:
            - fmu_input: FMU-formatted input DataFrame
            - raw_input: Raw multi-CDU DataFrame with metadata
            - scenarios: List of scenario specifications
        """
        logger.info("="*60)
        logger.info("Systematic FMU Input Generation")
        logger.info(f"Mode: {'Independent CDUs (LHS)' if independent_cdus else 'Shared sequence'}")
        logger.info("="*60)
        
        # Step 1: Generate all scenarios (pool)
        logger.info("\n1. Generating scenario pool...")
        scenarios = generate_all_scenarios(
            steady_state_duration=steady_state_duration,
            step_ramp_duration=step_ramp_duration,
            include_random=include_random,
            seed=self.seed
        )
        
        scenario_counts = {}
        for s in scenarios:
            t = s.scenario_type.value
            scenario_counts[t] = scenario_counts.get(t, 0) + 1
        
        logger.info(f"   Generated {len(scenarios)} scenarios in pool:")
        for t, count in scenario_counts.items():
            logger.info(f"     - {t}: {count}")
        
        # Calculate total duration
        if duration_hours is not None:
            total_duration_seconds = duration_hours * 3600
        else:
            # Use all scenarios' total duration
            total_duration_seconds = sum(s.duration_seconds for s in scenarios)
            total_duration_seconds += len(scenarios) * self.input_config.transition_ramp_seconds
        
        logger.info(f"   Target duration: {total_duration_seconds / 3600:.1f} hours")
        
        if independent_cdus:
            # Step 2: Build independent sequences for each CDU
            logger.info("\n2. Building independent CDU sequences (LHS sampling)...")
            
            raw_df = self.builder.build_independent_cdu_sequences(
                all_scenarios=scenarios,
                total_duration_seconds=total_duration_seconds,
                sequencer=self.sequencer
            )
            
            # Log diversity statistics
            if hasattr(raw_df, 'attrs') and 'cdu_metadata' in raw_df.attrs:
                metadata = raw_df.attrs['cdu_metadata']
                scenario_counts_per_cdu = [m['n_scenarios'] for m in metadata.values()]
                logger.info(f"   Scenarios per CDU: min={min(scenario_counts_per_cdu)}, "
                           f"max={max(scenario_counts_per_cdu)}, "
                           f"mean={np.mean(scenario_counts_per_cdu):.1f}")
            
            logger.info(f"   Raw input shape: {raw_df.shape}")
            
        else:
            # Legacy: shared sequence mode
            logger.info("\n2. Sequencing scenarios (shared mode)...")
            
            if sequence_method == "monotonic":
                sequenced = self.sequencer.sequence_monotonic_sweeps(scenarios)
            else:
                sequenced = self.sequencer.sequence_by_type_then_greedy(scenarios)
            
            stats = self.sequencer.get_sequence_stats(sequenced)
            logger.info(f"   Total duration: {stats['total_duration_hours']:.1f} hours")
            logger.info(f"   Mean transition cost: {stats['mean_transition_cost']:.3f}")
            
            # Limit duration if specified
            if duration_hours is not None:
                max_seconds = duration_hours * 3600
                filtered_sequenced = []
                cumulative_time = 0
                
                for seq in sequenced:
                    if cumulative_time + seq.scenario.duration_seconds <= max_seconds:
                        filtered_sequenced.append(seq)
                        cumulative_time += seq.scenario.duration_seconds + self.input_config.transition_ramp_seconds
                    else:
                        break
                
                sequenced = filtered_sequenced
                logger.info(f"   Limited to {len(sequenced)} scenarios for {duration_hours} hours")
            
            logger.info("\n3. Building input time series...")
            raw_df = self.builder.build_multi_cdu_sequence(
                sequenced,
                apply_jitter=apply_jitter
            )
            logger.info(f"   Raw input shape: {raw_df.shape}")
        
        # Step 3: Format for FMU
        logger.info("\n4. Formatting for FMU...")
        fmu_df = self._format_for_fmu(raw_df)
        logger.info(f"   FMU input shape: {fmu_df.shape}")
        
        # Verify CDU independence (for independent mode)
        if independent_cdus:
            self._log_independence_stats(raw_df)
        
        logger.info("\n" + "="*60)
        logger.info("Generation complete!")
        logger.info("="*60)
        
        return fmu_df, raw_df, scenarios
    
    def _log_independence_stats(self, raw_df: pd.DataFrame):
        """Log statistics showing CDU independence"""
        logger.info("\n5. CDU Independence Statistics:")
        
        # Get phase columns
        phase_cols = [c for c in raw_df.columns if '_phase' in c]
        
        if len(phase_cols) >= 2:
            # Sample a few timesteps to show diversity
            sample_timesteps = [0, len(raw_df)//4, len(raw_df)//2, 3*len(raw_df)//4]
            
            for t in sample_timesteps:
                if t >= len(raw_df):
                    continue
                phases = {col.replace('_phase', ''): raw_df[col].iloc[t] 
                         for col in phase_cols[:5]}  # First 5 CDUs
                logger.info(f"   t={t}: {phases}")
            
            # Calculate phase diversity metric
            phase_matrix = raw_df[phase_cols].values
            unique_phases_per_timestep = np.array([
                len(np.unique(phase_matrix[t, :])) 
                for t in range(min(1000, len(raw_df)))
            ])
            avg_diversity = np.mean(unique_phases_per_timestep)
            logger.info(f"   Average phase diversity: {avg_diversity:.2f} unique phases per timestep")
    
    def _format_for_fmu(self, multi_cdu_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to FMU input format with correct column names.
        """
        fmu_data = {}
        
        # External temperature (shared across all CDUs)
        fmu_data['simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext'] = \
            multi_cdu_df['t_ext_k'].values
        
        # CDU data with correct FMU column names
        for col in multi_cdu_df.columns:
            if '_q_flow_w' in col:
                cdu_name = col.replace('_q_flow_w', '')
                cdu_num = int(cdu_name.split('_')[1])
                
                fmu_col = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_Q_flow_total'
                fmu_data[fmu_col] = multi_cdu_df[col].values
            
            elif '_t_air_k' in col and '_phase' not in col and '_scenario' not in col:
                cdu_name = col.replace('_t_air_k', '')
                cdu_num = int(cdu_name.split('_')[1])
                
                fmu_col = f'simulator_1_datacenter_1_computeBlock_{cdu_num}_cabinet_1_sources_T_Air'
                fmu_data[fmu_col] = multi_cdu_df[col].values
        
        return pd.DataFrame(fmu_data)


def generate_systematic_fmu_dataset(
    n_cdus: int = 49,
    duration_hours: Optional[int] = None,
    timestep_seconds: int = 1,
    seed: int = 42,
    output_dir: str = "data",
    save_output: bool = False,
    config: Optional[Dict] = None,
    steady_state_duration: int = 600,
    step_ramp_duration: int = 30,
    include_random: bool = True,
    sequence_method: str = "monotonic",
    apply_jitter: bool = True,
    independent_cdus: bool = True  # NEW: default to independent mode
) -> Dict:
    """
    Generate complete systematic FMU-compatible dataset.
    
    Interface matches generate_complete_fmu_dataset() for consistency.
    
    Parameters
    ----------
    n_cdus : int
        Number of CDUs to simulate
    duration_hours : int, optional
        Maximum duration in hours. If None, generates all scenarios (~73 hours)
    timestep_seconds : int
        Time step in seconds (default 1)
    seed : int
        Random seed for reproducibility
    output_dir : str
        Directory to save output files
    save_output : bool
        Whether to save the output to file
    config : Dict, optional
        System configuration (uses Marconi100 defaults if None)
    steady_state_duration : int
        Duration of each steady-state scenario in seconds
    step_ramp_duration : int
        Duration of ramps in step scenarios in seconds
    include_random : bool
        Whether to include random realistic scenarios
    sequence_method : str
        "monotonic" for sweep-based ordering, "greedy" for nearest-neighbor
    apply_jitter : bool
        Whether to apply per-CDU timing jitter (shared mode only)
    independent_cdus : bool
        If True, each CDU gets independently sampled scenarios.
        If False, all CDUs share the same scenario sequence.
    
    Returns
    -------
    Dict containing:
        - 'fmu_input': Complete FMU input DataFrame
        - 'raw_input': Raw input DataFrame with metadata
        - 'scenarios': Scenario specifications
        - 'filename': Output filename if saved
    """
    # Load config if not provided
    if config is None:
        try:
            from raps.config import ConfigManager
            config = ConfigManager(system_name="marconi100").get_config()
            print("Using Marconi100 configuration")
        except ImportError:
            config = {
                'MIN_POWER': 12.3652,
                'MAX_POWER': 38.5552,
                'system_name': 'default'
            }
            print("Using default configuration")
    
    mode_str = "Independent (LHS)" if independent_cdus else "Shared"
    print(f"Generating Systematic FMU dataset:")
    print(f"  - CDUs: {n_cdus}")
    print(f"  - Duration: {duration_hours if duration_hours else 'all scenarios (~73)'} hours")
    print(f"  - Timestep: {timestep_seconds} seconds")
    print(f"  - Mode: {mode_str}")
    print(f"  - Steady-state duration: {steady_state_duration}s per point")
    print(f"  - Sequence method: {sequence_method}")
    
    # Create generator
    generator = SystematicInputGenerator(
        n_cdus=n_cdus,
        timestep_seconds=timestep_seconds,
        seed=seed,
        config=config
    )
    
    # Generate data
    fmu_input_df, raw_input_df, scenarios = generator.generate(
        duration_hours=duration_hours,
        steady_state_duration=steady_state_duration,
        step_ramp_duration=step_ramp_duration,
        include_random=include_random,
        sequence_method=sequence_method,
        apply_jitter=apply_jitter,
        independent_cdus=independent_cdus
    )
    
    # Save output if requested
    filename = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        system_name = config.get('system_name', 'systematic')
        actual_hours = len(fmu_input_df) * timestep_seconds / 3600
        mode_suffix = "independent" if independent_cdus else "shared"
        
        filename = f"systematic_fmu_input_{int(actual_hours)}hrs_{system_name}_{n_cdus}CDU_{mode_suffix}.parquet"
        filepath = os.path.join(output_dir, filename)
        fmu_input_df.to_parquet(filepath, index=False)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"systematic_metadata_{int(actual_hours)}hrs_{mode_suffix}.json")
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'n_cdus': n_cdus,
            'duration_hours': actual_hours,
            'timestep_seconds': timestep_seconds,
            'n_scenarios': len(scenarios),
            'seed': seed,
            'independent_cdus': independent_cdus,
            'scenario_counts': {}
        }
        for s in scenarios:
            t = s.scenario_type.value
            metadata['scenario_counts'][t] = metadata['scenario_counts'].get(t, 0) + 1
        
        # Include per-CDU metadata if available
        if hasattr(raw_input_df, 'attrs') and 'cdu_metadata' in raw_input_df.attrs:
            metadata['cdu_metadata'] = raw_input_df.attrs['cdu_metadata']
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save raw input with phase information for analysis
        raw_file = os.path.join(output_dir, f"systematic_raw_{int(actual_hours)}hrs_{mode_suffix}.parquet")
        raw_input_df.to_parquet(raw_file, index=False)
        
        print(f"\nDataset saved to: {filepath}")
        print(f"Raw data saved to: {raw_file}")
        print(f"Metadata saved to: {metadata_file}")
    
    return {
        'fmu_input': fmu_input_df,
        'raw_input': raw_input_df,
        'scenarios': scenarios,
        'filename': filename
    }


# Convenience alias
generate_systematic_dataset = generate_systematic_fmu_dataset


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate systematic FMU input dataset"
    )
    parser.add_argument("--n-cdus", type=int, default=49,
                       help="Number of CDUs")
    parser.add_argument("--duration-hours", type=int, default=None,
                       help="Maximum duration in hours (default: all scenarios)")
    parser.add_argument("--timestep-seconds", type=int, default=1,
                       help="Timestep in seconds")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--steady-state-duration", type=int, default=600,
                       help="Steady-state scenario duration in seconds")
    parser.add_argument("--no-random", action="store_true",
                       help="Exclude random realistic scenarios")
    parser.add_argument("--no-jitter", action="store_true",
                       help="Disable per-CDU jitter")
    parser.add_argument("--sequence-method", type=str, default="monotonic",
                       choices=["monotonic", "greedy"],
                       help="Scenario sequencing method")
    parser.add_argument("--save", action="store_true",
                       help="Save output to file")
    parser.add_argument("--system-name", type=str, default="marconi100",
                       help="System configuration name")
    parser.add_argument("--shared-mode", action="store_true",
                       help="Use shared scenario sequence for all CDUs (legacy mode)")
    
    args = parser.parse_args()
    
    # Load config
    try:
        from raps.config import ConfigManager
        config = ConfigManager(system_name=args.system_name).get_config()
    except ImportError:
        config = None
    
    result = generate_systematic_fmu_dataset(
        n_cdus=args.n_cdus,
        duration_hours=args.duration_hours,
        timestep_seconds=args.timestep_seconds,
        seed=args.seed,
        output_dir=args.output_dir,
        save_output=args.save,
        config=config,
        steady_state_duration=args.steady_state_duration,
        include_random=not args.no_random,
        apply_jitter=not args.no_jitter,
        sequence_method=args.sequence_method,
        independent_cdus=not args.shared_mode
    )
    
    print(f"\n✓ Generated {len(result['scenarios'])} scenarios in pool")
    print(f"  FMU input shape: {result['fmu_input'].shape}")
    actual_hours = len(result['fmu_input']) * args.timestep_seconds / 3600
    print(f"  Total duration: {actual_hours:.1f} hours")
    print(f"  Mode: {'Shared' if args.shared_mode else 'Independent (LHS)'}")