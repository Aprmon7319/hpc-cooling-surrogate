"""
Optimal scenario sequencing to minimize transition magnitudes.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .scenario_definitions import ScenarioSpec, ScenarioType, OperatingPoint


@dataclass
class SequencedScenario:
    """Scenario with sequencing metadata"""
    scenario: ScenarioSpec
    sequence_index: int
    start_time_seconds: int
    transition_cost: float


class ScenarioSequencer:
    """Orders scenarios to minimize transition costs."""
    
    def __init__(
        self,
        initial_point: Optional[OperatingPoint] = None,
        transition_buffer_seconds: int = 30  # Reduced from 60
    ):
        self.initial_point = initial_point or OperatingPoint(
            q_flow_fraction=0.50,
            t_air_k=298.0,
            t_ext_k=288.0
        )
        self.transition_buffer = transition_buffer_seconds
    
    def _get_scenario_start_point(self, scenario: ScenarioSpec) -> OperatingPoint:
        """Get the starting operating point of a scenario"""
        if scenario.operating_points:
            return scenario.operating_points[0]
        return OperatingPoint(q_flow_fraction=0.50, t_air_k=298.0, t_ext_k=288.0)
    
    def _get_scenario_end_point(self, scenario: ScenarioSpec) -> OperatingPoint:
        """Get the ending operating point of a scenario"""
        if scenario.operating_points:
            return scenario.operating_points[-1]
        return OperatingPoint(q_flow_fraction=0.50, t_air_k=298.0, t_ext_k=288.0)
    
    def _transition_cost(
        self,
        from_point: OperatingPoint,
        to_point: OperatingPoint
    ) -> float:
        """Calculate transition cost between two operating points."""
        q_diff = abs(from_point.q_flow_fraction - to_point.q_flow_fraction)
        t_air_diff = abs(from_point.t_air_k - to_point.t_air_k) / 50.0
        t_ext_diff = abs(from_point.t_ext_k - to_point.t_ext_k) / 50.0
        return 2.0 * q_diff + t_air_diff + t_ext_diff
    
    def sequence_random(
        self,
        scenarios: List[ScenarioSpec],
        seed: int = 42
    ) -> List[SequencedScenario]:
        """Fully random sequencing."""
        rng = np.random.RandomState(seed)
        shuffled = list(scenarios)
        rng.shuffle(shuffled)
        
        sequenced = []
        current_point = self.initial_point
        current_time = 0
        
        for scenario in shuffled:
            start_point = self._get_scenario_start_point(scenario)
            cost = self._transition_cost(current_point, start_point)
            
            sequenced.append(SequencedScenario(
                scenario=scenario,
                sequence_index=len(sequenced),
                start_time_seconds=current_time,
                transition_cost=cost
            ))
            
            current_time += self.transition_buffer + scenario.duration_seconds
            current_point = self._get_scenario_end_point(scenario)
        
        return sequenced
    
    def sequence_interleaved(
        self,
        scenarios: List[ScenarioSpec],
        seed: int = 42
    ) -> List[SequencedScenario]:
        """Interleaved sequencing - mix scenario types throughout."""
        rng = np.random.RandomState(seed)
        
        # All scenario types including new ones
        type_order = [
            ScenarioType.STEADY_STATE,
            ScenarioType.STEP_RESPONSE,
            ScenarioType.T_EXT_STEP,
            ScenarioType.RAMP_SWEEP,
            ScenarioType.T_EXT_RAMP,
            ScenarioType.COMBINED_STRESS,
            ScenarioType.SINUSOIDAL,
            ScenarioType.RANDOM_REALISTIC,
        ]
        
        grouped = {t: [] for t in type_order}
        for scenario in scenarios:
            if scenario.scenario_type in grouped:
                grouped[scenario.scenario_type].append(scenario)
        
        # Shuffle within each group
        for t in type_order:
            rng.shuffle(grouped[t])
        
        # Interleave
        interleaved = []
        while any(grouped[t] for t in type_order):
            for t in type_order:
                if grouped[t]:
                    interleaved.append(grouped[t].pop(0))
        
        sequenced = []
        current_point = self.initial_point
        current_time = 0
        
        for scenario in interleaved:
            start_point = self._get_scenario_start_point(scenario)
            cost = self._transition_cost(current_point, start_point)
            
            sequenced.append(SequencedScenario(
                scenario=scenario,
                sequence_index=len(sequenced),
                start_time_seconds=current_time,
                transition_cost=cost
            ))
            
            current_time += self.transition_buffer + scenario.duration_seconds
            current_point = self._get_scenario_end_point(scenario)
        
        return sequenced
    
    def sequence_by_type_then_greedy(
        self,
        scenarios: List[ScenarioSpec]
    ) -> List[SequencedScenario]:
        """Sequence by type then greedy within each type."""
        type_order = [
            ScenarioType.STEADY_STATE,
            ScenarioType.RAMP_SWEEP,
            ScenarioType.T_EXT_RAMP,
            ScenarioType.STEP_RESPONSE,
            ScenarioType.T_EXT_STEP,
            ScenarioType.COMBINED_STRESS,
            ScenarioType.SINUSOIDAL,
            ScenarioType.RANDOM_REALISTIC,
        ]
        
        grouped = {t: [] for t in type_order}
        for scenario in scenarios:
            if scenario.scenario_type in grouped:
                grouped[scenario.scenario_type].append(scenario)
        
        sequenced = []
        current_point = self.initial_point
        current_time = 0
        
        for scenario_type in type_order:
            group = grouped[scenario_type]
            if not group:
                continue
            
            remaining = list(group)
            
            while remaining:
                best_idx = 0
                best_cost = float('inf')
                
                for idx, scenario in enumerate(remaining):
                    start_point = self._get_scenario_start_point(scenario)
                    cost = self._transition_cost(current_point, start_point)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                
                scenario = remaining.pop(best_idx)
                
                sequenced.append(SequencedScenario(
                    scenario=scenario,
                    sequence_index=len(sequenced),
                    start_time_seconds=current_time,
                    transition_cost=best_cost
                ))
                
                current_time += self.transition_buffer + scenario.duration_seconds
                current_point = self._get_scenario_end_point(scenario)
        
        return sequenced
    
    def sequence_monotonic_sweeps(
        self,
        scenarios: List[ScenarioSpec]
    ) -> List[SequencedScenario]:
        """Monotonic sweeps for steady-state, then other scenarios."""
        steady_state = [s for s in scenarios if s.scenario_type == ScenarioType.STEADY_STATE]
        others = [s for s in scenarios if s.scenario_type != ScenarioType.STEADY_STATE]
        
        if not steady_state:
            return self.sequence_by_type_then_greedy(scenarios)
        
        # Build index
        ss_index = {}
        for s in steady_state:
            op = s.operating_points[0]
            key = (op.q_flow_fraction, op.t_air_k, op.t_ext_k)
            ss_index[key] = s
        
        q_levels = sorted(set(k[0] for k in ss_index.keys()))
        t_air_levels = sorted(set(k[1] for k in ss_index.keys()))
        t_ext_levels = sorted(set(k[2] for k in ss_index.keys()))
        
        ordered_ss = []
        q_forward = True
        t_air_forward = True
        
        for t_ext in t_ext_levels:
            t_air_iter = t_air_levels if t_air_forward else reversed(t_air_levels)
            
            for t_air in t_air_iter:
                q_iter = q_levels if q_forward else list(reversed(q_levels))
                
                for q in q_iter:
                    key = (q, t_air, t_ext)
                    if key in ss_index:
                        ordered_ss.append(ss_index[key])
                
                q_forward = not q_forward
            
            t_air_forward = not t_air_forward
        
        sequenced = []
        current_point = self.initial_point
        current_time = 0
        
        for scenario in ordered_ss:
            start_point = self._get_scenario_start_point(scenario)
            cost = self._transition_cost(current_point, start_point)
            
            sequenced.append(SequencedScenario(
                scenario=scenario,
                sequence_index=len(sequenced),
                start_time_seconds=current_time,
                transition_cost=cost
            ))
            
            current_time += self.transition_buffer + scenario.duration_seconds
            current_point = self._get_scenario_end_point(scenario)
        
        # Append others using interleaved
        for scenario in others:
            start_point = self._get_scenario_start_point(scenario)
            cost = self._transition_cost(current_point, start_point)
            
            sequenced.append(SequencedScenario(
                scenario=scenario,
                sequence_index=len(sequenced),
                start_time_seconds=current_time,
                transition_cost=cost
            ))
            
            current_time += self.transition_buffer + scenario.duration_seconds
            current_point = self._get_scenario_end_point(scenario)
        
        return sequenced
    
    def get_sequence_stats(self, sequenced: List[SequencedScenario]) -> dict:
        """Get statistics about the sequence"""
        if not sequenced:
            return {}
        
        costs = [s.transition_cost for s in sequenced]
        total_time = sequenced[-1].start_time_seconds + sequenced[-1].scenario.duration_seconds
        
        # Count all scenario types
        type_counts = {}
        for s in sequenced:
            t = s.scenario.scenario_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'n_scenarios': len(sequenced),
            'total_duration_hours': total_time / 3600,
            'mean_transition_cost': np.mean(costs),
            'max_transition_cost': np.max(costs),
            'total_transition_cost': np.sum(costs),
            'scenarios_by_type': type_counts
        }