import numpy as np
from scipy.stats import qmc
from typing import List, Dict, Optional


class ScenarioGenerator:
    """Generate scenarios using Latin Hypercube Sampling"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_scenario_lhs(
        self,
        n_scenarios: int = 10,
        n_cdus: int = 51,
        distribution: List[float] = [0.5, 0.4, 0.1],
        seed: Optional[int] = None  # Add seed parameter
    ) -> List[Dict]:
        """
        Generate scenarios for multiple CDUs with proper distribution
        
        Parameters:
        -----------
        n_scenarios : int
            Total number of scenarios to generate
        n_cdus : int
            Number of CDUs
        distribution : List[float]
            Distribution of [normal, edge, fault] scenarios
        seed : int, optional
            Random seed for this specific generation (overrides instance seed)
        
        Returns:
        --------
        List[Dict] : List of scenario dictionaries
        """
        # Use provided seed or instance seed
        random_seed = seed if seed is not None else self.seed
        np.random.seed(random_seed)
        
        # Calculate scenario distribution
        n_normal = int(n_scenarios * distribution[0])
        n_edge = int(n_scenarios * distribution[1])
        n_fault = n_scenarios - n_normal - n_edge
        
        all_scenarios = []
        
        # Generate scenarios for each CDU
        for cdu_id in range(n_cdus):
            scenarios = []
            
            # Normal scenarios
            if n_normal > 0:
                sampler = qmc.LatinHypercube(d=2, seed=random_seed + cdu_id * 3)
                params = sampler.random(n_normal)
                for i in range(n_normal):
                    scenarios.append({
                        "cdu_id": cdu_id,
                        "type": "normal",
                        "params": params[i],
                        "power_series": None
                    })
            
            # Edge scenarios
            if n_edge > 0:
                sampler = qmc.LatinHypercube(d=3, seed=random_seed + cdu_id * 3 + 1)
                params = sampler.random(n_edge)
                for i in range(n_edge):
                    scenarios.append({
                        "cdu_id": cdu_id,
                        "type": "edge",
                        "params": params[i],
                        "power_series": None
                    })
            
            # Fault scenarios
            if n_fault > 0:
                sampler = qmc.LatinHypercube(d=3, seed=random_seed + cdu_id * 3 + 2)
                params = sampler.random(n_fault)
                for i in range(n_fault):
                    scenarios.append({
                        "cdu_id": cdu_id,
                        "type": "fault",
                        "params": params[i],
                        "power_series": None
                    })
            
            # Shuffle scenarios for this CDU with seeded random state
            rng = np.random.RandomState(random_seed + cdu_id * 100)
            rng.shuffle(scenarios)
            
            all_scenarios.extend(scenarios)
        
        return all_scenarios


def generate_scenario_lhs(
    n_scenarios: int = 10,
    n_cdus: int = 51,
    distribution: List[float] = [0.5, 0.4, 0.1],
    seed: int = 42,
) -> List[Dict]:
    """
    Convenience function to generate scenarios
    
    Parameters:
    -----------
    n_scenarios : int
        Total number of scenarios to generate
    n_cdus : int
        Number of CDUs
    base_power : float
        Base power level in kW
    distribution : List[float]
        Distribution of [normal, edge, fault] scenarios
    seed : int
        Random seed
    power_series : bool
        Whether to generate power series data
    
    Returns:
    --------
    List[Dict] : List of scenario dictionaries
    """
    generator = ScenarioGenerator(seed=seed)
    return generator.generate_scenario_lhs(
        n_scenarios=n_scenarios,
        n_cdus=n_cdus,
        distribution=distribution,
        seed=seed  # Pass seed explicitly
    )