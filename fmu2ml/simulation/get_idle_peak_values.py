import pandas as pd
from raps.config import ConfigManager
from raps.power import PowerManager, compute_node_power
from raps.flops import FLOPSManager
from raps.workload import Workload
from raps.account import Accounts
from raps.engine import Engine 
from raps.ui import LayoutManager
from types import SimpleNamespace


def run_simulation_and_extract_power(workload_type,system_name='marconi100', timesteps=3600):
    
    print(f"Setting up {workload_type} simulation...")
    
    # Create args namespace to mimic command-line arguments
    args = SimpleNamespace(
        system=system_name,
        workload=workload_type,
        cooling=False,  # We don't need cooling model for this analysis
        output=False,   # No need to save files
        plot=[],
        numjobs=2000,    # Enough jobs to cover all nodes
        layout="layout1",
        policy='fcfs',
    )
    
    args_dict = vars(args)
    
    # Get System configuration
    config = ConfigManager(system_name=system_name).get_config()
    args_dict['config'] = config
    
    # Set up power manager
    power_manager = PowerManager(compute_node_power, **config)
    
    # Set up FLOPS manager
    flops_manager = FLOPSManager(**args_dict)
    
    # Create engine
    engine = Engine(
        power_manager=power_manager,
        flops_manager=flops_manager,
        cooling_model=None,
        **args_dict
    )
    
    # Create layout manager
    layout_manager = LayoutManager(args.layout, engine=engine, debug=False, **config)
    
    # Create workload
    wl = Workload(config)
    jobs = getattr(wl, workload_type)(num_jobs=args.numjobs)
    
    # Create accounts
    accounts = Accounts(jobs)
    engine.accounts = accounts
    
    # Run simulation
    print(f"Running {workload_type} simulation for {timesteps} seconds...")
    layout_manager.run(jobs, timesteps=timesteps)
    
    # Get power statistics
    stats = engine.get_stats()
    total_power = float(stats.get('average power').split(" ")[0])*1000
    
    # Extract rack power information directly
    print(f"Extracting rack power for {workload_type} workload...")
    
    # Get the rack power values
    rack_power, rect_losses = power_manager.compute_rack_power()
    
    # Convert to DataFrame
    columns = ["CDU"]
    for i in range(1, config['RACKS_PER_CDU'] + 1):
        columns.append(f"Rack_{i}")
    columns.append("Total")
    
    rack_powers_df = pd.DataFrame(rack_power, columns=columns)
    
    return rack_powers_df, total_power, engine.power_manager.history[-1][1]
