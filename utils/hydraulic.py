"""
utils/hydraulic.py
"""
import wntr

def demonstrate_scenario_based_analysis(wn):
    """
    Demonstrate how to simulate a scenario-based approach.
    For example, break a pipe (set it to 'closed') and observe the impact on the system.
    """
    print("=== Scenario-Based Analysis: Pipe Break ===")

    # 1) Pick a pipe to break
    pipe_to_break = wn.pipe_name_list[0]  # Choose first pipe in list (adjust as needed)
    original_status = wn.get_link(pipe_to_break).status

    # 2) Break the pipe (set it to CLOSED using user_status)
    wn.get_link(pipe_to_break).user_status = wntr.network.LinkStatus.Closed


    # 3) Re-run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # 4) Evaluate performance impact (e.g., average system pressure)
    avg_pressure = results.node['pressure'].mean(axis=1).mean()
    print(f"Scenario: Pipe {pipe_to_break} closed => Average pressure = {avg_pressure:.2f} m")

    # 5) Restore original status
    wn.get_link(pipe_to_break).user_status = wntr.network.LinkStatus.Opened
    print(f"Restored original status of pipe {pipe_to_break}.\n")

def demonstrate_demand_change(wn):
    """
    Another scenario example: Increase all demands by 20%.
    Then compare average pressure to baseline.
    """
    print("=== Scenario-Based Analysis: Demand Change ===")
    # Store original demands
    original_demands = {}
    for junction_name, junction in wn.junctions():
        original_demands[junction_name] = junction.demand_timeseries_list[0].base_value

    # Increase demands by 20%
    for junction_name, junction in wn.junctions():
        junction.demand_timeseries_list[0].base_value *= 1.2

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Evaluate new average pressure
    avg_pressure = results.node['pressure'].mean(axis=1).mean()
    print(f"After 20% demand increase => average pressure = {avg_pressure}")

    # Restore original demands
    for junction_name, junction in wn.junctions():
        junction.demand_timeseries_list[0].base_value = original_demands[junction_name]
    print("Demand change scenario complete.\n")

def run_epanet_simulation(wn):
    """
    Run a hydraulic simulation using the EpanetSimulator.
    Returns the wntr.sim.results.SimulationResults object.
    """
    print("=== EPANET Simulation ===")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    print("Epanet simulation complete.\n")
    return results

def analyze_hydraulic_results(results, node_name='101'):
    """
    Example: Access pressures for a specific node and
    print the first few rows of the time series.
    """
    # '101' is just an example node that may exist in some test networks.
    # Adjust to a real node name in your INP file if needed.
    print("=== Hydraulic Results Analysis ===")
    if node_name in results.node['pressure'].columns:
        pressure_series = results.node['pressure'][node_name]
        print(f"Pressure at node {node_name}:")
        print(pressure_series.head())
    else:
        print(f"Node {node_name} not found in pressure results.\n"
              f"Available nodes:\n{results.node['pressure'].columns.tolist()}")
    print("")

def run_wntr_simulation(wn):
    """
    Run a hydraulic simulation using the WNTRSimulator (Python-based).
    This can capture pressure-dependent demands, leaks, etc.
    """
    print("=== WNTR Simulation ===")
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    print("WNTR simulation complete.\n")
    return results