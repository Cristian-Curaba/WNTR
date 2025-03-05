#!/usr/bin/env python
"""
some_testing.py
"""

import os

from scipy.sparse.linalg import spsolve
from typing import List, Dict, Set, Union
import warnings
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import wntr

import networkx as nx
import numpy as np
import pandas as pd

def load_network_from_inp(inp_path):
    """
    Loads an EPANET INP file and returns a WaterNetworkModel instance.
    """
    print(f"Loading INP file: {inp_path}")
    wn = wntr.network.WaterNetworkModel(inp_path)
    print("...done.\n")
    return wn

def print_basic_network_info(wn):
    """
    Print basic information about the water network model.
    """
    print("=== Basic Network Info ===")
    print(f"Number of junctions: {len(wn.junction_name_list)}")
    print(f"Number of reservoirs: {len(wn.reservoir_name_list)}")
    print(f"Number of tanks: {len(wn.tank_name_list)}")
    print(f"Number of pipes: {len(wn.pipe_name_list)}")
    print(f"Number of valves: {len(wn.valve_name_list)}")
    print(f"Number of pumps: {len(wn.pump_name_list)}")
    print("")

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

def analyze_resilience_metrics(wn, results):
    """
    Compute the Todini resilience index using hydraulic simulation results.
    """
    print("=== Resilience Metrics: Todini Index ===")

    # Extract necessary data from simulation results
    head = results.node["head"]
    pressure = results.node["pressure"]
    demand = results.node["demand"]
    flowrate = results.link["flowrate"]

    # Define a required pressure threshold (Pstar)
    Pstar = 15  # Adjust this based on your system requirements

    # Compute Todini Index
    try:
        todini_index = wntr.metrics.hydraulic.todini_index(head, pressure, demand, flowrate, wn, Pstar)
        print(f"Todini Resilience Index: {todini_index.mean()} (average over all timesteps)")
    except Exception as e:
        print(f"Error computing Todini index: {e}")

    print("")

def run_water_quality_sim(wn):
    """
    Run a water quality simulation if the INP file has WQ data configured.
    This might fail if your INP doesn't have water quality parameters set.
    """
    print("=== Water Quality Simulation (EPANET) ===")
    # Make sure to set wn.options.quality for a relevant WQ analysis type (e.g. 'CHEMICAL')
    wn.options.quality.mode = 'CHEMICAL'
    wn.options.quality.species = 'chlorine'  # or your chemical name
    # For a real simulation, you might need to set initial conditions, source quality, etc.
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    print("Water quality simulation complete.\n")

    # Suppose we check concentration at a node:
    node_name = '101'
    if node_name in results.node['quality'].columns:
        wq_series = results.node['quality'][node_name]
        print(f"Water quality at node {node_name}:\n{wq_series.head()}")
    else:
        print(f"No water quality data for node {node_name}.")
    print("")

def demonstrate_morphological_skeletonization(wn, output_inp='skeletonized.inp'):
    """
    Example morphological transformation: skeletonization (removing minor pipes).
    If the skeletonization is too aggressive or doesn't match your network, adjust thresholds.
    """
    from wntr.morph import skeletonize

    print("=== Morphological Transformation: Skeletonization ===")
    print("Original number of pipes:", len(wn.pipe_name_list))

    # Skeletonize with a user-defined diameter threshold, for example
    # (Any pipe with a diameter below threshold is considered minor.)
    # Adjust threshold as needed for your network
    threshold = 10  # units consistent with your INP file
    wn_skel = skeletonize(wn, diameter_threshold=threshold)

    print("After skeletonization, number of pipes:", len(wn_skel.pipe_name_list))

    # Write out to a new INP file (optional)
    wn_skel.write_inpfile(output_inp, units=wn.options.hydraulics.units)
    print(f"Skeletonized network saved to {output_inp}\n")

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

def calculate_hitting_time_directed(G, source, target):
    """
    Calculates the hitting time in a directed graph using a random-walk transition matrix.
    Returns float('inf') if the target is not reachable from the source.
    """

    # If source == target, hitting time is 0
    if source == target:
        return 0.0

    # Sort nodes for consistent indexing
    nodes = sorted(G.nodes())
    n = len(nodes)

    # Create adjacency matrix in sorted order of nodes
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)

    # Construct transition probability matrix P
    # P[i, j] = probability of going from node i to node j
    # i.e., each row i of A sums to out-degree of node i
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_sum = A[i].sum()
        if row_sum > 0:
            P[i, :] = A[i, :] / row_sum

    # Get the numeric indices of source and target
    try:
        s = nodes.index(source)
        t = nodes.index(target)
    except ValueError:
        # If source or target is not in the graph
        return float('inf')

    # If target is unreachable (no transitions from s), return inf
    # Quick check: if the entire row in P for s is zero, no outflow
    if np.allclose(P[s, :], 0.0):
        return float('inf')

    # We use an approach for absorbing Markov chains:
    #   - Treat 'target' as absorbing, then solve the system (I - Q)h = 1
    #   - Q is the transition matrix P with the target row/column removed
    #   - h[s] is the expected hitting time from s to t
    #
    # Step 1: remove target row and column from P
    Q = np.delete(np.delete(P, t, axis=0), t, axis=1)
    # Adjust indices if s > t (because of removal)
    if s > t:
        s -= 1

    # Number of non-absorbing states
    n_sub = n - 1

    # Step 2: Solve (I - Q) * h = 1
    I = np.eye(n_sub, dtype=float)
    b = np.ones(n_sub, dtype=float)

    try:
        h = np.linalg.solve(I - Q, b)
    except np.linalg.LinAlgError:
        # Matrix might be singular if the node is unreachable or the graph is not strongly connected
        return float('inf')

    # If s is outside range (for safety), hitting time is inf
    if not (0 <= s < n_sub):
        return float('inf')

    return float(h[s])

def calculate_network_metrics(wn):
    """
    Calculate advanced centrality metrics using NetworkX for the water network.
    Returns a dictionary of DataFrames (or Series) containing centrality values for each node.
    """
    print("=== Calculating Network Centralities ===")

    # Get the *directed* NetworkX graph from the WNTR network
    G = wn.get_graph()

    # Also create an undirected copy for metrics that require undirected graphs
    G_undirected = G.to_undirected()

    # Initialize dictionary to store centrality results
    centralities = {}

    # 1. Current-Flow Betweenness Centrality (requires undirected)
    try:
        current_flow_bet = nx.current_flow_betweenness_centrality(G_undirected)
        centralities['current_flow_betweenness'] = pd.Series(current_flow_bet)
        print("✓ Current-Flow Betweenness calculated")
    except Exception as e:
        print(f"Error calculating Current-Flow Betweenness: {e}")

    # 2. Communicability Centrality (requires undirected)
    try:
        communicability = nx.communicability_betweenness_centrality(G_undirected)
        centralities['communicability'] = pd.Series(communicability)
        print("✓ Communicability Centrality calculated")
    except Exception as e:
        print(f"Error calculating Communicability: {e}")

    # 3. Katz Centrality (works on directed or undirected, but we'll use the directed graph)
    try:
        # alpha should be less than the reciprocal of the largest eigenvalue
        try:
            alpha = 1 / max(abs(nx.adjacency_spectrum(G)))
        except:
            alpha = 0.1
        katz = nx.katz_centrality(G, alpha=alpha)
        centralities['katz'] = pd.Series(katz)
        print("✓ Katz Centrality calculated")
    except Exception as e:
        print(f"Error calculating Katz Centrality: {e}")

    # 4. Minimum Hitting Time from tanks/reservoirs (using the directed graph)
    # ------------------------------------------------
    # We replace the old 'nx.hitting_time' with our custom function
    # ------------------------------------------------
    sources = []
    for tank_name, tank in wn.tanks():
        sources.append(tank_name)
    for reservoir_name, reservoir in wn.reservoirs():
        sources.append(reservoir_name)

    min_hitting_times = {}
    print(f"Calculating hitting times in the directed graph from {len(sources)} sources...")

    # Compute minimum hitting time for each node from any of the source nodes
    for target in G.nodes():
        if target in sources:
            # If the node itself is a tank or reservoir, set hitting time to 0
            min_hitting_times[target] = 0.0
            continue

        hitting_times_to_target = []
        for source in sources:
            try:
                hit_time = calculate_hitting_time_directed(G, source, target)
                if not np.isinf(hit_time):
                    hitting_times_to_target.append(hit_time)
            except Exception as e:
                print(f"Warning: Could not calculate hitting time from {source} to {target}: {e}")

        if hitting_times_to_target:
            min_hitting_times[target] = min(hitting_times_to_target)
        else:
            min_hitting_times[target] = float('inf')

    min_hitting_times_series = pd.Series(min_hitting_times)
    # Store in centralities
    centralities['min_hitting_time'] = min_hitting_times_series

    # Debug information for hitting times
    print("\nDebug Information (Hitting Times):")
    print(f"Number of nodes processed: {len(min_hitting_times_series)}")
    print(f"Number of infinite values: {sum(np.isinf(min_hitting_times_series))}")
    print("\nSample of minimum hitting times:")
    print(min_hitting_times_series.head())

    # 5. Vitality Centrality (using closeness in the undirected graph)
    try:
        vitality = {}
        base_closeness = sum(nx.closeness_centrality(G_undirected).values())
        for node in G.nodes():
            H = G_undirected.copy()
            H.remove_node(node)
            if len(H) > 0:  # Check if graph is not empty
                new_closeness = sum(nx.closeness_centrality(H).values())
                vitality[node] = base_closeness - new_closeness
            else:
                vitality[node] = base_closeness
        centralities['vitality'] = pd.Series(vitality)
        print("✓ Vitality Centrality calculated")
    except Exception as e:
        print(f"Error calculating Vitality: {e}")

    # 6. Random Walk Network Resilience (Spectral Gap) on undirected graph
    try:
        L = nx.normalized_laplacian_matrix(G_undirected)
        eigenvalues = np.linalg.eigvals(L.toarray())
        spectral_gap = np.sort(np.abs(eigenvalues))[1]  # 2nd smallest eigenvalue

        spectral_centrality = {}
        for node in G.nodes():
            H = G_undirected.copy()
            H.remove_node(node)
            if len(H) > 0:
                L_new = nx.normalized_laplacian_matrix(H)
                eigenvalues_new = np.linalg.eigvals(L_new.toarray())
                spectral_gap_new = np.sort(np.abs(eigenvalues_new))[1]
                spectral_centrality[node] = spectral_gap - spectral_gap_new
            else:
                spectral_centrality[node] = spectral_gap
        centralities['spectral_gap'] = pd.Series(spectral_centrality)
        print("✓ Spectral Gap Analysis calculated")
    except Exception as e:
        print(f"Error calculating Spectral Gap: {e}")

    # 7. Modularity-Based Influence (undirected)
    try:
        communities = nx.community.greedy_modularity_communities(G_undirected)
        modularity_influence = {}
        for node in G.nodes():
            # Find which community the node belongs to
            node_community = None
            for i, community in enumerate(communities):
                if node in community:
                    node_community = i
                    break
            if node_community is not None:
                internal_connections = sum(
                    1 for neighbor in G.neighbors(node)
                    if neighbor in communities[node_community]
                )
                external_connections = G.degree(node) - internal_connections
                modularity_influence[node] = internal_connections / (external_connections + 1)
            else:
                modularity_influence[node] = 0
        centralities['modularity_influence'] = pd.Series(modularity_influence)
        print("✓ Modularity-Based Influence calculated")
    except Exception as e:
        print(f"Error calculating Modularity Influence: {e}")

    # --------------  Custom Betweenness --------------
    # Now compute (or get) your sources from the network
    sources = get_source_nodes(wn, G)
    print(f"Sources identified: {sources}")

    # Compute your custom betweenness restricted to those sources
    print("Calculating custom betweenness (subset-based)...")
    custom_bc_series = calculate_custom_betweenness_subset(G, sources)
    centralities['custom_betweenness'] = custom_bc_series

    # ------------- Custom Flow Betweenness ---------
    print("Calculating custom flow betweenness")
    custom_bc_flow_series=multi_source_current_flow_betweenness(G, sources)
    centralities['custom_flow_betweenness']= custom_bc_flow_series


    # === Global Network Metrics (undirected) ===
    print("\n=== Global Network Metrics ===")

    # 1. Meshedness Coefficient
    try:
        n = G_undirected.number_of_nodes()
        m = G_undirected.number_of_edges()
        max_edges = n * (n - 1) / 2
        meshedness = (m - (n - 1)) / (max_edges - (n - 1)) if max_edges > (n - 1) else 0.0
        print(f"✓ Global Meshedness Coefficient: {meshedness:.3f}")
    except Exception as e:
        print(f"Error calculating Meshedness: {e}")

    # 2. Average Degree (directed or undirected; here we use the directed G)
    try:
        avg_degree = sum(dict(G.degree()).values()) / n
        print(f"✓ Average Degree: {avg_degree:.2f}")
    except Exception as e:
        print(f"Error calculating Average Degree: {e}")

    # 3. Network Failure Analysis (directed checks)
    #    If your network is not strictly a DAG, you can use descendants() on a directed graph.
    #    Otherwise, you might want to handle it differently.
    def check_critical_disconnection(graph, source_nodes):
        """
        Check if more than 10% of nodes are disconnected from sources
        using a directed BFS/DFS approach (descendants).
        """
        reachable_nodes = set()
        for s_node in source_nodes:
            if s_node in graph:
                reachable_nodes.update(nx.descendants(graph, s_node))
                reachable_nodes.add(s_node)
        return len(reachable_nodes) < 0.9 * graph.number_of_nodes()

    try:
        source_nodes = wn.tank_name_list + wn.reservoir_name_list

        # Random Failures Analysis
        n_trials = 100
        random_failures = []

        for _ in range(n_trials):
            G_copy = G.copy()
            nodes_removed = 0
            node_list = list(G_copy.nodes())
            np.random.shuffle(node_list)

            for node in node_list:
                if node not in source_nodes:  # Don't remove source nodes
                    G_copy.remove_node(node)
                    nodes_removed += 1
                    if check_critical_disconnection(G_copy, source_nodes):
                        random_failures.append(nodes_removed)
                        break

        avg_random_failures = np.mean(random_failures) if random_failures else 0
        print(f"✓ Average number of random failures before critical disconnection: {avg_random_failures:.1f}")

        # Adversarial Failures Analysis (using betweenness centrality)
        G_copy = G.copy()
        adversarial_failures = 0

        while True:
            if G_copy.number_of_nodes() <= 1:
                break
            btw = nx.betweenness_centrality(G_copy)
            target_node = max(
                (node for node in btw.items() if node[0] not in source_nodes),
                key=lambda x: x[1],
                default=None
            )
            if not target_node:
                break
            G_copy.remove_node(target_node[0])
            adversarial_failures += 1

            if check_critical_disconnection(G_copy, source_nodes):
                break

        print(f"✓ Number of adversarial failures before critical disconnection: {adversarial_failures}")

    except Exception as e:
        print(f"Error calculating failure metrics: {e}")

    # Additional Network Statistics
    print("\n=== Additional Network Statistics ===")
    try:
        print(f"Total Nodes: {G.number_of_nodes()}")
        print(f"Total Edges: {G.number_of_edges()}")
        print(f"Source Nodes (Tanks + Reservoirs): {len(source_nodes)}")
        print(f"Network Diameter (on undirected): {nx.diameter(G_undirected)}")
        try:
            print(f"Average Shortest Path Length (on undirected): "
                  f"{nx.average_shortest_path_length(G_undirected):.2f}")
        except:
            print("Average Shortest Path Length: N/A (Graph not connected)")
    except Exception as e:
        print(f"Error calculating additional statistics: {e}")

    print("\nMetrics calculation complete.\n")
    return centralities

def demonstrate_graph_visualization(wn, centralities):
    """
    Creates visualizations of the network:
    - Pressure and flowrate visualization (pressure as node color, flowrate as link width)
    - Individual centrality visualizations without hydraulic info
    """
    print("=== Advanced Graph Visualization ===")

    # Create directory for images if it doesn't exist
    network_name = os.path.splitext(os.path.basename(wn.name))[0]
    image_dir = os.path.join('images', network_name)
    os.makedirs(image_dir, exist_ok=True)

    # Run simulation for pressure and flowrate
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Extract node pressure and link flowrate at time = 0
    node_pressure = results.node['pressure'].loc[0]
    link_flowrate = results.link['flowrate'].loc[0]

    # Get the graph and positions
    G = wn.to_graph()
    pos = nx.get_node_attributes(G, 'pos')

    # First, create a mapping from node pairs to link names
    link_mapping = {}
    for link_name, link_obj in wn.links():
        start_node = link_obj.start_node_name
        end_node = link_obj.end_node_name
        link_mapping[(start_node, end_node)] = link_name
        link_mapping[(end_node, start_node)] = link_name  # Add reverse mapping too

    # Calculate link widths
    max_flow = max(abs(flow) for flow in link_flowrate)
    min_width = 1
    max_width = 5

    # Create edge width dictionary
    edge_widths = []
    edges = list(G.edges())  # Get list of edges from graph

    # print("Debug information:")
    # print(f"Number of edges in graph: {len(edges)}")
    # print(f"Number of links in mapping: {len(link_mapping)}")
    # print("\nFirst few links in flowrate data:")
    # for link, flow in list(link_flowrate.items())[:5]:
    #     print(f"Link: {link}, Flow: {flow}")

    for (u, v) in edges:
        if (u, v) in link_mapping:
            link_name = link_mapping[(u, v)]
            # print(f"Edge ({u}, {v}) mapped to link {link_name}")
            if link_name in link_flowrate:
                flow = abs(link_flowrate[link_name])
                width = min_width + (max_width - min_width) * (flow / max_flow)
                #print(f"  Flow: {flow}, Width: {width}")
                edge_widths.append(width)
            else:
                print(f"  No flow data for link {link_name}")
                edge_widths.append(min_width)
        else:
            print(f"Edge ({u}, {v}) not found in link mapping")
            edge_widths.append(min_width)

    # print("\nResulting edge widths:")
    # print(edge_widths)

    # Define node categories styling
    node_categories = {
        'reservoirs': {'marker': '^', 'color': 'blue', 'size': 120},
        'tanks': {'marker': 's', 'color': 'purple', 'size': 100},
        'valves': {'marker': '*', 'color': 'red', 'size': 100},
        'junctions': {'marker': 'o', 'color': 'black', 'size': 40}
    }

    node_types = {
        'reservoirs': wn.reservoir_name_list,
        'tanks': wn.tank_name_list,
        'valves': wn.valve_name_list,
        'junctions': wn.junction_name_list
    }

    # 1. Create pressure + flowrate visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot nodes with pressure
    node_collection = nx.draw_networkx_nodes(
        G, pos,
        node_color=[node_pressure[node] for node in G.nodes()],
        node_size=40,
        cmap=plt.cm.coolwarm,
        vmin=node_pressure.min(),
        vmax=node_pressure.max(),
        ax=ax
    )

    # Plot edges with varying widths
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.7,
        ax=ax
    )

    # Add colorbar for pressure
    plt.colorbar(node_collection, label='Pressure (m)')

    # Add legend for flowrate scale
    flow_legend_elements = [
        Line2D([0], [0], color='gray', linewidth=min_width, label='Min Flow'),
        Line2D([0], [0], color='gray', linewidth=(min_width + max_width)/2, label='Med Flow'),
        Line2D([0], [0], color='gray', linewidth=max_width, label='Max Flow')
    ]

    # Add node category markers
    for category, node_list in node_types.items():
        if not node_list:
            continue
        x_coords, y_coords = zip(*[pos[node] for node in node_list if node in pos])
        ax.scatter(x_coords, y_coords,
                  marker=node_categories[category]['marker'],
                  c=node_categories[category]['color'],
                  s=node_categories[category]['size'],
                  label=category.capitalize(),
                  edgecolor='k')

    # Add both legends
    node_legend = ax.legend(title="Node Categories", loc="upper right", fontsize=10)
    ax.add_artist(node_legend)
    ax.legend(handles=flow_legend_elements, title="Flow Rate",
             loc="upper left", fontsize=10)

    plt.title("Pressure and Flowrate Distribution")
    filename = os.path.join(image_dir, "pressure_flowrate.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {filename}")
    plt.close()

    # 2. Create individual centrality visualizations
    for centrality_name, centrality_values in centralities.items():
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot nodes with centrality values
        node_collection = nx.draw_networkx_nodes(
            G, pos,
            node_color=[centrality_values[node] for node in G.nodes()],
            node_size=40,
            cmap=plt.cm.viridis,
            vmin=centrality_values.min(),
            vmax=centrality_values.max(),
            ax=ax
        )

        # Plot edges (uniform style)
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.0,
            alpha=0.3,
            ax=ax
        )

        # Add colorbar for centrality
        plt.colorbar(node_collection,
                    label=f"{centrality_name.replace('_', ' ').title()} Value")

        # Add node category markers
        for category, node_list in node_types.items():
            if not node_list:
                continue
            x_coords, y_coords = zip(*[pos[node] for node in node_list if node in pos])
            ax.scatter(x_coords, y_coords,
                      marker=node_categories[category]['marker'],
                      c=node_categories[category]['color'],
                      s=node_categories[category]['size'],
                      label=category.capitalize(),
                      edgecolor='k')

        ax.legend(title="Node Categories", loc="upper right", fontsize=10)
        plt.title(f"{centrality_name.replace('_', ' ').title()} Centrality")

        filename = os.path.join(image_dir, f"{centrality_name}_centrality.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
        plt.close()

    print("All visualizations complete.\n")

def get_source_nodes(wn, G=None):
    """
    Return a set of source nodes for a water network model.
    'Sources' include:
      - Reservoirs
      - Tanks
      - Nodes with in-degree = 0 in the graph (i.e., no inflow)

    Parameters
    ----------
    wn : wntr.network.WaterNetworkModel
        The WNTR water network model.
    G : nx.DiGraph, optional
        The directed NetworkX graph. If not supplied, will be
        obtained automatically via wn.get_graph().

    Returns
    -------
    set
        A set of node labels considered as sources.
    """
    if G is None:
        G = wn.get_graph()  # Retrieve the directed graph if not provided

    # Start by collecting reservoir and tank names
    sources = set(wn.reservoir_name_list).union(wn.tank_name_list)

    # Also add nodes with in-degree = 0
    for node in G.nodes():
        if G.in_degree(node) == 0:
            sources.add(node)

    # Ensure we only include nodes actually present in the graph
    sources = sources.intersection(G.nodes())
    return sources

def calculate_custom_betweenness_subset(G, sources):
    """
    Calculate a custom betweenness metric for directed graph G
    restricted to paths that start at any node in 'sources' and end at
    any node in G.

    Uses NetworkX's betweenness_centrality_subset.
    If your NetworkX version is older and doesn't have this function,
    consider upgrading or implementing a custom approach.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph for the water network.
    sources : collection
        The set (or list) of nodes to use as path sources.

    Returns
    -------
    pd.Series
        A Series mapping node -> custom betweenness value.
    """
    # Convert to lists if needed and filter out any nodes not in G
    valid_sources = [s for s in sources if s in G.nodes()]

    if not valid_sources:
        # If there are no valid sources, return a zero-valued Series for each node
        return pd.Series(data=0.0, index=G.nodes(), name="custom_betweenness")

    try:
        bc_dict = nx.betweenness_centrality_subset(
            G,
            sources=valid_sources,
            targets=G.nodes(),
            normalized=False,  # or True if you want normalized betweenness
            weight=None        # or set to the relevant edge attribute
        )
        return pd.Series(bc_dict, name="custom_betweenness")

    except AttributeError:
        # If betweenness_centrality_subset is not available in older NetworkX
        msg = (
            "Your NetworkX version does not have betweenness_centrality_subset.\n"
            "Try upgrading (pip install --upgrade networkx) or implement a manual approach."
        )
        print(msg)
        # Return zero for all nodes in G to avoid crashing
        return pd.Series(data=0.0, index=G.nodes(), name="custom_betweenness")

def multi_source_current_flow_betweenness(
        G: nx.DiGraph,
        sources: Union[List, Set],
        weight: str = None,
        normalized: bool = True
) -> pd.Series:
    """
    Compute current flow betweenness centrality for multiple source nodes in a directed graph.

    Parameters:
    -----------
    G : NetworkX directed graph (DiGraph)
        The input directed graph
    sources : list or set
        The source nodes from which flows originate
    weight : str, optional
        Edge weight attribute name
    normalized : bool, optional
        If True, normalize the centrality values

    Returns:
    --------
    pd.Series
        Series indexed by nodes with their multi-source current flow betweenness centrality
    """
    sources = set(sources)
    if not sources.issubset(G.nodes()):
        raise ValueError("All source nodes must be in the graph")

    # Initialize centrality dictionary with zeros
    centrality = {node: 0.0 for node in G.nodes()}

    # Process each source separately with its reachable nodes
    for source in sources:
        # Get the subgraph of nodes reachable from this source
        reachable = set(nx.descendants(G, source)) | {source}
        if len(reachable) <= 1:
            continue

        # Create subgraph of reachable nodes
        subgraph = G.subgraph(reachable).copy()

        # Create integer mapping for matrix operations
        mapping = {node: i for i, node in enumerate(subgraph.nodes())}
        reverse_mapping = {i: node for node, i in mapping.items()}
        H = nx.relabel_nodes(subgraph, mapping)

        # Get source index in the mapped graph
        source_idx = mapping[source]
        n = H.number_of_nodes()

        try:
            # Compute the Laplacian matrix for the directed graph
            # For directed graphs, we use the asymmetric Laplacian
            L = nx.directed_laplacian_matrix(H, weight=weight).tocsc()

            # Initialize current vector (b)
            b = np.zeros(n)
            b[source_idx] = 1.0

            # Solve for potentials
            try:
                potentials = spsolve(L, b)

                # Compute and accumulate current flows
                for u, v in H.edges():
                    u_idx, v_idx = u, v  # Already mapped to integers
                    weight_uv = H[u][v].get(weight, 1.0)
                    current_flow = weight_uv * abs(potentials[u_idx] - potentials[v_idx])

                    # Add flow contributions
                    centrality[reverse_mapping[u_idx]] += current_flow
                    centrality[reverse_mapping[v_idx]] += current_flow

            except Exception as e:
                warnings.warn(f"Error solving linear system for source {source}: {str(e)}")
                continue

        except Exception as e:
            warnings.warn(f"Error processing source {source}: {str(e)}")
            continue

    # Normalize if requested
    if normalized and sum(centrality.values()) > 0:
        total_flow = sum(centrality.values())
        centrality = {k: v / total_flow for k, v in centrality.items()}

    # Convert to pandas Series
    return pd.Series(centrality, name='custom_flow_betweenness')

def analyze_centrality_distribution(centrality: Dict) -> Dict:
    """
    Analyze the distribution of centrality values.

    Parameters:
    -----------
    centrality : Dict
        Dictionary of centrality values

    Returns:
    --------
    Dict
        Statistical summary of centrality values
    """
    values = list(centrality.values())
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': min(values),
        'max': max(values),
        'top_nodes': sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    }

def main():
    # Path to an example INP file. Adjust as needed.
    inp_file = os.path.join('examples', 'networks', 'Net2.inp')

    # 1. Load Network
    wn = load_network_from_inp(inp_file)

    # 2. Print Basic Info
    print_basic_network_info(wn)

    # 3. EPANET Simulation
    epanet_results = run_epanet_simulation(wn)
    # analyze_hydraulic_results(epanet_results, node_name='1')  # Replace with valid node

    # 4. WNTR Simulation
    # wntr_results = run_wntr_simulation(wn)
    # analyze_hydraulic_results(wntr_results, node_name='1')

    # 5. Resilience Metrics
    # analyze_resilience_metrics(wn, wntr_results)

    # 6. (Optional) Water Quality Simulation
    # Comment out if your INP file is not set up for WQ
    # run_water_quality_sim(wn)

    # 7. Morphological transformations (skeletonization)
    # If you want to preserve your original network object, copy it first
    # demonstrate_morphological_skeletonization(wn)

    # 8. Scenario-based analysis (pipe break, demand changes, etc.)
    demonstrate_scenario_based_analysis(wn)
    demonstrate_demand_change(wn)

    # 9. Graph Visualization
    centralities = calculate_network_metrics(wn)
    demonstrate_graph_visualization(wn, centralities)

    print("\n=== Finished all demonstrations! ===")

if __name__ == "__main__":
    main()
