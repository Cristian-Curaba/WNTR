# emergency_pre_analysis/utils.py
import time
from collections import defaultdict
from functools import wraps
from typing import Set

import networkx as nx
import numpy as np
import pandas as pd
import os


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time( )
        result = func(*args, **kwargs)
        end_time = time.time( )
        execution_time = end_time - start_time
        # logging.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result

    return wrapper


def get_source_nodes(wn) -> tuple:
    """
    Get source nodes and atypical nodes from a water network.
    Source nodes: tanks, reservoirs, and nodes with in-degree=0 and out-degree=1
    Atypical nodes: nodes with in-degree=0 and out-degree>1

    Parameters:
    -----------
    wn : WNTR WaterNetworkModel
        The water network model

    Returns:
    --------
    tuple
        (list of source node IDs, list of atypical node IDs)
    """
    sources = []
    atypical_nodes = []

    # Get tanks
    for tank_name, tank in wn.tanks( ):
        sources.append(tank_name)

    # Get reservoirs
    for reservoir_name, reservoir in wn.reservoirs( ):
        sources.append(reservoir_name)

    # Get the network graph
    G = wn.get_graph( )

    for node_name in wn.junction_name_list:
        # Skip if already identified as a source (tank or reservoir)
        if node_name in sources:
            continue

        # Check node degrees
        in_degree = G.in_degree(node_name)
        out_degree = G.out_degree(node_name)

        if in_degree == 0:
            if out_degree == 1:
                sources.append(node_name)
            elif out_degree > 1:
                atypical_nodes.append(node_name)

    print(f"Identified {len(atypical_nodes)} atypical nodes: {atypical_nodes}")

    return sources, atypical_nodes


def convert_multidigraph_to_digraph(multi_G, weight_attr = 'weight', aggregation = 'sum'):
    """
    Convert a NetworkX MultiDiGraph to a DiGraph by aggregating parallel edges.

    Parameters:
    -----------
    multi_G : NetworkX MultiDiGraph
        Input multigraph to convert
    weight_attr : str, default='weight'
        Edge attribute to use as weight
    aggregation : str, default='sum'
        Method to aggregate parallel edge weights: 'sum', 'max', 'min', or 'mean'

    Returns:
    --------
    NetworkX DiGraph
        Simple directed graph with aggregated edge weights
    """
    import networkx as nx
    import numpy as np

    G = nx.DiGraph( )
    G.add_nodes_from(multi_G.nodes(data=True))

    # Group edges by their endpoints and aggregate weights
    edge_weights = {}
    edge_data = {}

    for u, v, data in multi_G.edges(data=True):
        weight = data.get(weight_attr, 1.0)
        if (u, v) not in edge_weights:
            edge_weights[(u, v)] = [weight]
            edge_data[(u, v)] = {k: v for k, v in data.items( ) if k != weight_attr}
        else:
            edge_weights[(u, v)].append(weight)

    # Aggregation functions
    agg_funcs = {'sum': sum, 'max': max, 'min': min, 'mean': np.mean}

    if aggregation not in agg_funcs:
        raise ValueError(f"Aggregation must be one of {list(agg_funcs.keys( ))}")

    # Add edges with aggregated weights
    for (u, v), weights in edge_weights.items( ):
        data = edge_data[(u, v)].copy( )
        data[weight_attr] = agg_funcs[aggregation](weights)
        G.add_edge(u, v, **data)

    return G


def check_unreachable_nodes(G, sources):
    reachable_nodes = set( )
    for source in sources:
        for target in G.nodes( ):
            if nx.has_path(G, source, target):
                reachable_nodes.add(target)

    # Print if there's at least one node that is not reachable from any source
    if len(reachable_nodes) < len(G.nodes( )):
        print("Warning: There are nodes in the graph that are not reachable from any source.")
        unreachable_nodes = set(G.nodes( )) - reachable_nodes
        print(f"Unreachable nodes: {unreachable_nodes}")
        return unreachable_nodes

    return None  # Return None if all nodes are reachable


def estimate_node_pressures(wn, source_pressures):
    """
    Estimate pressures at all nodes using a naive hydraulic approach.
    Handles all node types (Junction, Tank, Reservoir) and link types (Pipe, Pump, Valve).

    Parameters:
    -----------
    wn : WNTR WaterNetworkModel
        The water network model
    source_pressures : dict
        Dictionary mapping source node IDs to their known/assumed pressures

    Returns:
    --------
    dict
        Dictionary of node IDs -> estimated pressures
    """
    import numpy as np
    from collections import deque

    def get_node_elevation(node):
        """Helper function to safely get node elevation."""
        if node.node_type.lower( ) == 'reservoir':
            return node.base_head
        elif node.node_type.lower( ) == 'tank':
            return node.elevation + node.init_level
        else:  # Junction
            return node.elevation

    # Initialize the pressures dictionary with source node pressures
    pressures = dict(source_pressures)

    # Initialize a queue for BFS with the source nodes
    queue = deque(source_pressures.keys( ))
    visited = set(source_pressures.keys( ))

    # Simple BFS to propagate approximate pressures
    while queue:
        current_node = queue.popleft( )

        # Get outgoing edges from current_node
        try:
            out_edges = [e for e in wn.get_links_for_node(current_node) if
                         wn.get_link(e).start_node_name == current_node]
        except Exception as e:
            print(f"Warning: Error getting links for node {current_node}: {str(e)}")
            continue

        for link_name in out_edges:
            try:
                link = wn.get_link(link_name)
                neighbor_node = link.end_node_name

                if neighbor_node not in visited:
                    pressure_drop = 0  # Default pressure drop

                    # Handle different link types
                    link_type = link.link_type.lower( )

                    if link_type == 'pipe':
                        # For pipes, use length and diameter for pressure drop
                        length = getattr(link, 'length', 100)
                        diameter = getattr(link, 'diameter', 0.3)
                        pressure_drop = 0.1 * length / (diameter ** 4.87)

                    elif link_type == 'pump':
                        # For pumps, assume they add head (negative pressure drop)
                        if hasattr(link, 'pump_curve') and link.pump_curve is not None:
                            try:
                                pressure_drop = -link.pump_curve.nominal_head
                            except:
                                pressure_drop = -20  # default pump head gain
                        else:
                            pressure_drop = -20  # default pump head gain

                    elif link_type in ['valve', 'prv', 'psv', 'pbv', 'fcv', 'tcv', 'gpv']:
                        # For valves, use setting if available, otherwise assume minimal drop
                        if hasattr(link, 'setting') and link.setting is not None:
                            pressure_drop = link.setting
                        else:
                            pressure_drop = 1

                    # Get node objects
                    current_node_obj = wn.get_node(current_node)
                    neighbor_node_obj = wn.get_node(neighbor_node)

                    # Safely get elevations
                    try:
                        current_elev = get_node_elevation(current_node_obj)
                        neighbor_elev = get_node_elevation(neighbor_node_obj)
                        elev_diff = neighbor_elev - current_elev
                    except Exception as e:
                        print(
                            f"Warning: Error calculating elevation difference for {current_node}->{neighbor_node}: {str(e)}")
                        elev_diff = 0

                    # Calculate new pressure
                    try:
                        pressures[neighbor_node] = pressures[current_node] - pressure_drop - elev_diff
                    except Exception as e:
                        print(f"Warning: Error calculating pressure for node {neighbor_node}: {str(e)}")
                        pressures[neighbor_node] = pressures[current_node]  # fallback

                    queue.append(neighbor_node)
                    visited.add(neighbor_node)

            except Exception as e:
                print(f"Warning: Error processing link {link_name}: {str(e)}")
                continue

    # Handle unvisited nodes
    for node_id in wn.node_name_list:
        if node_id not in pressures:
            try:
                # For unvisited nodes, use average of neighbors or default
                neighbor_pressures = []
                for link_name in wn.get_links_for_node(node_id):
                    link = wn.get_link(link_name)
                    other_node = (link.start_node_name if link.end_node_name == node_id else link.end_node_name)
                    if other_node in pressures:
                        neighbor_pressures.append(pressures[other_node])

                if neighbor_pressures:
                    pressures[node_id] = np.mean(neighbor_pressures)
                else:
                    # If no neighbor pressures, use node elevation + default pressure
                    node = wn.get_node(node_id)
                    base_pressure = 20.0  # default pressure head
                    pressures[node_id] = get_node_elevation(node) + base_pressure

            except Exception as e:
                print(f"Warning: Error handling unvisited node {node_id}: {str(e)}")
                pressures[node_id] = 20.0  # fallback pressure

    return pressures


# def reorient_edges_by_pressure(wn, source_pressures = None, pressure = None, verbose = True):
#     """
#     Reorient edges in a water network based on pressure-based heuristics,
#     modifying the water network model (wn) in place.
#
#     Parameters:
#     -----------
#     wn : WNTR WaterNetworkModel
#         The water network model (will be modified in place)
#     source_pressures : dict, optional
#         Dictionary mapping source node IDs to their pressures
#     pressure : dict, optional
#         Dictionary mapping all node IDs to their pressures. If provided,
#         this will be used directly instead of estimating pressures
#     verbose : bool, optional
#         Whether to print additional information
#
#     Returns:
#     --------
#     tuple
#         (number_reoriented, remaining_atypical_nodes)
#     """
#     # Get initial sources and atypical nodes
#     sources, atypical_nodes = get_source_nodes(wn)
#
#     if verbose:
#         print(f"Initial sources: {sources}")
#         print(f"Initial atypical nodes: {atypical_nodes}")
#
#     # Determine node pressures
#     if pressure is not None:
#         node_pressures = pressure
#     else:
#         # Initialize source pressures if not provided
#         if source_pressures is None:
#             source_pressures = {}
#             default_pressure = 30  # Default additional head (m)
#
#             for node_id in sources:
#                 try:
#                     node = wn.get_node(node_id)
#                     node_type = node.node_type.lower( )
#
#                     if node_type == 'tank':
#                         source_pressures[node_id] = node.elevation + node.init_level
#                     elif node_type == 'reservoir':
#                         source_pressures[node_id] = node.base_head
#                     else:  # Junction
#                         source_pressures[node_id] = node.elevation + default_pressure
#                 except Exception as e:
#                     print(f"Warning: Error setting source pressure for {node_id}: {str(e)}")
#                     source_pressures[node_id] = default_pressure
#
#         # Estimate pressures
#         node_pressures = estimate_node_pressures(wn, source_pressures)
#
#     # Store pipes to reorient
#     pipes_to_reorient = []
#
#     # Process atypical nodes first
#     for atypical_node in atypical_nodes:
#         try:
#             outgoing_links = [link_name for link_name in wn.get_links_for_node(atypical_node) if
#                               wn.get_link(link_name).start_node_name == atypical_node]
#
#             for link_name in outgoing_links:
#                 link = wn.get_link(link_name)
#
#                 # Only reorient pipes (not pumps or valves)
#                 if link.link_type.lower( ) != 'pipe':
#                     continue
#
#                 start_node = link.start_node_name
#                 end_node = link.end_node_name
#
#                 if node_pressures.get(end_node, 0) > node_pressures.get(start_node, 0):
#                     if verbose:
#                         print(f"Reversing pipe {link_name}: {start_node} -> {end_node} to {end_node} -> {start_node}")
#                         print(f"  Pressure at {start_node}: {node_pressures.get(start_node, 'unknown')}")
#                         print(f"  Pressure at {end_node}: {node_pressures.get(end_node, 'unknown')}")
#
#                     pipes_to_reorient.append((link_name, end_node, start_node))
#
#         except Exception as e:
#             print(f"Warning: Error processing atypical node {atypical_node}: {str(e)}")
#             continue
#
#     # Check remaining pipes
#     for link_name, link in wn.links( ):
#         try:
#             if link.link_type.lower( ) != 'pipe':
#                 continue
#
#             start_node = link.start_node_name
#             end_node = link.end_node_name
#
#             if node_pressures.get(end_node, 0) > node_pressures.get(start_node, 0):
#                 if verbose:
#                     print(f"Reversing pipe {link_name}: {start_node} -> {end_node} to {end_node} -> {start_node}")
#                     print(f"  Pressure at {start_node}: {node_pressures.get(start_node, 'unknown')}")
#                     print(f"  Pressure at {end_node}: {node_pressures.get(end_node, 'unknown')}")
#
#                 pipes_to_reorient.append((link_name, end_node, start_node))
#
#         except Exception as e:
#             print(f"Warning: Error processing link {link_name}: {str(e)}")
#             continue
#
#     # Now reorient all collected pipes
#     reoriented_count = 0
#     for pipe_name, new_start, new_end in pipes_to_reorient:
#         try:
#             reorient_pipe(wn, pipe_name, new_start, new_end)
#             reoriented_count += 1
#         except Exception as e:
#             print(f"Warning: Failed to reorient pipe {pipe_name}: {str(e)}")
#
#     # Check remaining atypical nodes
#     _, remaining_atypical = get_source_nodes(wn)
#
#     if verbose:
#         print(f"\nReoriented {reoriented_count} edges")
#         print(f"Remaining atypical nodes: {len(remaining_atypical)} -> {remaining_atypical}")
#
#     return reoriented_count, remaining_atypical


def reorient_edges_by_flow(wn, flows, verbose=True):
    """
    Reorient edges in a water network based on flow direction,
    modifying the water network model (wn) in place.

    Parameters:
    -----------
    wn : WNTR WaterNetworkModel
        The water network model (will be modified in place)
    flows : dict
        Dictionary mapping link IDs to their flow rates
    verbose : bool, optional
        Whether to print additional information

    Returns:
    --------
    tuple
        (number_reoriented, remaining_atypical_nodes)
    """
    # Get initial sources and atypical nodes
    sources, atypical_nodes = get_source_nodes(wn)

    if verbose:
        print(f"Initial sources: {sources}")
        print(f"Initial atypical nodes: {atypical_nodes}")

    # Store pipes to reorient
    pipes_to_reorient = []

    # Process all pipes
    for link_name, link in wn.links():
        try:
            if link.link_type.lower() != 'pipe':
                continue

            # Get the flow for this pipe
            flow = flows.get(link_name, 0)

            # Negative flow means flow direction is opposite to pipe orientation
            if flow < 0:
                start_node = link.start_node_name
                end_node = link.end_node_name

                if verbose:
                    print(f"Reversing pipe {link_name}: {start_node} -> {end_node} to {end_node} -> {start_node}")
                    print(f"  Flow in pipe: {flow}")

                pipes_to_reorient.append((link_name, end_node, start_node))

        except Exception as e:
            print(f"Warning: Error processing link {link_name}: {str(e)}")
            continue

    # Now reorient all collected pipes
    reoriented_count = 0
    for pipe_name, new_start, new_end in pipes_to_reorient:
        try:
            reorient_pipe(wn, pipe_name, new_start, new_end)
            reoriented_count += 1
        except Exception as e:
            print(f"Warning: Failed to reorient pipe {pipe_name}: {str(e)}")

    # Check remaining atypical nodes
    _, remaining_atypical = get_source_nodes(wn)

    if verbose:
        print(f"\nReoriented {reoriented_count} edges")
        print(f"Remaining atypical nodes: {len(remaining_atypical)} -> {remaining_atypical}")

    return reoriented_count, remaining_atypical


def reorient_pipe(wn, pipe_name, new_start, new_end):
    """
    Safely remove and re-add a pipe to reverse its direction.

    Parameters:
    -----------
    wn : WNTR WaterNetworkModel
        The water network model
    pipe_name : str
        Name of the pipe to reorient
    new_start : str
        New start node name
    new_end : str
        New end node name
    """
    try:
        # Get the original pipe
        pipe = wn.get_link(pipe_name)
        if pipe is None:
            raise ValueError(f"Pipe {pipe_name} not found in network")

        # Store all pipe properties with default values
        length = getattr(pipe, 'length', 100)  # default length of 100
        diameter = getattr(pipe, 'diameter', 0.3048)  # default diameter of 12 inches in meters
        roughness = getattr(pipe, 'roughness', 100)  # default roughness
        minor_loss = getattr(pipe, 'minor_loss', 0.0)
        check_valve = getattr(pipe, 'check_valve_flag', False)

        # Remove the original pipe
        wn.remove_link(pipe_name)

        # Add the new pipe with reversed direction using the public API
        wn.add_pipe(pipe_name, new_start, new_end, length=length, diameter=diameter, roughness=roughness,
                    minor_loss=minor_loss, check_valve=check_valve)

        # Verify the pipe was added successfully
        new_pipe = wn.get_link(pipe_name)
        if new_pipe is None:
            raise ValueError(f"Failed to add reversed pipe {pipe_name}")

    except Exception as e:
        print(f"Error in reorient_pipe for {pipe_name}: {str(e)}")
        raise


def check_and_transform_to_directed_acyclic_graph(G, sources):
    """
    Check if a graph is a DAG and transform it if necessary by reversing edges in cycles.
    Uses a heuristic based on distance from sources to cycle edges.

    Parameters:
    -----------
    G : NetworkX DiGraph or MultiDiGraph
        The input graph to check and transform
    sources : list
        List of source nodes to consider for distance calculations

    Returns:
    --------
    NetworkX DiGraph
        A directed acyclic graph
    """
    import networkx as nx

    # Ensure we're working with a simple DiGraph
    if G.is_multigraph( ):
        G = convert_multidigraph_to_digraph(G, aggregation='sum')

    # Validate sources
    for source in sources:
        if source not in G:
            raise ValueError(f"Source node {source} not found in graph")

    def find_nearest_cycle_edge(G, sources, cycle):
        """
        Find the edge in the cycle that's closest to any source node.
        Returns (distance, edge, weight) tuple.
        """
        min_distance = float('inf')
        best_edge = None
        best_weight = float('inf')

        # Create undirected version of graph for distance calculation
        G_undir = G.to_undirected( )

        # Check each edge in the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if not G.has_edge(u, v):
                continue

            edge_weight = G[u][v].get('weight', 1.0)

            # Find minimum distance from any source to either endpoint
            for source in sources:
                try:
                    # Check distance to both endpoints of the edge
                    dist_u = nx.shortest_path_length(G_undir, source, u)
                    dist_v = nx.shortest_path_length(G_undir, source, v)
                    min_dist = min(dist_u, dist_v)

                    # Update if this is the closest edge found so far
                    if min_dist < min_distance or (min_dist == min_distance and edge_weight < best_weight):
                        min_distance = min_dist
                        best_edge = (u, v)
                        best_weight = edge_weight
                except nx.NetworkXNoPath:
                    continue

        return min_distance, best_edge, best_weight

    # Main loop to remove cycles
    while True:
        try:
            cycles = list(nx.simple_cycles(G))
            if not cycles:  # No cycles found
                return G

            print(f"Found {len(cycles)} cycles in the graph.")

            # Process each cycle
            cycle = cycles[0]  # Process one cycle at a time

            # Find the edge to reverse using the new heuristic
            min_distance, edge_to_reverse, _ = find_nearest_cycle_edge(G, sources, cycle)

            if edge_to_reverse is None:
                raise nx.NetworkXUnfeasible("Could not find suitable edge to reverse")

            # Reverse the chosen edge
            u, v = edge_to_reverse
            edge_data = G[u][v].copy( )
            G.remove_edge(u, v)
            G.add_edge(v, u, **edge_data)
            print(f"Reversed edge {u}->{v} to {v}->{u} (distance from source: {min_distance})")

        except nx.NetworkXUnfeasible as e:
            print(f"Error while processing graph: {e}")
            raise


def calculate_source_reachability(G, sources):
    """
    Calculate the number of sources that can reach each node in the network.

    Parameters:
    ----------
    G : networkx.DiGraph
        The directed graph representing the network
    sources : list, optional
        List of source nodes. If None, uses nodes with no predecessors

    Returns:
    -------
    pd.Series
        Series containing the count of sources that can reach each node
    """

    # Initialize reachability count for each node
    reachability = defaultdict(int)

    # Perform BFS from each source
    for source in sources:
        # Get all nodes reachable from this source
        reachable_nodes = nx.descendants(G, source)
        # Include the source itself
        reachable_nodes.add(source)

        # Increment count for each reachable node
        for node in reachable_nodes:
            reachability[node] += 1

    # Ensure all nodes are in the result (including unreachable ones)
    final_result = pd.Series(0, index=G.nodes( ), dtype=int)
    final_result.update(pd.Series(reachability))

    return final_result


def is_critical_failure_state(graph: nx.DiGraph, current_sources: Set[str], original_reachability: int,
                              critical_threshold: float) -> bool:
    """
    Check if the graph is in a critical state, where the number of reachable nodes
    from the sources is below the critical threshold.
    """
    reachable = calculate_reachability(graph, current_sources)
    return len(reachable) < critical_threshold * original_reachability


def simulate_failures(graph: nx.DiGraph, initial_node: str, sources: Set[str], original_reachability: int,
                      critical_threshold: float) -> int:
    """
    Simulate random node failures until the graph reaches a critical state.
    Returns the number of additional failures needed.
    """
    # Create a working copy of the graph and remove the initial node
    working_graph = graph.copy( )
    working_graph.remove_node(initial_node)

    # Update the sources if the initial node is a source
    current_sources = sources - {initial_node}

    # If the graph is already in a critical state, return 0
    if is_critical_failure_state(working_graph, current_sources, original_reachability, critical_threshold):
        return 0

    # Get the list of remaining nodes (excluding sources)
    remaining_nodes = list(set(working_graph.nodes( )) - current_sources)
    failures = 0

    # Simulate random failures until the graph reaches a critical state
    while remaining_nodes:
        # Remove a random node
        node_to_remove = np.random.choice(remaining_nodes)
        remaining_nodes.remove(node_to_remove)
        working_graph.remove_node(node_to_remove)
        failures += 1

        # Check if the graph is now in a critical state
        if is_critical_failure_state(working_graph, current_sources, original_reachability, critical_threshold):
            break

    return failures


def process_node_for_failure(args: tuple) -> tuple:
    """
    Process a single node to calculate its resilience score.
    """
    (node, graph, sources, original_reachability, critical_threshold, n_simulations) = args
    total_failures = 0

    for _ in range(n_simulations):
        failures = simulate_failures(graph, node, sources, original_reachability, critical_threshold)
        total_failures += failures

    # Return the average number of failures needed
    return node, total_failures / n_simulations


def calculate_reachability(graph: nx.DiGraph, sources: Set[str]) -> Set[str]:
    """
    Calculate the set of nodes reachable from the given sources in the graph.
    """
    reachable = set( )
    for source in sources:
        if source in graph:
            reachable.add(source)
            reachable.update(nx.descendants(graph, source))
    return reachable


def create_directory_structure ( base_dir ):
    """Create the directory structure for organizing centrality plots."""
    scenarios = { 'random_failures': [ 'average_number_of_failures_before_criticality_unweighted',
                                       'vitality_random_walk_betweenness', 'vitality_geodetic_betweenness',
                                       'subset_geodetic_betweenness', 'subset_random_walk_betweenness' ],
        'adversarial_failures': [ ],
        'contamination': [ 'vitality_closeness', 'subset_closeness_centrality_unweighted', 'average_hitting_time', ],
        'adversarial_contamination': [ 'horton_lines', 'min_hitting_time' ], 'electric_outages': [ ] }

    # Create main scenario directories
    for scenario in scenarios.keys( ):
        os.makedirs( os.path.join( base_dir, scenario, 'unweighted' ), exist_ok=True )
        os.makedirs( os.path.join( base_dir, scenario, 'weighted' ), exist_ok=True )
        os.makedirs( os.path.join( base_dir, scenario, 'specific_cases' ), exist_ok=True )

    return scenarios

