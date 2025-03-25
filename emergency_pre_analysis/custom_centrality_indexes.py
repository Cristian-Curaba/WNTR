"""
emergency_pre_analysis/custom_centrality_indexes.py
"""
import concurrent
import inspect
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from typing import Set, List, Union

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from emergency_pre_analysis.utils import *


# TODO coefficient should be calculated on G_demands where we consider only the node demands liked iff exists a path
def calculate_global_metrics ( G, G_undirected, wn ):
    """
    Calculate global network metrics.
    """
    print( "\n=== Global Network Metrics ===" )

    # Additional Network Statistics
    print( "\n=== Additional Network Statistics ===" )
    # 1. Meshedness Coefficient
    try:
        n = G_undirected.number_of_nodes( )
        m = G_undirected.number_of_edges( )
        max_edges = 2 * n - 5
        meshedness = (m - n + 1) / max_edges if max_edges > (n - 1) else 0.0
        print( f"✓ Global Meshedness Coefficient: {meshedness:.3f}" )
    except Exception as e:
        print( f"Error calculating Meshedness: {e}" )

    # 2. Average Degree (directed or undirected; here we use the directed G)
    try:
        n = G.number_of_nodes( )
        avg_degree = sum( dict( G.degree( ) ).values( ) ) / n
        print( f"✓ Average Degree: {avg_degree:.2f}" )
    except Exception as e:
        print( f"Error calculating Average Degree: {e}" )

    # 3. Network Failure Analysis (directed checks)
    def check_critical_disconnection ( graph, source_nodes ):
        """
        Check if more than 10% of nodes are disconnected from sources
        using a directed BFS/DFS approach (descendants).
        """
        reachable_nodes = set( )
        for s_node in source_nodes:
            if s_node in graph:
                reachable_nodes.update( nx.descendants( graph, s_node ) )
                reachable_nodes.add( s_node )
        return len( reachable_nodes ) < 0.9 * graph.number_of_nodes( )

    try:
        source_nodes = wn.tank_name_list + wn.reservoir_name_list

        # Random Failures Analysis
        n_trials = 100
        random_failures = [ ]

        for _ in range( n_trials ):
            G_copy = G.copy( )
            nodes_removed = 0
            node_list = list( G_copy.nodes( ) )
            np.random.shuffle( node_list )

            for node in node_list:
                if node not in source_nodes:  # Don't remove source nodes
                    G_copy.remove_node( node )
                    nodes_removed += 1
                    if check_critical_disconnection( G_copy, source_nodes ):
                        random_failures.append( nodes_removed )
                        break

        avg_random_walk_failures = np.mean( random_failures ) if random_failures else 0
        print( f"✓ Average number of random failures before critical disconnection: {avg_random_walk_failures:.1f}" )

        # Adversarial Failures Analysis (using betweenness centrality)
        G_copy = G.copy( )
        adversarial_failures = 0

        while True:
            if G_copy.number_of_nodes( ) <= 1:
                break
            btw = nx.betweenness_centrality( G_copy )
            target_node = max( (node for node in btw.items( ) if node[ 0 ] not in source_nodes), key=lambda x: x[ 1 ],
                default=None )
            if not target_node:
                break
            G_copy.remove_node( target_node[ 0 ] )
            adversarial_failures += 1

            if check_critical_disconnection( G_copy, source_nodes ):
                break

        print( f"✓ Number of adversarial failures before critical disconnection: {adversarial_failures}" )

    except Exception as e:
        print( f"Error calculating failure metrics: {e}" )

    # 4. Existing edge/total possible edges
    try:
        n = G_undirected.number_of_nodes( )
        m = G_undirected.number_of_edges( )
        max_edges_planar_graph = 3 * n - 6
        edge_density = m / max_edges_planar_graph
        print( f"✓ Edge Density: {edge_density:.3f}" )
    except Exception as e:
        print( f"Error calculating Edge Density: {e}" )
    try:
        print( f"Total Nodes: {G.number_of_nodes( )}" )
        print( f"Total Edges: {G.number_of_edges( )}" )
        print( f"Source Nodes (Tanks + Reservoirs): {len( source_nodes )}" )
        print( f"Network Diameter (on undirected): {nx.diameter( G_undirected )}" )
        try:
            print( f"Average Shortest Path Length (on undirected): "
                   f"{nx.average_shortest_path_length( G_undirected ):.2f}" )
        except:
            print( "Average Shortest Path Length: N/A (Graph not connected)" )
    except Exception as e:
        print( f"Error calculating additional statistics: {e}" )


def calculate_undirected_metrics ( G_undirected ):
    """
    Calculate metrics that require an undirected graph.
    """
    print( "\n=== Calculating Undirected Graph Metrics ===" )
    centralities = { }

    # 1. Current-Flow Betweenness Centrality (requires undirected)
    try:
        current_flow_bet = nx.current_flow_betweenness_centrality( G_undirected )
        centralities[ 'current_flow_betweenness' ] = pd.Series( current_flow_bet )
        print( "✓ Current-Flow Betweenness calculated" )
    except Exception as e:
        print( f"Error calculating Current-Flow Betweenness: {e}" )

    # 2. Communicability Centrality (requires undirected)
    try:
        communicability = nx.communicability_betweenness_centrality( G_undirected )
        centralities[ 'communicability' ] = pd.Series( communicability )
        print( "✓ Communicability Centrality calculated" )
    except Exception as e:
        print( f"Error calculating Communicability: {e}" )

    # 3. Vitality Centrality (using closeness in the undirected graph)
    try:
        vitality = { }
        base_closeness = sum( nx.closeness_centrality( G_undirected ).values( ) )
        for node in G_undirected.nodes( ):
            H = G_undirected.copy( )
            H.remove_node( node )
            if len( H ) > 0:  # Check if graph is not empty
                new_closeness = sum( nx.closeness_centrality( H ).values( ) )
                vitality[ node ] = base_closeness - new_closeness
            else:
                vitality[ node ] = base_closeness
        centralities[ 'vitality' ] = pd.Series( vitality )
        print( "✓ Vitality Centrality calculated" )
    except Exception as e:
        print( f"Error calculating Vitality: {e}" )

    # 4. Random Walk Network Resilience (Spectral Gap) on undirected graph
    try:
        L = nx.normalized_laplacian_matrix( G_undirected )
        eigenvalues = np.linalg.eigvals( L.toarray( ) )
        spectral_gap = np.sort( np.abs( eigenvalues ) )[ 1 ]  # 2nd smallest eigenvalue

        spectral_centrality = { }
        for node in G_undirected.nodes( ):
            H = G_undirected.copy( )
            H.remove_node( node )
            if len( H ) > 0:
                L_new = nx.normalized_laplacian_matrix( H )
                eigenvalues_new = np.linalg.eigvals( L_new.toarray( ) )
                spectral_gap_new = np.sort( np.abs( eigenvalues_new ) )[ 1 ]
                spectral_centrality[ node ] = spectral_gap - spectral_gap_new
            else:
                spectral_centrality[ node ] = spectral_gap
        centralities[ 'spectral_gap' ] = pd.Series( spectral_centrality )
        print( "✓ Spectral Gap Analysis calculated" )
    except Exception as e:
        print( f"Error calculating Spectral Gap: {e}" )

    # 5. Modularity-Based Influence (undirected)
    try:
        communities = nx.community.greedy_modularity_communities( G_undirected )
        modularity_influence = { }
        for node in G_undirected.nodes( ):
            # Find which community the node belongs to
            node_community = None
            for i, community in enumerate( communities ):
                if node in community:
                    node_community = i
                    break
            if node_community is not None:
                internal_connections = sum(
                    1 for neighbor in G_undirected.neighbors( node ) if neighbor in communities[ node_community ] )
                external_connections = G_undirected.degree( node ) - internal_connections
                modularity_influence[ node ] = internal_connections / (external_connections + 1)
            else:
                modularity_influence[ node ] = 0
        centralities[ 'modularity_influence' ] = pd.Series( modularity_influence )
        print( "✓ Modularity-Based Influence calculated" )
    except Exception as e:
        print( f"Error calculating Modularity Influence: {e}" )

    return centralities


def calculate_average_hitting_time_dag ( G, sources, targets = None, weight = None ):
    """
    Calculate the expected hitting times from source(s) to specified targets in a weighted DAG.
    Edge weights represent resistance, and hitting times are sums of resistances along paths.
    """
    # Handle multiple sources
    if isinstance( sources, (list, set, tuple) ):
        sources = list( sources )
        for s in sources:
            if not G.has_node( s ):
                raise ValueError( f"Source node {s} not found in graph" )
    else:
        if not G.has_node( sources ):
            raise ValueError( f"Source node {sources} not found in graph" )
        sources = [ sources ]

    # Handle targets parameter
    if targets is None:
        targets = set( G.nodes( ) )
    elif not isinstance( targets, (list, set, tuple) ):
        targets = { targets }
    else:
        targets = set( targets )

    # Validate targets
    invalid_targets = targets - set( G.nodes( ) )
    if invalid_targets:
        raise ValueError( f"Target nodes {invalid_targets} not found in graph" )

    # Get topological ordering
    try:
        topo_order = list( nx.topological_sort( G ) )
    except nx.NetworkXUnfeasible:
        raise ValueError( "Graph contains cycles, not a DAG" )

    # Create a mapping from node to its position in topological order
    topo_map = { node: i for i, node in enumerate( topo_order ) }

    # Dictionary to store the sum of hitting times for each target
    sum_hitting_times = { target: 0.0 for target in targets }
    reachable_count = { target: 0 for target in targets }

    # Calculate hitting times from each source
    for src in sources:
        hitting_times = { node: float( 'inf' ) for node in G.nodes( ) }
        hitting_times[ src ] = 0.0

        start_idx = topo_map[ src ]
        for i in range( start_idx, len( topo_order ) ):
            node = topo_order[ i ]

            if hitting_times[ node ] == float( 'inf' ):
                continue

            # Update hitting times for successors
            for successor in G.successors( node ):
                resistance = 1.0 if weight is None else G[ node ][ successor ].get( weight, 1.0 )
                new_hitting_time = hitting_times[ node ] + resistance

                if new_hitting_time < hitting_times[ successor ]:
                    hitting_times[ successor ] = new_hitting_time

        # After processing all nodes from this source, update sums and counts
        for target in targets:
            if hitting_times[ target ] != float( 'inf' ):
                sum_hitting_times[ target ] += hitting_times[ target ]
                reachable_count[ target ] += 1

    # Calculate average hitting times
    result = { }
    for target in targets:
        if target in sources:
            result[ target ] = 0.0
        elif reachable_count[ target ] > 0:
            result[ target ] = sum_hitting_times[ target ] / reachable_count[ target ]
        else:
            result[ target ] = float( 'inf' )

    res = pd.Series( result, name="Hitting time" )
    res.attrs[ 'weight' ] = weight
    res.attrs[ 'targets' ] = targets
    return res


def calculate_min_hitting_times_dag ( G, sources, targets = None, weight = None ):
    """
    Calculate minimum hitting times from any source to each node in a weighted DAG.
    Edge weights represent resistance, and hitting times are sums of resistances along paths.

    Parameters:
    -----------
    G : NetworkX DiGraph
        A directed acyclic graph with optional edge weights
    sources : list
        List of source nodes
    targets : list, optional
        List of target nodes. If None, all nodes are considered targets.
    weight : string, optional
        The edge attribute to use as resistance. Default is None.
        If None, all edges have weight 1.0.
        If specified but not found, uses 1.0 as default resistance.

    Returns:
    --------
    pd.Series
        Series mapping each node to its minimum hitting time from any source.
    """
    if targets is None:
        targets = list( G.nodes( ) )

    try:
        topo_order = list( nx.topological_sort( G ) )
    except nx.NetworkXUnfeasible:
        raise ValueError( "Graph contains cycles, not a DAG" )

    hitting_times = { node: float( 'inf' ) for node in G.nodes( ) }

    for source in sources:
        if source in G.nodes( ):
            hitting_times[ source ] = 0.0
        else:
            print( f"Warning: Source node {source} not in graph" )

    for node in topo_order:
        if hitting_times[ node ] == float( 'inf' ):
            continue

        for successor in G.successors( node ):
            if weight is None:
                # Unweighted case: each edge has weight 1.0
                resistance = 1.0
            else:
                # Weighted case: get edge weight, default to 1.0 if not found
                edge_data = G[ node ][ successor ]
                resistance = edge_data.get( weight, 1.0 )

            new_hitting_time = hitting_times[ node ] + resistance

            if new_hitting_time < hitting_times[ successor ]:
                hitting_times[ successor ] = new_hitting_time

    min_hitting_times = { target: hitting_times[ target ] for target in targets if target in G.nodes( ) }

    # Add debugging information
    unreachable = [ node for node, time in min_hitting_times.items( ) if time == float( 'inf' ) ]
    if unreachable:
        print( f"Warning: {len( unreachable )} nodes are unreachable from any source" )
        if len( unreachable ) < 10:
            print( f"Unreachable nodes: {unreachable}" )
        else:
            print( f"First 10 unreachable nodes: {unreachable[ :10 ]}" )

    reachable = [ node for node, time in min_hitting_times.items( ) if time < float( 'inf' ) ]
    print( f"Info: {len( reachable )} nodes are reachable from sources" )

    res = pd.Series( min_hitting_times, name=",min_hitting time" )
    res.attrs[ 'weight' ] = weight
    res.attrs[ 'targets' ] = targets
    return res


def calculate_subset_geodetic_betweenness ( G, sources, weight: str = None ):
    """
    Calculate a custom betweenness metric for directed graph G
    restricted to paths that start at any node in 'sources' and end at
    any node in G.

    Parameters
    ----------
    weight
    G : nx.DiGraph
        The directed graph for the water network.
    sources : collection
        The set (or list) of nodes to use as path sources.

    Returns
    -------
    pd.Series
        A Series mapping node ->  betweenness value.
    """
    # Convert to lists if needed and filter out any nodes not in G
    valid_sources = [ s for s in sources if s in G.nodes( ) ]

    if not valid_sources:
        # If there are no valid sources, return a zero-valued Series for each node
        return pd.Series( data=0.0, index=G.nodes( ), name="subset_geodetic_betweenness" )

    try:
        bc_dict = nx.betweenness_centrality_subset( G, sources=valid_sources, targets=G.nodes( ), normalized=True,
            weight=weight )
        res = pd.Series( bc_dict, name="subset_geodetic_betweenness" )
        res.attrs[ 'weight' ] = weight
        res.attrs[ 'sources' ] = sources
        return res

    except AttributeError:
        # If betweenness_centrality_subset is not available in older NetworkX
        msg = ("Your NetworkX version does not have betweenness_centrality_subset.\n"
               "Try upgrading (pip install --upgrade networkx) or implement a manual approach.")
        print( msg )
        # Return zero for all nodes in G to avoid crashing
        return pd.Series( data=0.0, index=G.nodes( ), name="subset_geodetic_betweenness" )


def calculate_subset_closeness ( G, weight = None, normalized: bool = True ):
    """
    Calculate subset closeness centrality for each node in a weighted directed graph.

    The subset closeness centrality of a node v is defined as:
    Cl_S(v) = 1 / sum_{s in S} sum_{w in V} H(s,v,w)
    where:
    - S is the set of sources
    - V is the set of all nodes
    - H(s,v,w) is the hitting time (sum of resistances) from s to w through v

    Parameters:
    -----------
    G : NetworkX DiGraph
        The directed graph. Sources should be identified by node attribute 'type'
        being either 'Reservoir' or 'Tank'.
    weight : string, optional
        The edge attribute representing resistance. Default is None.
        If None, all edges have weight 1.0.
        If specified but not found, uses 1.0 as default resistance.

    Returns:
    --------
    pd.Series
        Series containing subset closeness centrality values for each node.
        Index is node labels, values are centrality scores.
    """
    # Identify sources (nodes with type Reservoir or Tank)
    sources = [ node for node, attrs in G.nodes( data=True ) if attrs.get( 'type' ) in [ 'Reservoir', 'Tank' ] ]

    if not sources:
        raise ValueError( "No sources (Reservoir or Tank) found in the graph" )

    # Get all nodes
    nodes = list( G.nodes( ) )

    # Initialize closeness values
    closeness_values = { v: 0.0 for v in nodes }

    # For each source and each node as intermediate point
    for v in nodes:
        total_hitting_time = 0.0

        # Create a modified graph where we force paths through v
        for s in sources:
            # Skip if source and intermediate are the same
            if s == v:
                continue

            # Calculate hitting times from v to all other nodes
            hitting_times_from_v = calculate_average_hitting_time_dag( G, v )

            # Calculate hitting times from source to v
            hitting_times_to_v = calculate_average_hitting_time_dag( G, s, targets=v )
            hitting_time_s_to_v = hitting_times_to_v[ v ]

            # Sum up total hitting times through v
            if hitting_time_s_to_v != float( 'inf' ):
                for w in nodes:
                    if w != v and w != s:
                        hitting_time_v_to_w = hitting_times_from_v[ w ]
                        if hitting_time_v_to_w != float( 'inf' ):
                            # Total hitting time is sum of resistances to reach v and from v to w
                            total_hitting_time += hitting_time_s_to_v + hitting_time_v_to_w

        # Calculate closeness for node v
        if total_hitting_time > 0:
            closeness_values[ v ] = 1.0 / total_hitting_time
        else:
            closeness_values[ v ] = 0.0

    # Convert to pandas Series
    res = pd.Series( closeness_values )

    # Normalize the values to [0,1] range
    if normalized:
        if res.max( ) > 0:
            res = res / res.max( )

    res.attrs[ 'weight' ] = weight
    res.attrs[ 'normalized' ] = normalized
    return res


def calculate_subset_random_walk_betweenness ( G, sources, target = None, weight = None, reverse = False,
                                               normalized: bool = True ) -> pd.Series:
    """
    Calculate the subset random walk betweenness centrality for nodes in a directed graph.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph (preferably a DAG)
    sources : list
        A list of source nodes from which to start the random walks
    target : node or None, optional (default=None)
        If None, calculates betweenness for all nodes.
        If specified, calculates contribution of each node to paths ending at target.
    weight : string or None, optional (default=None)
        If None, all edge weights are considered equal.
        Otherwise holds the name of the edge attribute used as weight.
    reverse : bool, optional (default=False)
        If True, reverses all edges in the graph before calculation.

    Returns
    -------
    pandas.Series
        Series of nodes with subset random walk betweenness centrality as the value.
        If target is specified, returns contribution of each node to paths ending at target.
    """

    if not G.is_directed( ):
        raise nx.NetworkXError( "Graph must be directed" )

    # Create a working copy of the graph
    if reverse:
        H = G.reverse( copy=True )
        all_nodes = set( H.nodes( ) )
        sources = list( all_nodes - set( sources ) )
    else:
        H = G.copy( )

    if not nx.is_directed_acyclic_graph( H ):
        raise nx.NetworkXError( "Graph contains cycles; must be a DAG" )

    if target is not None and target not in H:
        raise nx.NetworkXError( f"Target node {target} not in graph" )

    # Initialize betweenness dictionary with small epsilon to avoid division by zero
    epsilon = 1e-10
    betweenness = dict.fromkeys( H, epsilon )

    topo_order = list( nx.topological_sort( H ) )

    for s in sources:
        if s not in H:
            continue

        # Initialize with epsilon instead of 0.0
        total_path_weights = defaultdict( lambda: epsilon )
        total_path_weights[ s ] = 1.0  # Changed from 0.0 to 1.0

        paths_through = defaultdict( lambda: defaultdict( float ) )

        for t in topo_order:
            if t == s:
                continue

            new_paths = defaultdict( lambda: defaultdict( float ) )
            target_weight = epsilon  # Initialize with epsilon

            for pred in H.predecessors( t ):
                edge_weight = 1.0 if weight is None else H[ pred ][ t ].get( weight, 1.0 )
                # Avoid division by zero
                inv_edge_weight = 1.0 / max( edge_weight, epsilon )

                if pred == s:
                    target_weight += inv_edge_weight
                else:
                    # Changed condition to account for epsilon
                    if total_path_weights[ pred ] > epsilon:
                        for intermediate in H:
                            if intermediate != s and paths_through[ intermediate ][ pred ] > epsilon:
                                new_weight = paths_through[ intermediate ][ pred ] * inv_edge_weight
                                new_paths[ intermediate ][ t ] += new_weight

                        pred_contribution = total_path_weights[ pred ] * inv_edge_weight
                        new_paths[ pred ][ t ] += pred_contribution
                        target_weight += pred_contribution

            for intermediate in new_paths:
                paths_through[ intermediate ][ t ] = new_paths[ intermediate ][ t ]

            total_path_weights[ t ] = target_weight

        if target is None:
            for intermediate in H:
                if intermediate == s:
                    continue

                for t in H:
                    if t != intermediate and t != s:
                        if total_path_weights[ t ] > epsilon:
                            betweenness[ intermediate ] += (
                                    paths_through[ intermediate ][ t ] / total_path_weights[ t ])
        else:
            for intermediate in H:
                if intermediate == s or intermediate == target:
                    continue

                if total_path_weights[ target ] > epsilon:
                    betweenness[ intermediate ] += (  # Changed from = to +=
                            paths_through[ intermediate ][ target ] / total_path_weights[ target ])

    # Convert dictionary to pandas Series
    betweenness_series = pd.Series( betweenness )

    # Remove the epsilon baseline
    betweenness_series = betweenness_series - epsilon

    # Normalize the values only if target is None
    if normalized is None:
        max_value = betweenness_series.max( )
        if max_value > epsilon:
            betweenness_series = betweenness_series / max_value

    # Clean up any remaining negligible values
    betweenness_series[ np.abs( betweenness_series ) < epsilon ] = 0.0
    res = betweenness_series
    res.attrs[ 'weight' ] = weight
    res.attrs[ 'sources' ] = sources
    res.attrs[ 'normalized' ] = normalized
    res.attrs[ 'reverse' ] = reverse

    return res


@timer_decorator
def vitality ( G, sources, weight = None, target_contribution = None, goal = None, aggregation_method = 'sum',
               normalized: bool = True ) -> pd.Series:
    """
    Calculate the vitality of nodes in a graph based on their contribution to a goal function.

    Parameters:
    -----------
    G : NetworkX graph
        The input graph
    sources : set or list
        Set of source nodes (used in goal functions like path-based metrics)
    weight : str, optional
        Edge attribute used as weight. Default is 'weight'
    target_contribution : node, optional
        Target node to estimate contribution for. If None, calculate for all nodes
    goal : function, optional
        Function that takes a graph and returns a value to be maximized
    aggregation_method : str, optional
        Function to aggregation_method multiple values ('sum', 'average', 'max')

    Returns:
    --------
    pd.Series or float
        Series of node vitality values if target_contribution is None, otherwise a single vitality value
    """
    # Define default goal function if none provided
    if goal is None:
        def default_goal ( g, srcs = None, wt = None ):
            # Default to average shortest path length if connected
            if nx.is_connected( g ) and len( g ) > 1:
                try:
                    return nx.average_shortest_path_length( g, weight=wt )
                except:
                    return 0
            return 0

        goal = default_goal

    # Define aggregation function
    if aggregation_method == 'sum':
        agg_func = sum
    elif aggregation_method == 'average':
        agg_func = lambda x: sum( x ) / len( x ) if x else 0
    elif aggregation_method == 'max':
        agg_func = max
    else:
        raise ValueError( "Aggregation must be 'sum', 'average', or 'max'" )

    # Check goal function signature to determine what parameters to pass
    goal_params = inspect.signature( goal ).parameters
    goal_args = { }

    # Always pass the graph as the first argument
    # Check if other parameters are expected by the goal function
    if 'sources' in goal_params:
        goal_args[ 'sources' ] = sources
    if 'weight' in goal_params:
        goal_args[ 'weight' ] = weight
    if 'srcs' in goal_params:
        goal_args[ 'srcs' ] = sources
    if 'wt' in goal_params:
        goal_args[ 'wt' ] = weight

    # Get original goal value
    try:
        goal_result = goal( G, **goal_args )
        # Handle if goal returns a collection that needs aggregation
        if isinstance( goal_result, (list, tuple, set, np.ndarray) ):
            original_value = agg_func( goal_result )
        elif isinstance( goal_result, dict ):
            # If goal returns a dictionary, apply aggregation to values
            original_value = agg_func( goal_result.values( ) )
        elif isinstance( goal_result, pd.Series ) or isinstance( goal_result, pd.DataFrame ):
            # If goal returns a pandas Series or DataFrame, convert to a single value
            original_value = agg_func( goal_result.values.flatten( ) )
        else:
            original_value = goal_result
    except Exception as e:
        return pd.Series( { "Error": f"Error calculating original goal: {str( e )}" } )

    # Calculate vitality for a single target
    if target_contribution is not None:
        if target_contribution not in G:
            return 0

        # Remove the node and calculate new goal value
        G_copy = G.copy( )
        G_copy.remove_node( target_contribution )

        try:
            goal_result = goal( G_copy, **goal_args )
            # Handle if goal returns a collection that needs aggregation
            if isinstance( goal_result, (list, tuple, set, np.ndarray) ):
                new_value = agg_func( goal_result )
            elif isinstance( goal_result, dict ):
                # If goal returns a dictionary, apply aggregation to values
                new_value = agg_func( goal_result.values( ) )
            elif isinstance( goal_result, pd.Series ) or isinstance( goal_result, pd.DataFrame ):
                # If goal returns a pandas Series or DataFrame, convert to a single value
                new_value = agg_func( goal_result.values.flatten( ) )
            else:
                new_value = goal_result
        except Exception as e:
            return f"Error: {str( e )}"

        return original_value - new_value

    # Calculate vitality for all nodes
    results = { }
    for node in G.nodes( ):
        G_copy = G.copy( )
        G_copy.remove_node( node )

        # Update sources if the removed node was a source
        updated_goal_args = goal_args.copy( )
        if 'sources' in updated_goal_args and node in updated_goal_args[ 'sources' ]:
            updated_sources = [ s for s in updated_goal_args[ 'sources' ] if s != node ]
            updated_goal_args[ 'sources' ] = updated_sources
        if 'srcs' in updated_goal_args and node in updated_goal_args[ 'srcs' ]:
            updated_sources = [ s for s in updated_goal_args[ 'srcs' ] if s != node ]
            updated_goal_args[ 'srcs' ] = updated_sources

        try:
            goal_result = goal( G_copy, **updated_goal_args )
            # Handle if goal returns a collection that needs aggregation
            if isinstance( goal_result, (list, tuple, set, np.ndarray) ):
                new_value = agg_func( goal_result )
            elif isinstance( goal_result, dict ):
                # If goal returns a dictionary, apply aggregation to values
                new_value = agg_func( goal_result.values( ) )
            elif isinstance( goal_result, pd.Series ) or isinstance( goal_result, pd.DataFrame ):
                # If goal returns a pandas Series or DataFrame, convert to a single value
                new_value = agg_func( goal_result.values.flatten( ) )
            else:
                new_value = goal_result

            # Calculate vitality (difference between original and new value)
            results[ node ] = original_value - new_value
        except Exception as e:
            # Handle errors by setting to NaN instead of error message
            results[ node ] = float( 'nan' )
            print( f"Error calculating vitality for node {node}: {str( e )}" )

    # Return as pandas Series
    goal_name = getattr( goal, '__name__', str( goal ) )

    res = pd.Series( results, name=f"vitality_{goal_name}" )
    res.attrs[ 'weight' ] = weight
    res.attrs[ 'sources' ] = sources
    res.attrs[ 'normalized' ] = normalized
    res.attrs[ 'aggregation_method' ] = aggregation_method
    res.attrs[ 'target_contribution' ] = target_contribution
    return res


@timer_decorator
def calculate_average_number_of_failures_before_criticality_unoptimized ( G, sources, critical_threshold = 0.9 ):
    """
    Calculate a centrality index based on network resilience to node failures.
    For each node (including sources), calculates the average number of additional
    node failures needed to cause a critical scenario after the target node fails.

    A critical scenario occurs when the largest connected component contains less than
    a threshold percentage (default 90%) of the reachable nodes from the remaining sources.

    Parameters:
    -----------
    G : NetworkX DiGraph
        The directed graph to analyze
    sources : list or set
        List of source nodes to consider for reachability calculations
    critical_threshold : float, optional
        Threshold ratio (0-1) of reachable nodes that defines a critical scenario.
        Default is 0.9 (90% of original reachable nodes).

    Returns:
    --------
    pd.Series
        Series containing resilience scores for each node.
        Higher values indicate more failures needed to reach critical state.
    """
    # Validate sources
    if not isinstance( sources, (list, set, tuple) ):
        sources = [ sources ]
    sources = set( sources )

    invalid_sources = sources - set( G.nodes( ) )
    if invalid_sources:
        raise ValueError( f"Source nodes {invalid_sources} not found in graph" )

    def get_reachability ( graph, current_sources ):
        """
        Helper function to calculate reachable nodes from current sources.
        """
        reachable = set( )
        for source in current_sources:
            if source in graph:
                reachable.add( source )
                reachable.update( nx.descendants( graph, source ) )
        return reachable

    def is_critical_state ( graph, current_sources, original_reachability ):
        """
        Helper function to determine if graph is in critical state.
        Returns True if reachability from remaining sources is less than threshold [%].
        """
        if not graph.nodes( ):
            return True

        reachable = get_reachability( graph, current_sources )
        current_reachability = len( reachable )

        # Critical if reachability is less than threshold% of original
        return current_reachability < critical_threshold * original_reachability

    def simulate_failures ( graph, initial_node, original_reachability ):
        """
        Simulate random node failures until critical state is reached.
        Returns number of additional failures needed.
        """
        # Create working copy and remove initial node
        working_graph = graph.copy( )
        working_graph.remove_node( initial_node )

        # Update sources for this simulation
        current_sources = sources - { initial_node }

        # If already critical, return 0
        if is_critical_state( working_graph, current_sources, original_reachability ):
            return 0

        remaining_nodes = list( set( working_graph.nodes( ) ) )
        failures = 0

        # Simulate failures until critical state reached
        while remaining_nodes and not is_critical_state( working_graph, current_sources, original_reachability ):
            # Remove random node
            node_to_remove = np.random.choice( remaining_nodes )
            remaining_nodes.remove( node_to_remove )
            working_graph.remove_node( node_to_remove )

            # Update sources if we removed a source
            if node_to_remove in current_sources:
                current_sources.remove( node_to_remove )

            failures += 1

        return failures

    # Calculate original reachability from sources
    original_reachable = get_reachability( G, sources )
    original_reachability = len( original_reachable )

    if original_reachability == 0:
        raise ValueError( "No nodes are reachable from the provided sources" )

    # Initialize results dictionary
    centrality_scores = { }

    # For each node in the graph (including sources)
    for node in G.nodes( ):
        # Perform multiple simulations and average results
        n_simulations = 100  # Number of simulations per node
        total_failures = 0

        for _ in range( n_simulations ):
            failures = simulate_failures( G, node, original_reachability )
            total_failures += failures

        # Store average number of failures needed
        centrality_scores[ node ] = total_failures / n_simulations

    return pd.Series( centrality_scores, name="average_number_of_failures_before_criticality_unweighted" )


@timer_decorator
def calculate_average_number_of_failures_before_criticality ( G: nx.DiGraph, sources: Union[ List, Set, str ],
                                                              critical_threshold: float = 0.9,
                                                              n_simulations: int = 100 ) -> pd.Series:
    """
    Calculate a network resilience metric based on the average number of random node
    failures needed to reach a critical state after each node fails.

    Parameters
    ----------
    G : networkx.DiGraph
        The directed graph to analyze
    sources : Union[List, Set, str]
        Source nodes to consider for reachability calculations
    critical_threshold : float, optional
        Threshold ratio (0-1) defining critical scenario (default: 0.9)
    n_simulations : int, optional
        Number of simulations per node (default: 100)

    Returns
    -------
    pd.Series
        Series containing resilience scores for each node.
    """
    # Input validation
    if not isinstance( G, nx.DiGraph ):
        raise TypeError( "Graph must be a NetworkX DiGraph" )

    if not isinstance( sources, (list, set, str) ):
        raise TypeError( "Sources must be a list, set, or string" )

    # Convert single source to set
    if isinstance( sources, str ):
        sources = { sources }
    sources = set( sources )

    # Validate sources
    invalid_sources = sources - set( G.nodes( ) )
    if invalid_sources:
        raise ValueError( f"Source nodes {invalid_sources} not found in graph" )

    # Calculate the original reachability from the sources
    original_reachable = calculate_reachability( G, sources )
    original_reachability = len( original_reachable )

    if original_reachability == 0:
        raise ValueError( "No nodes are reachable from the provided sources" )

    # Prepare data for parallel processing
    node_data = [ (node, G, sources, original_reachability, critical_threshold, n_simulations) for node in G.nodes( ) ]

    # Initialize progress bar
    total_nodes = len( node_data )
    pbar = tqdm( total=total_nodes, desc="Processing nodes", unit="node",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} nodes '
                   '[{elapsed}<{remaining}, {rate_fmt}]' )

    results = [ ]

    # Execute parallel processing with progress tracking
    with ProcessPoolExecutor( max_workers=multiprocessing.cpu_count( ) ) as executor:
        # Submit all tasks
        future_to_node = { executor.submit( process_node_for_failure, data ): data[ 0 ] for data in node_data }

        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed( future_to_node ):
            node = future_to_node[ future ]
            try:
                result = future.result( )
                results.append( result )
            except Exception as e:
                print( f"Error processing node {node}: {e}" )
            pbar.update( 1 )

    pbar.close( )

    # Convert results to pandas Series
    centrality_scores = dict( results )

    res = pd.Series( centrality_scores, name="average_failures_before_criticality" )

    res.attrs[ 'critical_threshold' ] = critical_threshold
    res.attrs[ 'sources' ] = sources

    return res


@timer_decorator
def spread_importance_pagerank ( G, node_importance, reverse = False, alpha = 0.85 ):
    """
    Spread node importance through the network using PersonalizedPageRank.

    Parameters:
    - G: NetworkX directed acyclic graph
    - node_importance: dict or pd.Series of node importance values
    - reverse: bool, if True spread against edge direction
    - alpha: damping factor (1-alpha is the teleportation probability)

    Returns:
    - pd.Series of propagated importance values
    """

    def get_node_key ( node ):
        """Extract the node key, handling both tuple and non-tuple nodes."""
        if isinstance( node, tuple ):
            return node[ 0 ]
        return node

    # Convert input to dict if it's a Series
    if isinstance( node_importance, pd.Series ):
        node_importance = node_importance.to_dict( )

    # Normalize personalization vector
    total = sum( node_importance.values( ) )
    if total > 0:
        personalization = { node: node_importance.get( get_node_key( node ), 0.0 ) / total for node in G.nodes( ) }
    else:
        # If all values are 0, use uniform distribution
        personalization = { node: 1.0 / len( G ) for node in G.nodes( ) }

    # If reverse is True, use the reversed graph
    if reverse:
        G = G.reverse( )

    # Use NetworkX's personalized PageRank
    propagated = nx.pagerank( G, alpha=alpha, personalization=personalization, dangling=None
        # Let NetworkX handle dangling nodes
    )

    # Scale back to original magnitude
    max_original = max( node_importance.values( ) )
    max_propagated = max( propagated.values( ) )
    if max_propagated > 0:
        scaling_factor = max_original / max_propagated
        propagated = { node: value * scaling_factor for node, value in propagated.items( ) }

    res = pd.Series( propagated )
    res.attrs[ 'node_importance' ] = node_importance
    res.attrs[ 'reverse' ] = reverse
    res.attrs[ 'alpha' ] = alpha
    return res


@timer_decorator
def spread_importance ( G, node_importance, reverse = False, alpha = 0.85 ):
    """
    Spread node importance through the network by distributing values along weighted edges.
    For each node v, its importance I(v) is the sum of its original importance plus
    the weighted sum of importance from nodes it points to.

    Parameters:
    - G: NetworkX directed acyclic graph
    - node_importance: dict or pd.Series of node importance values
    - reverse: bool, if True spread against edge direction
    - alpha: damping factor for importance contribution

    Returns:
    - pd.Series of propagated importance values
    """
    import pandas as pd
    import networkx as nx

    def get_node_key ( node ):
        """Extract the node key, handling both tuple and non-tuple nodes."""
        if isinstance( node, tuple ):
            return node[ 0 ]
        return node

    # Convert input to dict if it's a Series
    if isinstance( node_importance, pd.Series ):
        node_importance = node_importance.to_dict( )

    # Initialize importance dictionary with original values
    propagated = { node: node_importance.get( get_node_key( node ), 0.0 ) for node in G.nodes( ) }

    # Get topological sort of the graph
    if reverse:
        G = G.reverse( )
    nodes_ordered = list( nx.topological_sort( G ) )

    # Propagate values from successors to predecessors
    for node in reversed( nodes_ordered ):
        # Get predecessors and their edge weights
        pred_weights = { }
        total_weight = 0.0

        for pred in G.predecessors( node ):
            weight = G[ pred ][ node ].get( 'weight', 1.0 )
            pred_weights[ pred ] = weight
            total_weight += weight

        if total_weight > 0:
            # Add weighted portion of current node's importance to predecessors
            current_importance = propagated[ node ]
            for pred, weight in pred_weights.items( ):
                # Add contribution without depleting current node's importance
                propagated[ pred ] += (weight / total_weight) * current_importance * alpha

    res = pd.Series( propagated )
    res.attrs[ 'node_importance' ] = node_importance
    res.attrs[ 'reverse' ] = reverse
    res.attrs[ 'alpha' ] = alpha
    return res


@timer_decorator
def calculate_horton_lines ( G, reverse = False, weights = None ):
    """
    Calculate Horton's lines based centrality.

    Parameters:
    - G: NetworkX directed acyclic graph
    - reverse: bool, if True calculate against edge direction
    - weights: dict of edge weights

    Returns:
    - dict of Horton centrality values
    """

    if weights:
        nx.set_edge_attributes( G, weights, 'weight' )

    # If reverse, we need to reverse the graph
    if reverse:
        G = G.reverse( )

    # Calculate stream orders
    stream_orders = { node: 1 for node in G.nodes( ) }
    for node in nx.topological_sort( G ):
        predecessors = list( G.predecessors( node ) )
        if predecessors:
            stream_orders[ node ] = max( stream_orders[ p ] for p in predecessors ) + 1

    # Calculate length-weighted paths
    path_lengths = { node: 0 for node in G.nodes( ) }
    for node in nx.topological_sort( G ):
        predecessors = list( G.predecessors( node ) )
        if predecessors:
            max_length = max( path_lengths[ p ] + G[ p ][ node ].get( 'weight', 1 ) for p in predecessors )
            path_lengths[ node ] = max_length

    # Combine stream order and path length
    horton_centrality = { }
    for node in G.nodes( ):
        horton_centrality[ node ] = (
            stream_orders[ node ] * path_lengths[ node ] if path_lengths[ node ] > 0 else stream_orders[ node ])

    # Normalize
    max_value = max( horton_centrality.values( ) )
    if max_value > 0:
        horton_centrality = { n: v / max_value for n, v in horton_centrality.items( ) }

    res = pd.Series( horton_centrality, name='horton_lines' )
    res.attrs[ 'weights' ] = weights
    res.attrs[ 'reverse' ] = reverse

    return res


def aggregate_centrality ( structural_centrality, importance_centrality, alpha = 0.5,
                           aggregation_method = 'weighted_sum' ):
    """
    Aggregate structural centrality with spread importance using pandas Series.

    Parameters:
    - structural_centrality: pandas Series or dict of structural centrality values
    - spread_importance: pandas Series or dict of spread importance values
    - alpha: weight for structural centrality (1-alpha for spread importance)
    - method: aggregation method ('weighted_sum', 'multiplication', 'max')

    Returns:
    - pandas Series of aggregated centrality values
    """

    # Convert inputs to pandas Series if they're not already
    if not isinstance( structural_centrality, pd.Series ):
        structural_centrality = pd.Series( structural_centrality )
    if not isinstance( importance_centrality, pd.Series ):
        importance_centrality = pd.Series( importance_centrality )

    # Ensure both Series have the same index
    common_index = structural_centrality.index.intersection( importance_centrality.index )
    struct_cent = structural_centrality[ common_index ]
    spread_imp = importance_centrality[ common_index ]

    # Perform aggregation based on method
    if aggregation_method == 'weighted_sum':
        aggregated = alpha * struct_cent + (1 - alpha) * spread_imp
    elif aggregation_method == 'multiplication':
        aggregated = struct_cent * spread_imp
    elif aggregation_method == 'max':
        aggregated = pd.concat( [ struct_cent, spread_imp ], axis=1 ).max( axis=1 )
    else:
        raise ValueError( f"Unknown aggregation method: {aggregation_method}" )

    res = aggregated
    res.attrs[ 'weights' ] = structural_centrality
    res.attrs[ 'reverse' ] = importance_centrality
    res.attrs[ 'alpha' ] = alpha
    res.attrs[ 'aggregation_method' ] = aggregation_method
    return res


def calculate_directed_metrics ( G, wn ):
    """
    Calculate metrics for directed graphs, with special handling for DAGs when appropriate.
    """
    print( "\n=== Calculating Directed Graph Metrics ===" )

    sources = get_source_nodes( wn )[ 0 ]
    G = convert_multidigraph_to_digraph( G )
    G = check_and_transform_to_directed_acyclic_graph( G, sources )
    check_unreachable_nodes( G, sources )
    centralities = { }

    # Standard centrality metrics
    centralities[ 'min_hitting_time' ] = calculate_min_hitting_times_dag( G, sources )
    centralities[ 'average_hitting_time' ] = calculate_average_hitting_time_dag( G, sources )
    centralities[ 'subset_geodetic_betweenness' ] = calculate_subset_geodetic_betweenness( G, sources )
    centralities[ 'subset_random_walk_betweenness' ] = calculate_subset_random_walk_betweenness( G, sources,
                                                                                                 weight=None )
    centralities[ 'horton_lines' ] = calculate_horton_lines( G, reverse=False )

    # # Target-specific random walk betweenness
    # possible_targets = set(G.nodes()) - set(sources)
    # if not possible_targets:
    #     raise ValueError("No valid target nodes available (all nodes are sources)")
    # target = np.random.choice(list(possible_targets))
    # print(f"Selected random target node: {target}")
    # reverse = True
    # centralities[f'subset_random_walk_betweenness_on_{target}_{reverse}'] = calculate_subset_random_walk_betweenness(
    #     G, sources, target=target, reverse=reverse)
    #
    centralities[ 'subset_closeness_centrality_unweighted' ] = calculate_subset_closeness( G )
    centralities[
        'average_number_of_failures_before_criticality_unweighted' ] = calculate_average_number_of_failures_before_criticality(
        G, sources, critical_threshold=0.7 )
    # Vitality metrics
    if len( G.nodes( ) ) < 150:
        centralities[ 'vitality_random_walk_betweenness' ] = vitality( G, sources,
                                                                       goal=calculate_subset_random_walk_betweenness )
        centralities[ 'vitality_closeness' ] = vitality( G, sources, goal=calculate_subset_closeness )
    centralities[ 'vitality_geodetic_betweenness' ] = vitality( G, sources, goal=calculate_subset_geodetic_betweenness )

    print( "\nVitality metrics calculated:" )
    for metric_name in [ name for name in centralities.keys( ) if name.startswith( 'vitality_' ) ]:
        print( f"{metric_name}:" )
        print( centralities[ metric_name ].head( ) )
        print( )

    # Qualitative metricx
    # Generate random external importance
    external_importance = { node: int( np.random.sample( 1 ) > 0.9 ) for node in list( G.nodes( ) ) }

    print( f"external_importance: {external_importance}" )
    # Calculate spread importance in both directions
    centralities[ 'importance_spread_forward' ] = spread_importance( G, external_importance, reverse=False, alpha=1, )

    centralities[ 'importance_spread_backward' ] = spread_importance( G, external_importance, reverse=True, alpha=1, )

    # Store original external importance for reference
    centralities[ 'external_importance' ] = external_importance

    # Source-based redaudancy
    centralities[ 'source_reachability' ] = calculate_source_reachability( G, sources )
    return centralities
