# def calculate_hitting_time_directed(G, source, target, weight=None):
#     """
#     Calculates the hitting time in a weighted directed graph using a random-walk transition matrix.
#     Edge weights represent resistance. Returns float('inf') if target is not reachable from source.
#
#     Parameters:
#     -----------
#     G : NetworkX DiGraph
#         The directed graph with optional edge weights
#     source : node
#         Starting node
#     target : node
#         Target node
#     weight : string, optional
#         The edge attribute representing resistance. Default is None.
#         If None, all edges have weight 1.0.
#         If specified but not found, uses 1.0 as default resistance.
#
#     Returns:
#     --------
#     float
#         Expected hitting time from source to target
#     """
#     if source == target:
#         return 0.0
#
#     # Sort nodes for consistent indexing
#     nodes = sorted(G.nodes())
#     n = len(nodes)
#
#     # Create weighted adjacency matrix
#     if weight is None:
#         # If no weight specified, use binary adjacency matrix
#         A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
#     else:
#         # Use specified weight attribute
#         A = nx.to_numpy_array(G, nodelist=nodes, dtype=float, weight=weight)
#
#     # Construct transition probability matrix P
#     P = np.zeros((n, n), dtype=float)
#     for i in range(n):
#         if weight is None:
#             # For unweighted case, use simple probability
#             row_sum = A[i].sum()
#             if row_sum > 0:
#                 P[i, :] = A[i, :] / row_sum
#         else:
#             # For weighted case, use conductances (1/resistance)
#             conductances = np.where(A[i] > 0, 1.0 / A[i], 0.0)
#             row_sum = conductances.sum()
#             if row_sum > 0:
#                 P[i, :] = conductances / row_sum
#
#     try:
#         s = nodes.index(source)
#         t = nodes.index(target)
#     except ValueError:
#         return float('inf')
#
#     if np.allclose(P[s, :], 0.0):
#         return float('inf')
#
#     # Remove target row and column from P
#     Q = np.delete(np.delete(P, t, axis=0), t, axis=1)
#     if s > t:
#         s -= 1
#
#     n_sub = n - 1
#     I = np.eye(n_sub, dtype=float)
#     b = np.ones(n_sub, dtype=float)
#
#     try:
#         h = np.linalg.solve(I - Q, b)
#     except np.linalg.LinAlgError:
#         return float('inf')
#
#     if not (0 <= s < n_sub):
#         return float('inf')
#
#     # Scale the hitting time by the resistances if weights are used
#     if weight is not None:
#         # Calculate average resistance along the path
#         avg_resistance = np.mean([d.get(weight, 1.0) for u, v, d in G.edges(data=True)])
#         return float(h[s] * avg_resistance)
#     else:
#         return float(h[s])

# def orient_edges_by_pressure_and_topology(wn, pressure_sources=None):
#     """
#     Orient edges in a water network based on pressure sources, topology, and elevation.
#
#     Parameters:
#     -----------
#     wn : WNTR WaterNetworkModel
#         The water network model
#     pressure_sources : dict, optional
#         Dictionary mapping source nodes to their pressure values. If None,
#         estimates pressures based on elevation and node type.
#
#     Returns:
#     --------
#     tuple
#         (NetworkX DiGraph, WNTR WaterNetworkModel) - The oriented graph and modified network
#     """
#     import networkx as nx
#     import numpy as np
#
#     def validate_network(wn):
#         """Validate basic network properties"""
#         if not hasattr(wn, 'get_graph'):
#             raise ValueError("Input network must have get_graph method")
#         if not wn.node_name_list:
#             raise ValueError("Network has no nodes")
#         if not list(wn.pipes()):
#             raise ValueError("Network has no pipes")
#         return True
#
#     def estimate_source_pressures(wn):
#         """Estimate pressures for sources based on elevation and node type"""
#         pressures = {}
#         avg_elevation = np.mean([get_node_elevation(node) for node in wn.node_name_list])
#
#         for tank_name, tank in wn.tanks():
#             elevation = get_node_elevation(tank_name)
#             pressures[tank_name] = max(15, (elevation - avg_elevation) + 10)
#
#         for res_name, reservoir in wn.reservoirs():
#             elevation = get_node_elevation(res_name)
#             pressures[res_name] = max(20, (elevation - avg_elevation) + 15)
#
#         return pressures
#
#     def get_default_pipe_properties(wn):
#         """Calculate default pipe properties from existing data"""
#         diameters = []
#         lengths = []
#
#         for _, pipe in wn.pipes():
#             if hasattr(pipe, 'diameter'):
#                 diameters.append(pipe.diameter)
#             if hasattr(pipe, 'length'):
#                 lengths.append(pipe.length)
#
#         default_diameter = np.median(diameters) if diameters else 100
#         default_length = np.median(lengths) if lengths else 100
#
#         return default_diameter, default_length
#
#     def get_node_elevation(node_id):
#         """Get node elevation with fallback to average elevation"""
#         try:
#             node = wn.get_node(node_id)
#             if hasattr(node, 'elevation'):
#                 return node.elevation
#         except:
#             pass
#
#         elevations = []
#         for node_name in wn.node_name_list:
#             try:
#                 node = wn.get_node(node_name)
#                 if hasattr(node, 'elevation'):
#                     elevations.append(node.elevation)
#             except:
#                 continue
#
#         return np.mean(elevations) if elevations else 0
#
#     def calculate_edge_orientation_score(u, v, distances, pressures):
#         """Calculate score for edge orientation"""
#         edge_data = G_undir[u][v]
#         diameter = edge_data.get('diameter', default_diameter)
#         length = edge_data.get('length', default_length)
#
#         elevation_u = get_node_elevation(u)
#         elevation_v = get_node_elevation(v)
#         elevation_diff = elevation_u - elevation_v
#
#         g = 9.81
#         rho = 1000
#         elevation_pressure = rho * g * elevation_diff
#
#         pressure_score = 0
#         if pressures:
#             for source, pressure in pressures.items():
#                 if source in distances:
#                     dist_u = distances[source].get(u, float('inf'))
#                     dist_v = distances[source].get(v, float('inf'))
#
#                     source_pressure = pressure * rho * g
#                     decay_factor_u = np.exp(-0.1 * (dist_u / diameter))
#                     decay_factor_v = np.exp(-0.1 * (dist_v / diameter))
#
#                     pressure_u = source_pressure * decay_factor_u
#                     pressure_v = source_pressure * decay_factor_v
#
#                     pressure_score += pressure_u - pressure_v
#
#         total_pressure_diff = pressure_score + elevation_pressure
#         hydraulic_factor = (diameter + 1) / (np.sqrt(length + 1))
#
#         return total_pressure_diff * hydraulic_factor
#
#     # Validation
#     validate_network(wn)
#
#     # Initialize graphs
#     G_undir = wn.get_graph().to_undirected()
#     G_dir = nx.DiGraph()
#     G_dir.add_nodes_from(G_undir.nodes(data=True))
#
#     # Get default properties
#     default_diameter, default_length = get_default_pipe_properties(wn)
#
#     # Handle missing pressure sources
#     if pressure_sources is None:
#         pressure_sources = estimate_source_pressures(wn)
#
#     # Calculate distances
#     source_distances = {}
#     for source in pressure_sources:
#         try:
#             source_distances[source] = nx.shortest_path_length(G_undir, source)
#         except nx.NetworkXError:
#             continue
#
#     def print_edge_switch(u, v, data, score):
#         """Print information about switched edge orientation"""
#         link_name = data.get('name', f'{u}-{v}')
#         print(f"\nEdge {link_name} switched direction:")
#         print(f"  From: {u} -> {v}")
#         print(f"  To:   {v} -> {u}")
#         print(f"  Score: {score:.2f}")
#
#         # Print elevation information
#         elev_u = get_node_elevation(u)
#         elev_v = get_node_elevation(v)
#         print(f"  Elevation diff (original direction): {elev_u - elev_v:.2f}")
#
#     def find_atypical_sources(G):
#         """Find nodes that act as sources but aren't typical sources"""
#         atypical_sources = set()
#         for node in G.nodes():
#             if node in pressure_sources:
#                 continue
#             in_edges = list(G.in_edges(node))
#             out_edges = list(G.out_edges(node))
#             if len(in_edges) == 0 and len(out_edges) > 1:
#                 atypical_sources.add(node)
#         return atypical_sources
#
#     # Store initial pipe directions from the water network
#     initial_directions = {}
#     for pipe_name, pipe in wn.pipes():
#         initial_directions[pipe_name] = (pipe.start_node_name, pipe.end_node_name)
#
#     # Initialize directed graph
#     G_dir = nx.DiGraph()
#     G_dir.add_nodes_from(G_undir.nodes(data=True))
#
#     # Find initial atypical sources from the water network
#     initial_graph = nx.DiGraph()
#     for pipe_name, (start, end) in initial_directions.items():
#         initial_graph.add_edge(start, end)
#     initial_atypical_sources = find_atypical_sources(initial_graph)
#
#     print(f"\nInitial state:")
#     print(f"Typical sources: {len(pressure_sources)}")
#     print(f"Atypical sources: {len(initial_atypical_sources)}")
#     if initial_atypical_sources:
#         print("Atypical source nodes:", ", ".join(sorted(initial_atypical_sources)))
#
#     switches_count = 0
#     # Process edges
#     edges_to_orient = list(G_undir.edges(data=True))
#
#     # First pass: source-connected edges
#     for u, v, data in edges_to_orient:
#         pipe_name = data.get('name')
#         if pipe_name is None:
#             continue
#
#         initial_direction = initial_directions.get(pipe_name)
#         if initial_direction is None:
#             continue
#
#         score = calculate_edge_orientation_score(u, v, source_distances, pressure_sources)
#
#         if score > 0:
#             G_dir.add_edge(u, v, **data)
#             if wn.get_link(pipe_name):
#                 wn.get_link(pipe_name).start_node_name = u
#                 wn.get_link(pipe_name).end_node_name = v
#                 # Check if this is different from initial direction
#                 if initial_direction != (u, v):
#                     print_edge_switch(initial_direction[0], initial_direction[1], data, score)
#                     switches_count += 1
#         else:
#             G_dir.add_edge(v, u, **data)
#             if wn.get_link(pipe_name):
#                 wn.get_link(pipe_name).start_node_name = v
#                 wn.get_link(pipe_name).end_node_name = u
#                 # Check if this is different from initial direction
#                 if initial_direction != (v, u):
#                     print_edge_switch(initial_direction[0], initial_direction[1], data, score)
#                     switches_count += 1
#
#     # Find final atypical sources
#     final_atypical_sources = find_atypical_sources(G_dir)
#
#     # Print summary
#     print(f"\nOrientation Summary:")
#     print(f"Total edges switched: {switches_count}")
#     print(f"\nBefore orientation:")
#     print(f"  Typical sources: {len(pressure_sources)}")
#     print(f"  Atypical sources: {len(initial_atypical_sources)}")
#     if initial_atypical_sources:
#         print("  Atypical source nodes:", ", ".join(sorted(initial_atypical_sources)))
#
#     print(f"\nAfter orientation:")
#     print(f"  Typical sources: {len(pressure_sources)}")
#     print(f"  Atypical sources: {len(final_atypical_sources)}")
#     if final_atypical_sources:
#         print("  Atypical source nodes:", ", ".join(sorted(final_atypical_sources)))
#
#     # Print new atypical sources
#     new_atypical = final_atypical_sources - initial_atypical_sources
#     if new_atypical:
#         print(f"\nNewly created atypical sources:", ", ".join(sorted(new_atypical)))
#
#     return G_dir, wn

# @timer_decorator
# def calculate_shapley_values(G, goal, goal_args, agg_func):
#     """
#     Calculate Shapley values for nodes based on their contribution to the goal function.
#
#     Parameters:
#     -----------
#     G : NetworkX graph
#         The input graph
#     goal : function
#         Function that takes a graph and returns a value
#     goal_args : dict
#         Additional arguments to pass to the goal function
#     agg_func : function
#         Function to aggregate multiple values if goal returns a collection
#
#     Returns:
#     --------
#     dict
#         Dictionary of Shapley values for each node
#     """
#     nodes = list(G.nodes())
#     n = len(nodes)
#     shapley_values = {node: 0.0 for node in nodes}
#
#     # For small graphs (<= 10 nodes), calculate exact Shapley values
#     if n <= 10:
#         # Create a dictionary to cache coalition values
#         coalition_values = {}
#
#         # Calculate value for every possible coalition
#         for r in range(n + 1):
#             for coalition in itertools.combinations(nodes, r):
#                 coalition_set = frozenset(coalition)
#                 if not coalition:
#                     coalition_values[coalition_set] = 0
#                 else:
#                     try:
#                         subgraph = G.subgraph(coalition)
#
#                         # Update sources for this coalition
#                         updated_goal_args = goal_args.copy()
#                         if 'sources' in updated_goal_args:
#                             updated_sources = [s for s in updated_goal_args['sources'] if s in coalition]
#                             updated_goal_args['sources'] = updated_sources
#                         if 'srcs' in updated_goal_args:
#                             updated_sources = [s for s in updated_goal_args['srcs'] if s in coalition]
#                             updated_goal_args['srcs'] = updated_sources
#
#                         goal_result = goal(subgraph, **updated_goal_args)
#                         # Handle if goal returns a collection that needs aggregation
#                         if isinstance(goal_result, (list, tuple, set, np.ndarray)):
#                             coalition_values[coalition_set] = agg_func(goal_result)
#                         elif isinstance(goal_result, dict):
#                             # If goal returns a dictionary, apply aggregation to values
#                             coalition_values[coalition_set] = agg_func(goal_result.values())
#                         elif isinstance(goal_result, pd.Series) or isinstance(goal_result, pd.DataFrame):
#                             # If goal returns a pandas Series or DataFrame, convert to a single value
#                             coalition_values[coalition_set] = agg_func(goal_result.values.flatten())
#                         else:
#                             coalition_values[coalition_set] = goal_result
#                     except:
#                         coalition_values[coalition_set] = 0
#
#         # Calculate Shapley value for each node
#         for node in nodes:
#             other_nodes = [n for n in nodes if n != node]
#
#             # For each possible size of coalition without the node
#             for r in range(n):
#                 for coalition in itertools.combinations(other_nodes, r):
#                     coalition_set = frozenset(coalition)
#
#                     # Get value without the node
#                     val_without = coalition_values[coalition_set]
#
#                     # Get value with the node added
#                     with_node = frozenset(coalition_set.union({node}))
#                     val_with = coalition_values[with_node]
#
#                     # Calculate marginal contribution
#                     marginal = val_with - val_without
#
#                     # Weight by the probability of this coalition occurring
#                     weight = factorial(r) * factorial(n - r - 1) / factorial(n)
#                     shapley_values[node] += weight * marginal
#     else:
#         # For larger graphs, use Monte Carlo sampling
#         num_samples = min(1000, factorial(n))
#
#         for _ in range(num_samples):
#             # Generate random permutation of nodes
#             permutation = np.random.sample(nodes, size=n)
#
#             coalition = set()
#             prev_value = 0
#
#             for node in permutation:
#                 # Calculate value without this node
#                 if coalition:
#                     try:
#                         subgraph = G.subgraph(coalition)
#
#                         # Update sources for this coalition
#                         updated_goal_args = goal_args.copy()
#                         if 'sources' in updated_goal_args:
#                             updated_sources = [s for s in updated_goal_args['sources'] if s in coalition]
#                             updated_goal_args['sources'] = updated_sources
#                         if 'srcs' in updated_goal_args:
#                             updated_sources = [s for s in updated_goal_args['srcs'] if s in coalition]
#                             updated_goal_args['srcs'] = updated_sources
#
#                         goal_result = goal(subgraph, **updated_goal_args)
#                         # Handle if goal returns a collection that needs aggregation
#                         if isinstance(goal_result, (list, tuple, set, np.ndarray)):
#                             val_without = agg_func(goal_result)
#                         elif isinstance(goal_result, dict):
#                             # If goal returns a dictionary, apply aggregation to values
#                             val_without = agg_func(goal_result.values())
#                         elif isinstance(goal_result, pd.Series) or isinstance(goal_result, pd.DataFrame):
#                             # If goal returns a pandas Series or DataFrame, convert to a single value
#                             val_without = agg_func(goal_result.values.flatten())
#                         else:
#                             val_without = goal_result
#                     except:
#                         val_without = prev_value
#                 else:
#                     val_without = 0
#
#                 # Add node to coalition
#                 coalition.add(node)
#
#                 # Calculate value with this node
#                 try:
#                     subgraph = G.subgraph(coalition)
#
#                     # Update sources for this coalition
#                     updated_goal_args = goal_args.copy()
#                     if 'sources' in updated_goal_args:
#                         updated_sources = [s for s in updated_goal_args['sources'] if s in coalition]
#                         updated_goal_args['sources'] = updated_sources
#                     if 'srcs' in updated_goal_args:
#                         updated_sources = [s for s in updated_goal_args['srcs'] if s in coalition]
#                         updated_goal_args['srcs'] = updated_sources
#
#                     goal_result = goal(subgraph, **updated_goal_args)
#                     # Handle if goal returns a collection that needs aggregation
#                     if isinstance(goal_result, (list, tuple, set, np.ndarray)):
#                         val_with = agg_func(goal_result)
#                     elif isinstance(goal_result, dict):
#                         # If goal returns a dictionary, apply aggregation to values
#                         val_with = agg_func(goal_result.values())
#                     elif isinstance(goal_result, pd.Series) or isinstance(goal_result, pd.DataFrame):
#                         # If goal returns a pandas Series or DataFrame, convert to a single value
#                         val_with = agg_func(goal_result.values.flatten())
#                     else:
#                         val_with = goal_result
#                 except:
#                     val_with = val_without
#
#                 # Update Shapley value with marginal contribution
#                 shapley_values[node] += (val_with - val_without) / num_samples
#
#                 prev_value = val_with
#
#     return shapley_values


# def check_and_transform_to_directed_acyclic_graph(G):
#     """
#     Check if graph is directed and acyclic, transform if needed.
#
#     Parameters:
#     - G: NetworkX graph
#
#     Returns:
#     - DAG: NetworkX directed acyclic graph
#     """
#     if not isinstance(G, nx.DiGraph):
#         G = nx.DiGraph(G)
#
#     if not nx.is_directed_acyclic_graph(G):
#         # Find cycles and break them by removing edges with minimum weight
#         cycles = list(nx.simple_cycles(G))
#         while cycles:
#             cycle = cycles[0]
#             # Get edge weights in the cycle
#             cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)])
#                            for i in range(len(cycle))]
#             # Remove the edge with minimum weight
#             edge_to_remove = min(cycle_edges,
#                                  key=lambda e: G.edges[e].get('weight', 1))
#             G.remove_edge(*edge_to_remove)
#             cycles = list(nx.simple_cycles(G))
#
#     return G


# def demonstrate_graph_visualization_old(wn, centralities):
#     """
#     Creates visualizations of the network:
#     - Pressure visualization (pressure as node color, flowrate as link width, demand as node size)
#     - Individual centrality visualizations (centrality as node color, demand as node size)
#     """
#     print("=== Advanced Graph Visualization ===")
#
#     # Create directory for images if it doesn't exist
#     network_name = os.path.splitext(os.path.basename(wn.name))[0]
#     image_dir = os.path.join('images', network_name)
#     os.makedirs(image_dir, exist_ok=True)
#
#     # Run simulation for pressure and flowrate
#     sim = wntr.sim.EpanetSimulator(wn)
#     results = sim.run_sim()
#
#     # Extract node pressure, link flowrate, and node demand at time = 0
#     node_pressure = results.node['pressure'].loc[0]
#     link_flowrate = results.link['flowrate'].loc[0]
#     node_demand = results.node['demand'].loc[0]
#
#     # Get the graph and positions
#     G = wn.to_graph()
#     pos = nx.get_node_attributes(G, 'pos')
#
#     # First, create a mapping from node pairs to link names
#     link_mapping = {}
#     for link_name, link_obj in wn.links():
#         start_node = link_obj.start_node_name
#         end_node = link_obj.end_node_name
#         link_mapping[(start_node, end_node)] = link_name
#         link_mapping[(end_node, start_node)] = link_name  # Add reverse mapping too
#
#     # Calculate link widths
#     max_flow = max(abs(flow) for flow in link_flowrate)
#     min_width = 1
#     max_width = 5
#
#     # Create edge width dictionary
#     edge_widths = []
#     edges = list(G.edges())  # Get list of edges from graph
#
#     # Debug information
#     print(f"Number of edges in graph: {len(edges)}")
#     print(f"Number of links in mapping: {len(link_mapping)}")
#     print(f"Number of nodes in graph: {len(G.nodes())}")
#
#     for (u, v) in edges:
#         if (u, v) in link_mapping:
#             link_name = link_mapping[(u, v)]
#             if link_name in link_flowrate:
#                 flow = abs(link_flowrate[link_name])
#                 width = min_width + (max_width - min_width) * (flow / max_flow)
#                 edge_widths.append(width)
#             else:
#                 print(f"  No flow data for link {link_name}")
#                 edge_widths.append(min_width)
#         else:
#             print(f"Edge ({u}, {v}) not found in link mapping")
#             edge_widths.append(min_width)
#
#     # Define node categories styling
#     node_categories = {
#         'reservoirs': {'marker': '^', 'color': 'blue', 'size_factor': 3.0},
#         'tanks': {'marker': 's', 'color': 'purple', 'size_factor': 2.5},
#         'valves': {'marker': '*', 'color': 'red', 'size_factor': 2.5},
#         'junctions': {'marker': 'o', 'color': 'black', 'size_factor': 1.0}
#     }
#
#     node_types = {
#         'reservoirs': wn.reservoir_name_list,
#         'tanks': wn.tank_name_list,
#         'valves': wn.valve_name_list,
#         'junctions': wn.junction_name_list
#     }
#
#     # Calculate node sizes based on demand
#     # Filter out NaN values and get valid demands
#     valid_demands = [demand for node, demand in node_demand.items() if not np.isnan(demand)]
#
#     # Define base node sizes for demand categories
#     base_size = 40
#     min_size = 20
#     max_size = 120
#
#     # Create node size mapping based on demand
#     node_sizes = {}
#
#     if valid_demands:
#         # Find maximum demand for scaling
#         max_demand = max(valid_demands)
#
#         if max_demand > 0:
#             # Calculate size for each node based on demand
#             for node in G.nodes():
#                 if node not in node_demand or np.isnan(node_demand[node]) or node_demand[node] <= 0:
#                     # No demand nodes get minimum size
#                     node_sizes[node] = min_size
#                 else:
#                     # Scale size based on demand
#                     size_range = max_size - min_size
#                     demand_ratio = node_demand[node] / max_demand
#                     node_sizes[node] = min_size + size_range * demand_ratio
#         else:
#             # If all demands are zero or negative
#             for node in G.nodes():
#                 node_sizes[node] = min_size
#     else:
#         # If no valid demand data
#         for node in G.nodes():
#             node_sizes[node] = base_size
#
#     # 1. Create pressure + flowrate visualization with demand as node size
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Create a list of node colors and sizes
#     node_colors = []
#     node_size_list = []
#
#     for node in G.nodes():
#         # Get pressure for color
#         if node in node_pressure:
#             node_colors.append(node_pressure[node])
#         else:
#             print(f"Warning: Node {node} not found in pressure data")
#             node_colors.append(0)  # Default value
#
#         # Get size based on demand
#         if node in node_sizes:
#             node_size_list.append(node_sizes[node])
#         else:
#             node_size_list.append(min_size)  # Default size
#
#     # Plot nodes with pressure as color and demand as size
#     node_collection = nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_colors,
#         node_size=node_size_list,
#         cmap=plt.cm.coolwarm,
#         vmin=node_pressure.min(),
#         vmax=node_pressure.max(),
#         ax=ax
#     )
#
#     # Plot edges with varying widths
#     nx.draw_networkx_edges(
#         G, pos,
#         width=edge_widths,
#         edge_color='gray',
#         alpha=0.7,
#         ax=ax
#     )
#
#     # Add colorbar for pressure
#     plt.colorbar(node_collection, label='Pressure (m)')
#
#     # Add legend for flowrate scale
#     flow_legend_elements = [
#         Line2D([0], [0], color='gray', linewidth=min_width, label='Min Flow'),
#         Line2D([0], [0], color='gray', linewidth=(min_width + max_width) / 2, label='Med Flow'),
#         Line2D([0], [0], color='gray', linewidth=max_width, label='Max Flow')
#     ]
#
#     # Add legend for demand scale
#     demand_legend_elements = [
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
#                markersize=np.sqrt(min_size) / 2, label='No Demand'),
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
#                markersize=np.sqrt((min_size + max_size) / 2) / 2, label='Medium Demand'),
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
#                markersize=np.sqrt(max_size) / 2, label='High Demand')
#     ]
#
#     # Add node category markers for special nodes
#     for category, node_list in node_types.items():
#         if category in ['reservoirs', 'tanks', 'valves']:  # Only mark special nodes
#             if not node_list:
#                 continue
#             valid_nodes = [node for node in node_list if node in pos]
#             if not valid_nodes:
#                 continue
#             x_coords, y_coords = zip(*[pos[node] for node in valid_nodes])
#             ax.scatter(x_coords, y_coords,
#                        marker=node_categories[category]['marker'],
#                        c=node_categories[category]['color'],
#                        s=[node_sizes[node] * node_categories[category]['size_factor'] for node in valid_nodes],
#                        label=category.capitalize(),
#                        edgecolor='k')
#
#     # Add legends
#     flow_legend = ax.legend(handles=flow_legend_elements, title="Flow Rate",
#                             loc="upper left", fontsize=10)
#     ax.add_artist(flow_legend)
#
#     demand_legend = ax.legend(handles=demand_legend_elements, title="Node Demand",
#                               loc="lower left", fontsize=10)
#     ax.add_artist(demand_legend)
#
#     node_legend = ax.legend(title="Special Nodes", loc="upper right", fontsize=10)
#     ax.add_artist(node_legend)
#
#     plt.title("Pressure and Flowrate Distribution (Node Size = Demand)")
#     filename = os.path.join(image_dir, "pressure_flowrate_demand.png")
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved visualization: {filename}")
#     plt.close()
#
#     # 2. Create individual centrality visualizations with demand as node size
#     for centrality_name, centrality_values in centralities.items():
#         # Check for missing nodes
#         missing_nodes = [node for node in G.nodes() if node not in centrality_values.index]
#         if missing_nodes:
#             print(f"Warning: {len(missing_nodes)} nodes missing from {centrality_name} centrality")
#             print(f"First few missing nodes: {missing_nodes[:5]}")
#
#             # Skip this centrality if too many nodes are missing
#             if len(missing_nodes) > len(G.nodes()) / 2:
#                 print(f"Skipping {centrality_name} visualization due to too many missing nodes")
#                 continue
#
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         # Create a list of node colors and sizes
#         node_colors = []
#         node_size_list = []
#
#         for node in G.nodes():
#             # Get centrality for color
#             if node in centrality_values.index:
#                 node_colors.append(centrality_values[node])
#             else:
#                 # Use the minimum value for missing nodes
#                 node_colors.append(centrality_values.min())
#
#             # Get size based on demand
#             if node in node_sizes:
#                 node_size_list.append(node_sizes[node])
#             else:
#                 node_size_list.append(min_size)  # Default size
#
#         # Get valid min/max values for centrality
#         valid_values = [val for val in node_colors if not np.isnan(val) and not np.isinf(val)]
#         if not valid_values:
#             print(f"Skipping {centrality_name} visualization: no valid values")
#             plt.close()
#             continue
#
#         vmin = min(valid_values)
#         vmax = max(valid_values)
#
#         # Plot nodes with centrality as color and demand as size
#         node_collection = nx.draw_networkx_nodes(
#             G, pos,
#             node_color=node_colors,
#             node_size=node_size_list,
#             cmap=plt.cm.OrRd,
#             vmin=vmin,
#             vmax=vmax,
#             ax=ax
#         )
#
#         # Plot edges (uniform style)
#         nx.draw_networkx_edges(
#             G, pos,
#             edge_color='gray',
#             width=1.0,
#             alpha=0.3,
#             ax=ax
#         )
#
#         # Add colorbar for centrality
#         plt.colorbar(node_collection,
#                      label=f"{centrality_name.replace('_', ' ').title()} Value")
#
#         # Add node category markers for special nodes
#         for category, node_list in node_types.items():
#             if category in ['reservoirs', 'tanks', 'valves']:  # Only mark special nodes
#                 if not node_list:
#                     continue
#                 valid_nodes = [node for node in node_list if node in pos]
#                 if not valid_nodes:
#                     continue
#                 x_coords, y_coords = zip(*[pos[node] for node in valid_nodes])
#                 ax.scatter(x_coords, y_coords,
#                            marker=node_categories[category]['marker'],
#                            c=node_categories[category]['color'],
#                            s=[node_sizes[node] * node_categories[category]['size_factor'] for node in valid_nodes],
#                            label=category.capitalize(),
#                            edgecolor='k')
#
#         # Add legends
#         demand_legend = ax.legend(handles=demand_legend_elements, title="Node Demand",
#                                   loc="lower left", fontsize=10)
#         ax.add_artist(demand_legend)
#
#         node_legend = ax.legend(title="Special Nodes", loc="upper right", fontsize=10)
#         ax.add_artist(node_legend)
#
#         plt.title(f"{centrality_name.replace('_', ' ').title()} Centrality (Node Size = Demand)")
#         filename = os.path.join(image_dir, f"{centrality_name}_centrality_demand.png")
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Saved visualization: {filename}")
#         plt.close()
#
#     print("All visualizations complete.\n")