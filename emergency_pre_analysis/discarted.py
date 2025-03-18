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

def orient_edges_by_pressure_and_topology(wn, pressure_sources=None):
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