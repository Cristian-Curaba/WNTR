#!/usr/bin/env python
"""
main.py
"""

import plotly
from typing import List, Dict, Set, Union, Any

from emergency_pre_analysis.hydraulic import *
from emergency_pre_analysis.custom_centrality_indexes import *
from emergency_pre_analysis.utils import *

from wntr.morph import skeletonize
from wntr.network import WaterNetworkModel
from wntr.sim import SimulationResults
import wntr
import os


def load_network_from_inp(inp_path) -> WaterNetworkModel:
    """
    Loads an EPANET INP file and returns a WaterNetworkModel instance.
    """
    print(f"Loading INP file: {inp_path}")
    wn = wntr.network.WaterNetworkModel(inp_path)
    print("...done.\n")
    return wn

def print_basic_network_info(wn) -> None:
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

def calculate_network_metrics(wn):
    """
    Calculate advanced centrality metrics using NetworkX for the water network.
    Returns a dictionary of DataFrames (or Series) containing centrality values for each node.
    """
    print("=== Calculating Network Centralities ===")

    # Get the *directed* NetworkX graph from the WNTR network
    G = wn.to_graph()

    # Also create an undirected copy for metrics that require undirected graphs
    G_undirected = G.to_undirected()

    # Initialize dictionary to store centrality results
    centralities = {}

    # # Calculate metrics for undirected graph
    # centralities.update(calculate_undirected_metrics(G_undirected, wn))

    # Calculate metrics for directed graph
    centralities.update(calculate_directed_metrics(G, wn))

    # Calculate global network metrics
    calculate_global_metrics(G, G_undirected, wn)

    print("\nMetrics calculation complete.\n")
    return centralities

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


def complete_graph_visualization(
    wn: WaterNetworkModel,
    centralities: Dict[str, pd.Series],
    simulation_results: Any,
    pressure_time: int = -1,
    flow_time: int = -1
):
    """
    Creates network visualization with special symbols for different node types,
    including sources, and improved node sizing based on demands.

    Parameters remain the same as the original function.
    """
    # Create directory for images
    network_name = os.path.splitext(os.path.basename(wn.name))[0]
    image_dir = os.path.join('images', network_name)
    os.makedirs(image_dir, exist_ok=True)

    # Get source nodes that are not reservoirs or tanks
    source_nodes = set(get_source_nodes(wn)[0])
    reservoir_tank_nodes = set(node_name for node_name, node in wn.nodes()
                             if node.node_type in ['Reservoir', 'Tank'])
    pure_source_nodes = source_nodes - reservoir_tank_nodes

    # Get pressures and flows with better error handling
    pressure_dict = {}
    try:
        if hasattr(simulation_results.node['pressure'], 'iloc'):
            pressures = simulation_results.node['pressure'].iloc[pressure_time, :]
        else:
            pressures = simulation_results.node['pressure'][pressure_time]

        if isinstance(pressures, pd.Series):
            pressure_dict = pressures.to_dict()
        else:
            pressure_dict = {node: pressures[node] for node in wn.node_name_list}
    except Exception as e:
        print(f"Warning: Could not process pressure data: {str(e)}")
        pressure_dict = {node: 0 for node in wn.node_name_list}

    # Get flows with better error handling
    flow_dict = {}
    try:
        if hasattr(simulation_results.link['flowrate'], 'iloc'):
            flows = simulation_results.link['flowrate'].iloc[flow_time, :]
        else:
            flows = simulation_results.link['flowrate'][flow_time]

        if isinstance(flows, pd.Series):
            flow_dict = flows.to_dict()
        else:
            flow_dict = {link: flows[link] for link in wn.link_name_list}
    except Exception as e:
        print(f"Warning: Could not process flow data: {str(e)}")
        flow_dict = {link: 0 for link in wn.link_name_list}

    # Create node type and size dictionaries
    node_size_dict = {}
    node_symbol_dict = {}
    node_color_dict = {}

    # Define symbols and colors for special nodes
    SPECIAL_NODE_TYPES = {
        'Reservoir': {'symbol': 'square', 'color': 'blue', 'size': 15},
        'Tank': {'symbol': 'diamond', 'color': 'green', 'size': 15},
        'Pump': {'symbol': 'triangle-up', 'color': 'red', 'size': 15},
        'Source': {'symbol': 'star', 'color': 'orange', 'size': 15}
    }

    # Process nodes and their demands
    junction_demands = []
    for node_name, node in wn.nodes():
        # Set symbol and color based on node type
        node_type = node.node_type
        is_pure_source = node_name in pure_source_nodes

        if node_type in SPECIAL_NODE_TYPES:
            node_symbol_dict[node_name] = SPECIAL_NODE_TYPES[node_type]['symbol']
            node_color_dict[node_name] = SPECIAL_NODE_TYPES[node_type]['color']
            node_size_dict[node_name] = SPECIAL_NODE_TYPES[node_type]['size']
        elif is_pure_source:
            node_symbol_dict[node_name] = SPECIAL_NODE_TYPES['Source']['symbol']
            node_color_dict[node_name] = SPECIAL_NODE_TYPES['Source']['color']
            node_size_dict[node_name] = SPECIAL_NODE_TYPES['Source']['size']
        else:  # Junction
            node_symbol_dict[node_name] = 'circle'
            node_color_dict[node_name] = 'gray'

            # Get demand for junction sizing
            try:
                if node.base_demand is None:
                    base_demand = 0
                elif isinstance(node.base_demand, list):
                    base_demand = sum(demand_tuple[0] for demand_tuple in node.base_demand)
                else:
                    base_demand = node.base_demand

                junction_demands.append((node_name, abs(base_demand)))
            except:
                junction_demands.append((node_name, 0))

    # Scale junction sizes based on demands
    if junction_demands:
        demands = [d for _, d in junction_demands]
        max_demand = max(demands) if demands else 0
        min_demand = min(demands) if demands else 0

        for node_name, demand in junction_demands:
            if max_demand == min_demand:
                node_size_dict[node_name] = 5
            elif demand == 0:
                node_size_dict[node_name] = 5
            else:
                scaled_size = 5 + 10 * (demand - min_demand) / (max_demand - min_demand)
                node_size_dict[node_name] = scaled_size

    # Process flows for link widths
    if flow_dict:
        min_flow = abs(min(flow_dict.values(), key=abs))
        max_flow = abs(max(flow_dict.values(), key=abs))

        if (max_flow - min_flow) == 0:
            scaled_flows = {link: 1 for link in flow_dict}
        else:
            scaled_flows = {
                link: 1 + 7 * (abs(val) - min_flow) / (max_flow - min_flow)
                for link, val in flow_dict.items()
            }
    else:
        scaled_flows = {link: 1 for link in wn.link_name_list}

    # Create node popup info
    node_popup_info = pd.DataFrame(index=wn.node_name_list)
    node_popup_info['Node Type'] = ['Source' if node in pure_source_nodes
                                   else wn.get_node(node).node_type
                                   for node in wn.node_name_list]

    # Plot network with all attributes
    plot_interactive_network_with_links(
        wn,
        node_attribute=pressure_dict,
        node_attribute_name="Pressure",
        title="Network: Pressures and Flow",
        node_size_dict=node_size_dict,
        node_symbol_dict=node_symbol_dict,
        link_width=scaled_flows,
        add_to_node_popup=node_popup_info,
        filename=os.path.join(image_dir, "flow_and_pressure.html"),
        auto_open=False
    )

    # Plot centrality measures
    for centrality_name, centrality_series in centralities.items():
        if isinstance(centrality_series, dict):
            centrality_dict = centrality_series
        else:
            centrality_dict = centrality_series.to_dict()

        plot_interactive_network_with_links(
            wn,
            node_attribute=centrality_dict,
            node_attribute_name=centrality_name,
            title=f"Centrality: {centrality_name}",
            node_size_dict=node_size_dict,
            node_symbol_dict=node_symbol_dict,
            filename=os.path.join(image_dir, f"centrality_{centrality_name}.html"),
            auto_open=False
        )

    print(f"Plots generated successfully in {image_dir}.")


def plot_interactive_network_with_links(wn, node_attribute=None, node_attribute_name='Value',
                                      title=None, node_size=8, node_size_dict=None,
                                      node_symbol_dict=None,
                                      node_range=[None, None], node_cmap='Jet',
                                      node_labels=True, link_width=1,
                                      add_colorbar=True, reverse_colormap=False,
                                      figsize=[700, 450], round_ndigits=2,
                                      add_to_node_popup=None, filename='plotly_network.html',
                                      auto_open=True):
    """
        Interactive network plot with variable line widths, edge orientation, and variable node sizes.

        Parameters:
        -----------
        wn : wntr.network.WaterNetworkModel
            Water network model
        node_attribute : dict, optional
            Dictionary of node attributes
        node_attribute_name : str, optional
            Name of node attribute
        title : str, optional
            Plot title
        node_size : int, optional
            Default node size
        node_size_dict : dict, optional
            Dictionary of node sizes {node_name: size}
        node_range : list, optional
            Node attribute range
        node_cmap : str, optional
            Node colormap
        node_labels : bool, optional
            If True, include node labels in hover text
        link_width : int or dict, optional
            Link width or dictionary of link widths {link_name: width}
        add_colorbar : bool, optional
            If True, add colorbar
        reverse_colormap : bool, optional
            If True, reverse colormap
        figsize : list, optional
            Figure size
        round_ndigits : int, optional
            Number of digits to round node attribute values
        add_to_node_popup : pandas.DataFrame, optional
            Additional node information to include in hover text
        filename : str, optional
            Filename to save HTML file
        auto_open : bool, optional
            If True, open HTML file after creating

        Returns:
        --------
        plotly.graph_objs.Figure
        """
    if plotly is None:
        raise ImportError('plotly is required')

    G = wn.to_graph()

    # Process node_attribute
    if node_attribute is not None:
        if isinstance(node_attribute, list):
            node_cmap = 'Reds'
            add_colorbar = False
        node_attribute = _format_node_attribute(node_attribute, wn)
    else:
        add_colorbar = False

    # Create edge traces
    node_pair_to_link = {}
    for link_name, link in wn.links():
        node_pair_to_link[(link.start_node_name, link.end_node_name)] = link_name

    edge_traces = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        link_name = node_pair_to_link.get(edge)

        # Handle link width based on type
        if isinstance(link_width, dict):
            w = link_width.get(link_name, 1)
            flow_value = link_width.get(link_name, 0)
        else:
            w = link_width
            flow_value = 0

        # Create hover text
        if link_name:
            info = f"Pipe: {link_name}<br>Flow: {flow_value:.2f}"
        else:
            info = f"Edge: {edge[0]} → {edge[1]}"

        edge_traces.append(
            plotly.graph_objs.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                text=info,
                hoverinfo='text',
                mode='lines+markers',
                line=dict(color='#888', width=w),
                marker=dict(
                    symbol='arrow',
                    size=8,
                    color='#888',
                    angleref='previous'
                ),
                showlegend=False
            )
        )

    # Create separate node traces for different symbols
    node_traces = {}

    # Process nodes
    for node in G.nodes():
        x, y = G.nodes[node]['pos']

        # Determine node symbol
        symbol = (node_symbol_dict.get(node, 'circle')
                  if node_symbol_dict is not None else 'circle')

        # Create new trace for symbol if it doesn't exist
        if symbol not in node_traces:
            node_traces[symbol] = {
                'x': [], 'y': [],
                'sizes': [],
                'colors': [],
                'hover_texts': [],
                'nodes': []  # Store node names for attribute mapping
            }

        # Add node data to appropriate trace
        trace_data = node_traces[symbol]
        trace_data['x'].append(x)
        trace_data['y'].append(y)
        trace_data['nodes'].append(node)

        # Determine node size
        size = (node_size_dict.get(node, node_size)
                if node_size_dict is not None else node_size)
        trace_data['sizes'].append(size)

        # Determine node color based on attribute
        if node_attribute is not None and node in node_attribute:
            trace_data['colors'].append(node_attribute[node])
        else:
            trace_data['colors'].append(0)

        # Create hover text
        hover_text = []
        if node_labels:
            hover_text.append(f"Node: {node}")
        if node_size_dict and node in node_size_dict:
            hover_text.append(f"Size: {size:.2f}")
        if node_attribute is not None and node in node_attribute:
            val = node_attribute[node]
            val_str = (f"{val:.{round_ndigits}f}"
                       if isinstance(val, (int, float)) else str(val))
            hover_text.append(f"{node_attribute_name}: {val_str}")
        if add_to_node_popup is not None and node in add_to_node_popup.index:
            for col in add_to_node_popup.columns:
                val = add_to_node_popup.loc[node, col]
                if not pd.isna(val):
                    hover_text.append(f"{col}: {val}")

        trace_data['hover_texts'].append('<br>'.join(hover_text))

    # Create Scatter traces for each symbol group
    node_scatter_traces = []
    for symbol, trace_data in node_traces.items():
        scatter = plotly.graph_objs.Scatter(
            x=trace_data['x'],
            y=trace_data['y'],
            mode='markers',
            hoverinfo='text',
            hovertext=trace_data['hover_texts'],
            marker=dict(
                symbol=symbol,
                size=trace_data['sizes'],
                color=trace_data['colors'],  # Always use the colors list
                colorscale=node_cmap if node_attribute is not None else None,
                reversescale=reverse_colormap,
                showscale=add_colorbar and symbol == 'circle',  # Only show colorbar for main trace
                colorbar=dict(
                    thickness=15,
                    title=node_attribute_name,
                    xanchor='left',
                    titleside='right'
                ) if add_colorbar and symbol == 'circle' else None,
                line=dict(width=2)
            ),
            showlegend=False
        )

        # Set color range if specified
        if node_attribute is not None:
            if node_range[0] is not None:
                scatter.marker.cmin = node_range[0]
            if node_range[1] is not None:
                scatter.marker.cmax = node_range[1]

        node_scatter_traces.append(scatter)
    # Create figure
    fig = plotly.graph_objs.Figure(
        data=edge_traces + node_scatter_traces,
        layout=plotly.graph_objs.Layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=False,
            width=figsize[0],
            height=figsize[1],
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # Save and show
    plotly.offline.plot(fig, filename=filename, auto_open=auto_open)

    return fig


def _format_node_attribute(node_attribute, wn):
    """
    Helper function that formats node_attribute into a node->value dict.
    """
    if isinstance(node_attribute, str):
        # If node_attribute is a string, query wn
        # Something like: node_attribute = wn.query_node_attribute(node_attribute)
        # For demonstration purposes, just return a dict of random values
        return {node_name: 1.0 for node_name in wn.node_name_list}
    elif isinstance(node_attribute, list):
        # If it's a list of node names, assign '1.0' to each
        attr_dict = {}
        for node in node_attribute:
            attr_dict[node] = 1.0
        return attr_dict
    elif isinstance(node_attribute, pd.Series):
        return node_attribute.to_dict()
    elif isinstance(node_attribute, dict):
        return node_attribute
    else:
        return {node_name: 0.0 for node_name in wn.node_name_list}

def main():
    # Path to an example INP file. Adjust as needed.
    inp_file = os.path.join('examples', 'networks', 'Zampis.inp')

    # 1. Load Network
    wn = load_network_from_inp(inp_file)

    # 2. Print Basic Info
    print_basic_network_info(wn)

    # 3. EPANET Simulation
    epanet_results = run_epanet_simulation(wn)
    analyze_hydraulic_results(epanet_results, node_name='1')  # Replace with valid node

    # 4. WNTR Simulation
    # wntr_results = run_wntr_simulation(wn)
    # analyze_hydraulic_results(wntr_results, node_name='1')

    # 5. Resilience Metrics
    # calculate_tondini(wn, wntr_results)

    # 6. (Optional) Water Quality Simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    print(results)
    print("Water quality simulation complete.\n")

    # 7. Morphological transformations (skeletonization)

    # Calculate the median pipe diameter
    pipe_diameters = [pipe.diameter for _, pipe in wn.pipes()]
    median_diameter = np.median(pipe_diameters)

    # Perform skeletonization using the median diameter as threshold
    wn_skel = skeletonize(wn, pipe_diameter_threshold= median_diameter, return_copy=True)
    print("Skeletonized Network!")
    print_basic_network_info(wn_skel)

    # 8. Scenario-based analysis (pipe break, demand changes, etc.)
    demonstrate_scenario_based_analysis(wn)
    demonstrate_demand_change(wn)

    # 9. Improve edge orientation
    # Get initial atypical nodes
    _, initial_atypical = get_source_nodes(wn)
    try:
        num_reoriented, remaining_atypical = reorient_edges_by_pressure(wn, verbose=True)
        print(f"\nSummary:")
        print(f"- Reoriented {num_reoriented} edges")
        print(f"- Reduced atypical nodes from {len(initial_atypical)} to {len(remaining_atypical)}")
    except Exception as e:
        print(f"Error during edge reorientation: {str(e)}")



    # 10. Graph Visualization
    centralities = calculate_network_metrics(wn)
    complete_graph_visualization(wn, centralities, simulation_results=results)

    print("\n=== Finished all demonstrations! ===")

if __name__ == "__main__":
    main()
