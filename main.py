#!/usr/bin/env python
"""
main.py
"""

import random
from typing import Dict, Any

import plotly

from emergency_pre_analysis.custom_centrality_indexes import *
from emergency_pre_analysis.hydraulic import *
from emergency_pre_analysis.utils import *
from wntr.morph import skeletonize
from wntr.network import WaterNetworkModel
from emergency_pre_analysis.index_analysis import *


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
    values = list(centrality.values( ))
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': min(values),
        'max': max(values),
        'top_nodes': sorted(centrality.items( ), key=lambda x: x[1], reverse=True)[:5]
    }


def calculate_network_metrics(wn):
    """
    Calculate advanced centrality metrics using NetworkX for the water network.
    Returns a dictionary of DataFrames (or Series) containing centrality values for each node.
    """
    print("=== Calculating Network Centralities ===")

    # Get the *directed* NetworkX graph from the WNTR network
    G = wn.to_graph( )

    # Also create an undirected copy for metrics that require undirected graphs
    G_undirected = G.to_undirected( )

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


def create_visualization_dictionaries(wn, pure_source_nodes = None, flow_dict = None):
    """
    Create dictionaries for node sizes, symbols, colors, and scaled flows for network visualization.

    Parameters:
    -----------
    wn : wntr.network.WaterNetworkModel
        Water network model
    pure_source_nodes : list, optional
        List of node names that are pure sources
    flow_dict : dict, optional
        Dictionary of flow values for links

    Returns:
    --------
    tuple
        (node_size_dict, node_symbol_dict, scaled_flows)
    """
    if pure_source_nodes is None:
        pure_source_nodes = []

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
    for node_name, node in wn.nodes( ):
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
        min_flow = abs(min(flow_dict.values( ), key=abs))
        max_flow = abs(max(flow_dict.values( ), key=abs))

        if (max_flow - min_flow) == 0:
            scaled_flows = {link: 1 for link in flow_dict}
        else:
            scaled_flows = {
                link: 1 + 7 * (abs(val) - min_flow) / (max_flow - min_flow)
                for link, val in flow_dict.items( )
            }
    else:
        scaled_flows = {link: 1 for link in wn.link_name_list}

    return node_size_dict, node_symbol_dict, scaled_flows


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
    reservoir_tank_nodes = set(node_name for node_name, node in wn.nodes( )
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
            pressure_dict = pressures.to_dict( )
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
            flow_dict = flows.to_dict( )
        else:
            flow_dict = {link: flows[link] for link in wn.link_name_list}
    except Exception as e:
        print(f"Warning: Could not process flow data: {str(e)}")
        flow_dict = {link: 0 for link in wn.link_name_list}

    node_sizes, node_symbols, flows = create_visualization_dictionaries(
        wn,
        pure_source_nodes=pure_source_nodes,
        flow_dict=flow_dict
    )

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
        node_size_dict=node_sizes,
        node_symbol_dict=node_symbols,
        link_width=flows,
        add_to_node_popup=node_popup_info,
        filename=os.path.join(image_dir, "flow_and_pressure.html"),
        auto_open=False
    )

    # Plot centrality measures
    scenarios = create_directory_structure(image_dir)

    for centrality_name, centrality_series in centralities.items( ):
        # Helper function to determine the subfolder based on weight_name
        # TODO
        weight_name = None

        def get_weight_subfolder():
            if weight_name is None:
                return 'unweighted'
            return 'weighted'

        if isinstance(centrality_series, dict):
            centrality_dict = centrality_series
        else:
            centrality_dict = centrality_series.to_dict( )

        # Determine the appropriate scenario folder
        target_dir = image_dir  # Default to main directory

        # Check if it's a specific case (contains node reference)
        if '_on_' in centrality_name:
            for scenario in scenarios.keys( ):
                if any(metric in centrality_name for metric in scenarios[scenario]):
                    target_dir = os.path.join(image_dir, scenario, 'specific_cases')
                    break
        else:
            # Check for general scenario cases
            for scenario, metrics in scenarios.items( ):
                if any(metric in centrality_name for metric in metrics):
                    target_dir = os.path.join(image_dir, scenario, get_weight_subfolder( ))
                    break

        # Create the plot in the appropriate directory
        plot_interactive_network_with_links(
            wn,
            node_attribute=centrality_dict,
            node_attribute_name=centrality_name,
            title=f"Centrality: {centrality_name}",
            node_size_dict=node_sizes,
            node_symbol_dict=node_symbols,
            filename=os.path.join(target_dir, f"centrality_{centrality_name}.html"),
            auto_open=False
        )


def plot_interactive_network_with_links(wn, node_attribute = None, node_attribute_name = 'Value',
                                        title = None, node_size = 8, node_size_dict = None,
                                        node_symbol_dict = None,
                                        node_range = [None, None], node_cmap = 'Jet',
                                        node_labels = True, link_width = 1,
                                        add_colorbar = True, reverse_colormap = False,
                                        figsize = [700, 450], round_ndigits = 2,
                                        add_to_node_popup = None, filename = 'plotly_network.html',
                                        auto_open = True):
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

    G = wn.to_graph( )

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
    for link_name, link in wn.links( ):
        node_pair_to_link[(link.start_node_name, link.end_node_name)] = link_name

    edge_traces = []
    for edge in G.edges( ):
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
    for node in G.nodes( ):
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
    for symbol, trace_data in node_traces.items( ):
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
        return node_attribute.to_dict( )
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
    analyze_hydraulic_results(epanet_results, node_name=wn.get_node(
        wn.node_name_list[random.randint(0, 10)]))  # Replace with valid node

    flow_values = epanet_results.link['flowrate']
    flows = flow_values.loc[random.randint(0, 23) * 3600].to_dict( )
    reorient_edges_by_flow(wn, flows=flows)

    # 4. WNTR Simulation
    # wntr_results = run_wntr_simulation(wn)
    # analyze_hydraulic_results(wntr_results, node_name='1')

    # 5. Tondini Index
    try:
        results = epanet_results
        tondini_index = wntr.metrics.hydraulic.todini_index(results.node["head"], results.node["pressure"],
                                                            results.node["demand"], results.link["flowrate"], wn,
                                                            Pstar=15)
        print(f"Todini Resilience Index: {tondini_index.mean( )} (average over all timesteps)")
    except Exception as e:
        print(f"Error computing Todini index: {e}")

    # 7. Morphological transformations (skeletonization)

    # Calculate the median pipe diameter
    pipe_diameters = [pipe.diameter for _, pipe in wn.pipes( )]
    median_diameter = np.median(pipe_diameters)

    # Perform skeletonization using the median diameter as threshold
    wn_skel = skeletonize(wn, pipe_diameter_threshold=median_diameter, return_copy=True)
    print("Skeletonized Network!")
    print_basic_network_info(wn_skel)

    # 8. Scenario-based analysis (pipe break, demand changes, etc.)
    demonstrate_scenario_based_analysis(wn)
    demonstrate_demand_change(wn)

    # 9. Improve edge orientation
    # Get initial atypical nodes
    # _, initial_atypical = get_source_nodes(wn)
    # try:
    #     num_reoriented, remaining_atypical = reorient_edges_by_pressure(wn, verbose=True)
    #     print(f"\nSummary:")
    #     print(f"- Reoriented {num_reoriented} edges")
    #     print(f"- Reduced atypical nodes from {len(initial_atypical)} to {len(remaining_atypical)}")
    # except Exception as e:
    #     print(f"Error during edge reorientation: {str(e)}")

    # 10. Graph Visualization
    centralities = calculate_network_metrics(wn)
    complete_graph_visualization(wn, centralities, simulation_results=results)

    network_name = os.path.splitext(os.path.basename(wn.name))[0]
    image_dir = os.path.join('images', network_name)
    analyze_distributions(centralities, base_dir=image_dir)
    similar_pairs = correlation_analysis(centralities, base_dir=image_dir, correlation_threshold=0.8)
    print("Similar centralities based on the threshold:", similar_pairs)


if __name__ == "__main__":
    main( )
