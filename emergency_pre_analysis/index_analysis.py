# index_analisys.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from emergency_pre_analysis.utils import create_directory_structure

def analyze_distributions(centralities, base_dir='images/analysis'):
    """
    For each scenario and its associated centralities, plot the distributions.
    `centralities` should be a dict of {centrality_name: pd.Series}.
    Plots are saved to the scenario folder if the centrality is listed.
    """
    scenarios = create_directory_structure(base_dir)

    for scenario, centrality_list in scenarios.items():
        for c_name in centrality_list:
            if c_name in centralities:
                series = centralities[c_name]

                # Check if weighted or unweighted
                folder_type = 'unweighted'
                if hasattr(series, 'attrs'):
                    weight = series.attrs.get('weight', None)
                    if weight is not None:
                        folder_type = 'weighted'

                # Remove missing values
                cleaned_series = series.dropna()

                # If the data is constant or empty, skip
                if len(cleaned_series.unique()) <= 1:
                    print(f"{c_name} in {scenario}/{folder_type} has constant or empty data. Skipping.")
                    continue

                # Create distribution plot
                plt.figure(figsize=(8, 5))
                sns.histplot(cleaned_series, kde=True, color='blue', bins='auto')
                plt.title(f"Distribution of {c_name}")
                plt.xlabel(c_name)
                plt.ylabel("Frequency")

                # Save figure
                plot_path = os.path.join(base_dir, scenario, folder_type, f"{c_name}_distribution.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()

def correlation_analysis(centralities, base_dir='images/analysis', correlation_threshold=0.8):
    """
    Compute and plot a correlation analysis of centralities.
    'similar' centralities can be defined based on the correlation_threshold.
    Saves a heatmap to the base_dir.
    """
    # Convert centralities dict of Series -> DataFrame
    df = pd.DataFrame({name: s for name, s in centralities.items() if isinstance(s, pd.Series)})

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Centralities")
    heatmap_path = os.path.join(base_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    # Determine which pairs are 'similar' based on threshold
    similar_pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            if abs(corr_matrix.iloc[i, j]) >= correlation_threshold:
                similar_pairs.append((columns[i], columns[j], corr_matrix.iloc[i, j]))

    return similar_pairs

