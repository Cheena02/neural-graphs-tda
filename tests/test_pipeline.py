"""
Test Script for Topological Data Analysis Pipeline

This script tests the full pipeline for topological data analysis on the synthetic data
to ensure all components are working correctly before processing the real datasets.

Author: Cheena Yadav
Date: April 15th, 2025
"""

import os
import sys
import numpy as np
import networkx as nx
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from preprocessing.synthetic_data_generator import generate_synthetic_neural_data
from graph_generation.graph_derivation import (
    derive_correlation_graph,
    derive_mutual_information_graph
)
from src.tda_core import (
    compute_graph_persistence,
    compute_cubical_persistence,
    compare_topological_features,
    visualize_persistence_diagram,
    visualize_betti_curves
)
from visualization.tda_visualization import (
    visualize_graph_with_filtration,
    visualize_persistence_diagram_comparison,
    visualize_betti_curves_comparison,
    visualize_persistence_statistics,
    visualize_topological_feature_heatmap
)

# Create output directory
# output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        #  'results', 'test_results')
# Create output directory INSIDE the current code folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'test_results')
                        
os.makedirs(output_dir, exist_ok=True)

print(f"Starting test pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results will be saved to {output_dir}")

# Generate synthetic data
print("\nGenerating synthetic neural data...")
n_neurons = 50
n_timepoints = 200
n_conditions = 3

data_condition1 = generate_synthetic_neural_data(
    n_neurons=n_neurons,
    n_timepoints=n_timepoints,
    correlation_structure='random',
    noise_level=0.1,
    seed=42
)

data_condition2 = generate_synthetic_neural_data(
    n_neurons=n_neurons,
    n_timepoints=n_timepoints,
    correlation_structure='modular',
    noise_level=0.1,
    seed=42
)

data_condition3 = generate_synthetic_neural_data(
    n_neurons=n_neurons,
    n_timepoints=n_timepoints,
    correlation_structure='hierarchical',
    noise_level=0.1,
    seed=42
)

print(f"Generated data shapes: {data_condition1.shape}, {data_condition2.shape}, {data_condition3.shape}")

# Derive graphs using different methods
print("\nDeriving graphs using different methods...")

# Correlation graphs
corr_graph1 = derive_correlation_graph(data_condition1, threshold=0.3)
corr_graph2 = derive_correlation_graph(data_condition2, threshold=0.3)
corr_graph3 = derive_correlation_graph(data_condition3, threshold=0.3)

print(f"Correlation graphs: {len(corr_graph1.nodes())} nodes, {len(corr_graph1.edges())} edges")
print(f"Correlation graphs: {len(corr_graph2.nodes())} nodes, {len(corr_graph2.edges())} edges")
print(f"Correlation graphs: {len(corr_graph3.nodes())} nodes, {len(corr_graph3.edges())} edges")

# Mutual information graphs
mi_graph1 = derive_mutual_information_graph(data_condition1, threshold=0.1)
mi_graph2 = derive_mutual_information_graph(data_condition2, threshold=0.1)
mi_graph3 = derive_mutual_information_graph(data_condition3, threshold=0.1)

print(f"Mutual information graphs: {len(mi_graph1.nodes())} nodes, {len(mi_graph1.edges())} edges")
print(f"Mutual information graphs: {len(mi_graph2.nodes())} nodes, {len(mi_graph2.edges())} edges")
print(f"Mutual information graphs: {len(mi_graph3.nodes())} nodes, {len(mi_graph3.edges())} edges")

# Compute persistent homology for correlation graphs
print("\nComputing persistent homology for correlation graphs...")
corr_persistence1 = compute_graph_persistence(corr_graph1, weight_attr='weight', homology_dimensions=[0, 1])
corr_persistence2 = compute_graph_persistence(corr_graph2, weight_attr='weight', homology_dimensions=[0, 1])
corr_persistence3 = compute_graph_persistence(corr_graph3, weight_attr='weight', homology_dimensions=[0, 1])

print(f"Persistence pairs dimensions: {[len(pairs) for pairs in corr_persistence1['persistence_pairs']]}")
print(f"Persistence pairs dimensions: {[len(pairs) for pairs in corr_persistence2['persistence_pairs']]}")
print(f"Persistence pairs dimensions: {[len(pairs) for pairs in corr_persistence3['persistence_pairs']]}")

# Visualize persistence diagrams
print("\nVisualizing persistence diagrams...")
visualize_persistence_diagram(
    corr_persistence1,
    "Condition 1 - Correlation Graph Persistence Diagram",
    os.path.join(output_dir, "condition1_corr_persistence_diagram.png")
)

visualize_persistence_diagram(
    corr_persistence2,
    "Condition 2 - Correlation Graph Persistence Diagram",
    os.path.join(output_dir, "condition2_corr_persistence_diagram.png")
)

visualize_persistence_diagram(
    corr_persistence3,
    "Condition 3 - Correlation Graph Persistence Diagram",
    os.path.join(output_dir, "condition3_corr_persistence_diagram.png")
)

# Visualize Betti curves
print("\nVisualizing Betti curves...")
visualize_betti_curves(
    corr_persistence1,
    "Condition 1 - Correlation Graph Betti Curves",
    os.path.join(output_dir, "condition1_corr_betti_curves.png")
)

visualize_betti_curves(
    corr_persistence2,
    "Condition 2 - Correlation Graph Betti Curves",
    os.path.join(output_dir, "condition2_corr_betti_curves.png")
)

visualize_betti_curves(
    corr_persistence3,
    "Condition 3 - Correlation Graph Betti Curves",
    os.path.join(output_dir, "condition3_corr_betti_curves.png")
)

# Compare persistence diagrams
print("\nComparing persistence diagrams...")
visualize_persistence_diagram_comparison(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    0,  # Dimension 0
    "Correlation Graph H0 Persistence Diagram Comparison",
    os.path.join(output_dir, "corr_persistence_diagram_comparison_h0.png")
)

visualize_persistence_diagram_comparison(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    1,  # Dimension 1
    "Correlation Graph H1 Persistence Diagram Comparison",
    os.path.join(output_dir, "corr_persistence_diagram_comparison_h1.png")
)

# Compare Betti curves
print("\nComparing Betti curves...")
visualize_betti_curves_comparison(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    "Correlation Graph Betti Curves Comparison",
    os.path.join(output_dir, "corr_betti_curves_comparison.png")
)

# Extract and compare topological features
print("\nExtracting and comparing topological features...")
comparison_dir = os.path.join(output_dir, "feature_comparison")
os.makedirs(comparison_dir, exist_ok=True)

features_df = compare_topological_features(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    comparison_dir
)

print("\nTopological features comparison:")
print(features_df)

# Visualize graphs with filtration
print("\nVisualizing graphs with filtration...")
# Create filtration values from edge weights
filtration_values1 = {edge: data['weight'] for edge, data in corr_graph1.edges.items()}
filtration_values2 = {edge: data['weight'] for edge, data in corr_graph2.edges.items()}
filtration_values3 = {edge: data['weight'] for edge, data in corr_graph3.edges.items()}

# Add node filtration values (use minimum edge weight connected to node)
for node in corr_graph1.nodes():
    connected_edges = list(corr_graph1.edges(node, data='weight'))
    if connected_edges:
        filtration_values1[node] = min([w for _, _, w in connected_edges])
    else:
        filtration_values1[node] = 0

for node in corr_graph2.nodes():
    connected_edges = list(corr_graph2.edges(node, data='weight'))
    if connected_edges:
        filtration_values2[node] = min([w for _, _, w in connected_edges])
    else:
        filtration_values2[node] = 0

for node in corr_graph3.nodes():
    connected_edges = list(corr_graph3.edges(node, data='weight'))
    if connected_edges:
        filtration_values3[node] = min([w for _, _, w in connected_edges])
    else:
        filtration_values3[node] = 0

visualize_graph_with_filtration(
    corr_graph1,
    filtration_values1,
    "Condition 1 - Correlation Graph with Filtration",
    os.path.join(output_dir, "condition1_corr_graph_filtration.png")
)

visualize_graph_with_filtration(
    corr_graph2,
    filtration_values2,
    "Condition 2 - Correlation Graph with Filtration",
    os.path.join(output_dir, "condition2_corr_graph_filtration.png")
)

visualize_graph_with_filtration(
    corr_graph3,
    filtration_values3,
    "Condition 3 - Correlation Graph with Filtration",
    os.path.join(output_dir, "condition3_corr_graph_filtration.png")
)

# Test cubical complex persistence on synthetic image data
print("\nTesting cubical complex persistence on synthetic image data...")
# Create synthetic image data
image_size = 50
image1 = np.zeros((image_size, image_size))
image2 = np.zeros((image_size, image_size))
image3 = np.zeros((image_size, image_size))

# Add some structures to the images
# Image 1: Random noise
image1 = np.random.rand(image_size, image_size)

# Image 2: Circles
x, y = np.ogrid[:image_size, :image_size]
center1 = (image_size//3, image_size//3)
center2 = (2*image_size//3, 2*image_size//3)
dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
image2[dist1 <= 10] = 1
image2[dist2 <= 15] = 1
image2 += np.random.rand(image_size, image_size) * 0.1

# Image 3: Grid pattern
for i in range(0, image_size, 10):
    image3[i:i+2, :] = 1
    image3[:, i:i+2] = 1
image3 += np.random.rand(image_size, image_size) * 0.1

# Compute cubical persistence
print("\nComputing cubical persistence...")
cubical_persistence1 = compute_cubical_persistence(
    image1, filtration_method='sublevel', sigma=1.0, homology_dimensions=[0, 1, 2]
)

cubical_persistence2 = compute_cubical_persistence(
    image2, filtration_method='sublevel', sigma=1.0, homology_dimensions=[0, 1, 2]
)

cubical_persistence3 = compute_cubical_persistence(
    image3, filtration_method='sublevel', sigma=1.0, homology_dimensions=[0, 1, 2]
)

print(f"Cubical persistence pairs dimensions: {[len(pairs) for pairs in cubical_persistence1['persistence_pairs']]}")
print(f"Cubical persistence pairs dimensions: {[len(pairs) for pairs in cubical_persistence2['persistence_pairs']]}")
print(f"Cubical persistence pairs dimensions: {[len(pairs) for pairs in cubical_persistence3['persistence_pairs']]}")

# Visualize cubical persistence diagrams
print("\nVisualizing cubical persistence diagrams...")
visualize_persistence_diagram(
    cubical_persistence1,
    "Image 1 - Cubical Persistence Diagram",
    os.path.join(output_dir, "image1_cubical_persistence_diagram.png")
)

visualize_persistence_diagram(
    cubical_persistence2,
    "Image 2 - Cubical Persistence Diagram",
    os.path.join(output_dir, "image2_cubical_persistence_diagram.png")
)

visualize_persistence_diagram(
    cubical_persistence3,
    "Image 3 - Cubical Persistence Diagram",
    os.path.join(output_dir, "image3_cubical_persistence_diagram.png")
)

# Compare cubical persistence features
print("\nComparing cubical persistence features...")
cubical_comparison_dir = os.path.join(output_dir, "cubical_feature_comparison")
os.makedirs(cubical_comparison_dir, exist_ok=True)

cubical_features_df = compare_topological_features(
    [cubical_persistence1, cubical_persistence2, cubical_persistence3],
    ["Random", "Circles", "Grid"],
    cubical_comparison_dir
)

print("\nCubical topological features comparison:")
print(cubical_features_df)

# Visualize persistence statistics
print("\nVisualizing persistence statistics...")
visualize_persistence_statistics(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    "Correlation Graph Persistence Statistics",
    os.path.join(output_dir, "corr_persistence_statistics.png")
)

visualize_persistence_statistics(
    [cubical_persistence1, cubical_persistence2, cubical_persistence3],
    ["Random", "Circles", "Grid"],
    "Cubical Persistence Statistics",
    os.path.join(output_dir, "cubical_persistence_statistics.png")
)

# Visualize topological feature heatmap
print("\nVisualizing topological feature heatmap...")
visualize_topological_feature_heatmap(
    [corr_persistence1, corr_persistence2, corr_persistence3],
    ["Random", "Modular", "Hierarchical"],
    "Correlation Graph Topological Features",
    os.path.join(output_dir, "corr_topological_features_heatmap.png")
)

visualize_topological_feature_heatmap(
    [cubical_persistence1, cubical_persistence2, cubical_persistence3],
    ["Random", "Circles", "Grid"],
    "Cubical Topological Features",
    os.path.join(output_dir, "cubical_topological_features_heatmap.png")
)

print(f"\nTest pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"All results saved to {output_dir}")
