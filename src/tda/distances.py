"""
Persistence Diagram Distance Calculations
Stability metrics for comparing persistence diagrams across different stages
"""

import numpy as np
import gudhi


def wasserstein_distance(diag1, diag2, order=1):
    """Compute Wasserstein distance between two persistence diagrams."""
    try:
        if len(diag1) == 0 and len(diag2) == 0:
            return 0.0

        # Convert to numpy arrays if needed
        if isinstance(diag1, list):
            diag1 = np.array(diag1)
        if isinstance(diag2, list):
            diag2 = np.array(diag2)

        # Handle empty diagrams
        if len(diag1) == 0:
            diag1 = np.array([[0, 0]])
        if len(diag2) == 0:
            diag2 = np.array([[0, 0]])

        return gudhi.wasserstein.wasserstein_distance(diag1, diag2, order=order, internal_p=order)
    except Exception as e:
        print(f"Warning: Wasserstein distance calculation failed: {e}")
        return float('inf')


def compute_all_distances(clean_diag, noisy_diag, denoised_diag, dimensions=[0, 1]):
    """Compute all relevant distances between clean, noisy, and denoised diagrams."""
    distances = {}

    for dim in dimensions:
        h_key = f"H{dim}"

        # Get diagrams for this dimension
        clean_h = clean_diag.get(h_key, [])
        noisy_h = noisy_diag.get(h_key, [])
        denoised_h = denoised_diag.get(h_key, [])

        # Compute Wasserstein distances
        distances[f'wasserstein_clean_noisy_H{dim}'] = wasserstein_distance(clean_h, noisy_h, order=1)
        distances[f'wasserstein_clean_denoised_H{dim}'] = wasserstein_distance(clean_h, denoised_h, order=1)
        distances[f'wasserstein_noisy_denoised_H{dim}'] = wasserstein_distance(noisy_h, denoised_h, order=1)

    return distances
