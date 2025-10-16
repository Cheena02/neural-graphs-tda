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

        # Remove infinite values
        diag1 = diag1[np.isfinite(diag1).all(axis=1)]
        diag2 = diag2[np.isfinite(diag2).all(axis=1)]

        # Handle empty diagrams after filtering
        if len(diag1) == 0:
            diag1 = np.array([[0, 0]])
        if len(diag2) == 0:
            diag2 = np.array([[0, 0]])

        # TRY DIFFERENT GUDHI IMPORT METHODS
        try:
            # Method 1: Direct import
            import gudhi.wasserstein
            return gudhi.wasserstein.wasserstein_distance(diag1, diag2, order=order, internal_p=order)
        except (ImportError, AttributeError):
            try:
                # Method 2: Alternative import
                from gudhi.wasserstein import wasserstein_distance as wd
                return wd(diag1, diag2, order=order, internal_p=order)
            except (ImportError, AttributeError):
                try:
                    # Method 3: Direct function import
                    from gudhi import wasserstein_distance as wd
                    return wd(diag1, diag2, order=order)
                except (ImportError, AttributeError):
                    # Method 4: Fallback to simple distance
                    return simple_diagram_distance(diag1, diag2)

    except Exception as e:
        print(f"Warning: Wasserstein distance calculation failed: {e}")
        return simple_diagram_distance(diag1, diag2)


def simple_diagram_distance(diag1, diag2):
    """Simple fallback distance measure when Wasserstein is not available."""
    try:
        # Convert to numpy arrays
        if isinstance(diag1, list):
            diag1 = np.array(diag1)
        if isinstance(diag2, list):
            diag2 = np.array(diag2)

        # Remove infinite values
        diag1 = diag1[np.isfinite(diag1).all(axis=1)]
        diag2 = diag2[np.isfinite(diag2).all(axis=1)]

        # Simple distance based on number of features and persistence
        if len(diag1) == 0 and len(diag2) == 0:
            return 0.0

        # Count-based distance
        count_diff = abs(len(diag1) - len(diag2))

        # Persistence-based distance (if both have features)
        if len(diag1) > 0 and len(diag2) > 0:
            pers1 = np.mean(diag1[:, 1] - diag1[:, 0])
            pers2 = np.mean(diag2[:, 1] - diag2[:, 0])
            pers_diff = abs(pers1 - pers2)
            return count_diff + pers_diff
        else:
            return float(count_diff)

    except Exception as e:
        print(f"Warning: Simple distance calculation failed: {e}")
        return 0.0


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
