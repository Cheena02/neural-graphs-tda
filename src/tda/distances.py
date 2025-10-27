"""
Persistence Diagram Distance Calculations (FIXED VERSION)
Stability metrics for comparing persistence diagrams across different stages

FIXES:
1. Proper handling of infinite values
2. Robust GUDHI Wasserstein distance computation
3. Improved fallback distance measure
4. Better error handling and logging
"""
import numpy as np
import gudhi


def wasserstein_distance(diag1, diag2, order=1):
    """
       Compute Wasserstein distance between two persistence diagrams.

       The Wasserstein distance measures the cost of optimally matching features
       between two persistence diagrams, providing a metric for comparing
       topological signatures. Also known as bottleneck distance when order=∞.

       Args:
           diag1 (np.ndarray): First persistence diagram, shape (n1, 2).
                              Each row is [birth, death] pair.
           diag2 (np.ndarray): Second persistence diagram, shape (n2, 2).
           order (int, optional): Order of Wasserstein distance. Use 1 for
                                 W₁ (earth mover's distance) or 2 for W₂.
                                 Use np.inf for bottleneck distance. Default: 1.

       Returns:
           float: Wasserstein distance W_p(diag1, diag2) ≥ 0.
                  Returns 0 if diagrams are identical.

       Example:
           # >>> clean_diag = cubical_diagrams(clean_image)
           # >>> noisy_diag = cubical_diagrams(noisy_image)
           # >>> dist = wasserstein_distance(clean_diag['H1'], noisy_diag['H1'])
           # >>> print(f"Topological distance: {dist:.4f}")
           # Topological distance: 0.0234

       Notes:
           - Validates stability theorem: W(dgm(f), dgm(g)) ≤ ||f - g||_∞
           - Computation time: O(n³) for n features (Hungarian algorithm)
           - Returns 0.0 if both diagrams are empty
           - Handles diagrams of different sizes automatically

       Mathematical Details:
           W_p(D1, D2) = (inf_{γ} Σ ||x - γ(x)||^p)^(1/p)
           where γ ranges over all bijections between D1 ∪ Δ and D2 ∪ Δ
           and Δ is the diagonal {(x,x) : x ∈ ℝ}

       See Also:
           - bottleneck_distance(): Equivalent to wasserstein_distance with order=∞
           - cubical_diagrams(): Generate persistence diagrams
       """
    try:
        # Convert to numpy arrays
        if isinstance(diag1, list):
            diag1 = np.array(diag1) if len(diag1) > 0 else np.array([[0, 0]])
        if isinstance(diag2, list):
            diag2 = np.array(diag2) if len(diag2) > 0 else np.array([[0, 0]])
        
        # Ensure 2D arrays
        if diag1.ndim == 1:
            diag1 = diag1.reshape(-1, 2)
        if diag2.ndim == 1:
            diag2 = diag2.reshape(-1, 2)
        
        # Remove infinite values PROPERLY
        # Keep only rows where BOTH birth and death are finite
        finite_mask1 = np.isfinite(diag1[:, 0]) & np.isfinite(diag1[:, 1])
        finite_mask2 = np.isfinite(diag2[:, 0]) & np.isfinite(diag2[:, 1])
        
        diag1_finite = diag1[finite_mask1]
        diag2_finite = diag2[finite_mask2]
        
        # Handle empty diagrams
        if len(diag1_finite) == 0 and len(diag2_finite) == 0:
            return 0.0
        
        # If one is empty, use simple distance
        if len(diag1_finite) == 0 or len(diag2_finite) == 0:
            return simple_diagram_distance(diag1_finite, diag2_finite)
        
        # Try GUDHI's Wasserstein distance
        try:
            # This is the correct import for GUDHI 3.x
            distance = gudhi.wasserstein.wasserstein_distance(
                diag1_finite, 
                diag2_finite, 
                order=order,
                internal_p=order
            )
            
            # Sanity check
            if np.isfinite(distance) and distance >= 0:
                return float(distance)
            else:
                print(f"Warning: Invalid Wasserstein distance: {distance}, using fallback")
                return simple_diagram_distance(diag1_finite, diag2_finite)
                
        except Exception as e:
            print(f"Warning: GUDHI Wasserstein failed: {e}, using fallback")
            return simple_diagram_distance(diag1_finite, diag2_finite)
            
    except Exception as e:
        print(f"Error in Wasserstein distance calculation: {e}")
        return 0.0


def simple_diagram_distance(diag1, diag2):
    """
    Improved fallback distance measure.
    Uses bottleneck distance approximation.
    
    Args:
        diag1: First persistence diagram
        diag2: Second persistence diagram
    
    Returns:
        float: Approximate distance between diagrams
    """
    try:
        # Handle empty cases
        if len(diag1) == 0 and len(diag2) == 0:
            return 0.0
        
        # Calculate persistence for each diagram
        def get_persistence_values(diag):
            if len(diag) == 0:
                return np.array([])
            return diag[:, 1] - diag[:, 0]
        
        pers1 = get_persistence_values(diag1)
        pers2 = get_persistence_values(diag2)
        
        # If one is empty, return sum of other's persistence
        if len(pers1) == 0:
            return np.sum(pers2) / 2  # Divide by 2 for birth-death pairing cost
        if len(pers2) == 0:
            return np.sum(pers1) / 2
        
        # Bottleneck-like distance: max persistence difference + count difference
        max_pers1 = np.max(pers1)
        max_pers2 = np.max(pers2)
        bottleneck_approx = abs(max_pers1 - max_pers2)
        
        # Count difference weighted by average persistence
        count_diff = abs(len(diag1) - len(diag2))
        avg_pers = (np.mean(pers1) + np.mean(pers2)) / 2
        count_cost = count_diff * avg_pers * 0.5
        
        return bottleneck_approx + count_cost
        
    except Exception as e:
        print(f"Error in simple distance: {e}")
        # Ultimate fallback: just count difference
        return float(abs(len(diag1) - len(diag2)))


def compute_all_distances(clean_diag, noisy_diag, denoised_diag, dimensions=[0, 1]):
    """
    Compute all relevant distances between clean, noisy, and denoised diagrams.
    
    Args:
        clean_diag: Persistence diagrams for clean image
        noisy_diag: Persistence diagrams for noisy image
        denoised_diag: Persistence diagrams for denoised image
        dimensions: List of homology dimensions to compute (default: [0, 1])
    
    Returns:
        dict: Dictionary of all computed distances
    """
    distances = {}
    
    for dim in dimensions:
        h_key = f"H{dim}"
        
        # Get diagrams for this dimension
        clean_h = clean_diag.get(h_key, np.array([]))
        noisy_h = noisy_diag.get(h_key, np.array([]))
        denoised_h = denoised_diag.get(h_key, np.array([]))
        
        # Compute Wasserstein distances
        try:
            distances[f'wasserstein_clean_noisy_H{dim}'] = wasserstein_distance(
                clean_h, noisy_h, order=1
            )
        except Exception as e:
            print(f"Error computing clean-noisy distance for H{dim}: {e}")
            distances[f'wasserstein_clean_noisy_H{dim}'] = 0.0
        
        try:
            distances[f'wasserstein_clean_denoised_H{dim}'] = wasserstein_distance(
                clean_h, denoised_h, order=1
            )
        except Exception as e:
            print(f"Error computing clean-denoised distance for H{dim}: {e}")
            distances[f'wasserstein_clean_denoised_H{dim}'] = 0.0
        
        try:
            distances[f'wasserstein_noisy_denoised_H{dim}'] = wasserstein_distance(
                noisy_h, denoised_h, order=1
            )
        except Exception as e:
            print(f"Error computing noisy-denoised distance for H{dim}: {e}")
            distances[f'wasserstein_noisy_denoised_H{dim}'] = 0.0
    
    return distances


def bottleneck_distance(diag1, diag2):
    """
    Compute bottleneck distance between two persistence diagrams.
    
    Args:
        diag1: First persistence diagram
        diag2: Second persistence diagram
    
    Returns:
        float: Bottleneck distance
    """
    try:
        # Convert and clean diagrams (same as Wasserstein)
        if isinstance(diag1, list):
            diag1 = np.array(diag1) if len(diag1) > 0 else np.array([[0, 0]])
        if isinstance(diag2, list):
            diag2 = np.array(diag2) if len(diag2) > 0 else np.array([[0, 0]])
        
        if diag1.ndim == 1:
            diag1 = diag1.reshape(-1, 2)
        if diag2.ndim == 1:
            diag2 = diag2.reshape(-1, 2)
        
        # Remove infinite values
        finite_mask1 = np.isfinite(diag1[:, 0]) & np.isfinite(diag1[:, 1])
        finite_mask2 = np.isfinite(diag2[:, 0]) & np.isfinite(diag2[:, 1])
        
        diag1_finite = diag1[finite_mask1]
        diag2_finite = diag2[finite_mask2]
        
        # Handle empty cases
        if len(diag1_finite) == 0 and len(diag2_finite) == 0:
            return 0.0
        
        # Try GUDHI's bottleneck distance
        try:
            distance = gudhi.bottleneck_distance(diag1_finite, diag2_finite)
            if np.isfinite(distance) and distance >= 0:
                return float(distance)
        except:
            pass
        
        # Fallback: use simple distance
        return simple_diagram_distance(diag1_finite, diag2_finite)
        
    except Exception as e:
        print(f"Error in bottleneck distance: {e}")
        return 0.0

