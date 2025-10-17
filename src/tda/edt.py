"""
Euclidean Distance Transform (EDT) Filtration for Persistence Homology
Alternative to intensity-based cubical filtration, especially useful for blob analysis
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_opening, binary_closing
from skimage.filters import threshold_otsu
from .cubical import cubical_diagrams
import gudhi


def edt_diagrams(img01: np.ndarray, bin_thresh: float = None, invert: bool = False, coeff: int = 2):
    """
    Compute persistence diagrams using Euclidean Distance Transform filtration.

    Args:
        img01: Input image as numpy array with values in [0,1]
        bin_thresh: Binarization threshold. If None, uses Otsu's method
        invert: If True, inverts image before processing
        coeff: Coefficient field for homology computation

    Returns:
        dict: Dictionary with 'H0' and 'H1' keys containing persistence intervals
    """
    try:
        x = img01.copy()
        if invert:
            x = 1.0 - x

        # Auto-threshold if not provided
        if bin_thresh is None:
            bin_thresh = threshold_otsu(x)

        # Binarize image
        mask = (x >= bin_thresh).astype(np.uint8)

         # Clean up binary mask to remove noise
        mask = binary_closing(mask, structure=np.ones((3, 3)))
        mask = binary_opening(mask, structure=np.ones((3, 3)))

        # Compute Euclidean distance transform
        dist = distance_transform_edt(mask)
        vals = dist.astype(np.float64)
        # Calculate adaptive minimum persistence (10% of max distance)
        max_dist = vals.max()
        min_pers = max_dist * 0.10

        # Create cubical complex from distance field
        cc = gudhi.CubicalComplex(
            dimensions=list(vals.shape),
            top_dimensional_cells=vals.ravel(order="C")
        )

        # FIXED: Compute persistence with minimum threshold instead of 0.0
        cc.persistence(homology_coeff_field=coeff, min_persistence=min_pers)

        return {
            "H0": cc.persistence_intervals_in_dimension(0),
            "H1": cc.persistence_intervals_in_dimension(1)
        }

    except Exception as e:
        print(f"Warning: EDT filtration failed: {e}")
        # FIXED: Return numpy arrays for consistency
        return {"H0": np.array([]), "H1": np.array([])}


def compare_filtrations(img01: np.ndarray, intensity_params: dict = None, edt_params: dict = None):
    """Compare intensity-based vs EDT-based filtrations on the same image."""

    # Default parameters
    if intensity_params is None:
        intensity_params = {'superlevel': True}
    if edt_params is None:
        edt_params = {'bin_thresh': None, 'invert': False}

    # Compute both filtrations
    intensity_diag = cubical_diagrams(img01, **intensity_params)
    if 'coeff' not in edt_params:
        edt_params['coeff'] = intensity_params.get('coeff', 2)
    edt_diag = edt_diagrams(img01, **edt_params)

    # Compute basic metrics
    comparison = {
        'intensity_betti_0': len(intensity_diag.get('H0', [])),
        'intensity_betti_1': len(intensity_diag.get('H1', [])),
        'edt_betti_0': len(edt_diag.get('H0', [])),
        'edt_betti_1': len(edt_diag.get('H1', [])),
        'intensity_diagrams': intensity_diag,
        'edt_diagrams': edt_diag
    }

    return comparison

