"""
Euclidean Distance Transform (EDT) Filtration for Persistence Homology (FIXED VERSION)
Alternative to intensity-based cubical filtration, especially useful for blob analysis

FIXES:
1. Uses consistent 2% threshold for fair comparison with intensity filtration
2. Added optional CSV saving to compare_filtrations function
3. Better error handling and logging
"""

import numpy as np
import os
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
        
    FIXED: Uses 2% threshold (not 10%) for fair comparison with intensity filtration
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
        
        # FIXED: Use 2% threshold (not 10%) for fair comparison
        # This makes it comparable to intensity filtration's 2% of [0,1] range
        max_dist = vals.max()
        min_pers = max_dist * 0.02  # Changed from 0.10 to 0.02

        # Create cubical complex from distance field
        cc = gudhi.CubicalComplex(
            dimensions=list(vals.shape),
            top_dimensional_cells=vals.ravel(order="C")
        )

        # Compute persistence with minimum threshold
        cc.persistence(homology_coeff_field=coeff, min_persistence=min_pers)

        return {
            "H0": cc.persistence_intervals_in_dimension(0),
            "H1": cc.persistence_intervals_in_dimension(1)
        }

    except Exception as e:
        print(f"Warning: EDT filtration failed: {e}")
        # Return numpy arrays for consistency
        return {"H0": np.array([]), "H1": np.array([])}


def compare_filtrations(img01: np.ndarray, intensity_params: dict = None, 
                       edt_params: dict = None, save_path: str = None,
                       image_name: str = None):
    """
    Compare intensity-based vs EDT-based filtrations on the same image.
    
    FIXED: Now supports optional CSV saving for immediate result persistence.
    
    Args:
        img01: Input image as numpy array
        intensity_params: Parameters for intensity filtration
        edt_params: Parameters for EDT filtration
        save_path: Optional path to save comparison results to CSV
        image_name: Optional image name for CSV record
    
    Returns:
        dict: Comparison results including Betti numbers and diagrams
    """
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
    
    # NEW: Save to CSV if path provided
    if save_path:
        try:
            import pandas as pd
            
            # Create data row
            data_row = {
                'image_name': image_name if image_name else 'unknown',
                'intensity_b0': comparison['intensity_betti_0'],
                'intensity_b1': comparison['intensity_betti_1'],
                'edt_b0': comparison['edt_betti_0'],
                'edt_b1': comparison['edt_betti_1'],
            }
            
            # Check if file exists to determine if we need header
            file_exists = os.path.exists(save_path)
            
            # Create DataFrame and append to CSV
            df = pd.DataFrame([data_row])
            df.to_csv(save_path, mode='a', header=not file_exists, index=False)
            
            print(f"Comparison results saved to: {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not save comparison to CSV: {e}")
    
    return comparison

