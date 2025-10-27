"""""
TDA- Cubical_Complex Calculation
Author:Cheena Yadav
"""""



import numpy as np
import gudhi
from .thresholds import auto_min_persistence


def cubical_diagrams(img01: np.ndarray, superlevel: bool = False, coeff: int = 2):
    """
    img01: 2D float array in [0,1]
    superlevel=False -> sublevel (intensity lower-star)
    superlevel=True  -> superlevel (1 - intensity)
    returns dict with all intervals (including infinite) for H0 and H1
    """
    # Invert the image if analyzing bright (superlevel) features
    vals = 1.0 - img01 if superlevel else img01

    flat = vals.astype(np.float64).ravel(order="C")
    cc = gudhi.CubicalComplex(dimensions=list(vals.shape), top_dimensional_cells=flat)

    # FIXED: Use auto_min_persistence for noise-adaptive threshold
    # This estimates noise level using MAD and sets threshold at 3σ
    # Filters 99.7% of noise-induced features while preserving real structure
    min_pers = auto_min_persistence(img01)
    
    # Compute persistence with adaptive minimum threshold
    cc.persistence(homology_coeff_field=coeff, min_persistence=min_pers)

    # Important: We are NOT filtering for finite intervals anymore.
    # The infinite interval represents the main connected component.
    D0 = cc.persistence_intervals_in_dimension(0)
    D1 = cc.persistence_intervals_in_dimension(1)

    return {"H0": D0, "H1": D1}



