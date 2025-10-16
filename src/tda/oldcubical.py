"""""
TDA- Cubical_Complex Calculation
Author:Cheena Yadav
"""""



import numpy as np
import gudhi


def cubical_diagrams(img01: np.ndarray, superlevel: bool = False, coeff: int = 2,
                     min_persistence: float = None):
    """
    img01: 2D float array in [0,1]
    superlevel: True for bright features, False for dark features
    coeff: Homology coefficient field
    min_persistence: Minimum persistence threshold (if None, uses default 0.05)

    Returns dict with all intervals (including infinite) for H0 and H1
    :rtype: dict[str, Any]
    """
    # Invert the image if analyzing bright (superlevel) features
    vals = 1.0 - img01 if superlevel else img01

    flat = vals.astype(np.float64).ravel(order="C")
    cc = gudhi.CubicalComplex(dimensions=list(vals.shape), top_dimensional_cells=flat)

    # Use provided min_persistence or default
    if min_persistence is None:
        min_persistence = 0.05  # Default: 5% of [0,1] range

    # Use the persistence threshold
    cc.persistence(homology_coeff_field=coeff, min_persistence=min_persistence)

    # Get all intervals (including infinite)
    D0 = cc.persistence_intervals_in_dimension(0)
    D1 = cc.persistence_intervals_in_dimension(1)

    return {"H0": D0, "H1": D1}



