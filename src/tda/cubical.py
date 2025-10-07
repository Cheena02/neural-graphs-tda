import numpy as np
import gudhi

def cubical_diagrams(img01: np.ndarray, superlevel: bool = False, coeff: int = 2):
    """
    img01: 2D float array in [0,1]
    superlevel=False -> sublevel (intensity lower-star)
    superlevel=True  -> superlevel (1 - intensity)
    returns dict with all intervals (including infinite) for H0 and H1
    """
    # The logic to invert the image for analyzing bright features
    # is now handled in the main pipeline before calling this function.
    vals = img01
    
    flat = vals.astype(np.float64).ravel(order="C")
    cc = gudhi.CubicalComplex(dimensions=list(vals.shape), top_dimensional_cells=flat)
    cc.persistence(homology_coeff_field=coeff, min_persistence=0.0)
    
    # Important: We are NOT filtering for finite intervals anymore.
    # The infinite interval represents the main connected component.
    D0 = cc.persistence_intervals_in_dimension(0)
    D1 = cc.persistence_intervals_in_dimension(1)
    
    return {"H0": D0, "H1": D1}
