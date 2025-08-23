import numpy as np
import gudhi

def cubical_diagrams(img01: np.ndarray, superlevel: bool = False, coeff: int = 2):
    """
    img01: 2D float array in [0,1]
    superlevel=False -> sublevel (intensity lower-star)
    superlevel=True  -> superlevel (1 - intensity)
    returns dict with finite intervals for H0 and H1
    """
    vals = 1.0 - img01 if superlevel else img01
    flat = vals.astype(np.float64).ravel(order="C")
    cc = gudhi.CubicalComplex(dimensions=list(vals.shape), top_dimensional_cells=flat)
    cc.persistence(homology_coeff_field=coeff, min_persistence=0.0)
    D0 = cc.persistence_intervals_in_dimension(0)
    D1 = cc.persistence_intervals_in_dimension(1)
    if D0.size: D0 = D0[np.isfinite(D0[:,1])]
    if D1.size: D1 = D1[np.isfinite(D1[:,1])]
    return {"H0": D0, "H1": D1}
