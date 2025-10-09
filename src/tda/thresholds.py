
import numpy as np, cv2

def estimate_noise_sigma01(img01: np.ndarray) -> float:
    """
    Estimate noise on a [0,1] image using high-pass residual MAD.
    Returns an approximate sigma of zero-mean Gaussian noise.
    """
    if img01.dtype not in (np.float32, np.float64):
        m = float(img01.max()) if img01.size else 1.0
        img01 = img01.astype(np.float32) / max(1.0, m)
    smooth = cv2.GaussianBlur(img01, (0, 0), sigmaX=1.0)
    resid  = img01 - smooth
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(1.4826 * mad)  # sigma ≈ 1.4826 * MAD

def auto_min_persistence(img01: np.ndarray, alpha: float = 3.0,
                         floor: float = 0.02, cap: float = 0.12) -> float:
    """
    Choose min_persistence = clip(alpha * sigma_noise, floor, cap).
    alpha≈3 keeps ~99.7% of Gaussian noise below the cut.
    """
    sigma = estimate_noise_sigma01(img01)
    mp = max(floor, min(cap, alpha * sigma))
    return float(mp)
