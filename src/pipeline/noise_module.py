# src/augment/noise.py
from __future__ import annotations
import numpy as np
import cv2

def _rng(seed: int):
    return np.random.default_rng(seed)

def apply_noise(img: np.ndarray, kind: str, level: float, seed: int) -> np.ndarray:
    """
    Apply noise to a grayscale image/volume already in [0,1] float32.

    kind:
      - 'gauss'    : additive N(0, level)
      - 'sp'       : salt & pepper with prob=level
      - 'speckle'  : multiplicative (1 + N(0, level))
      - 'poisson'  : Poisson with peak counts = level (int; higher = cleaner)
      - 'blur'     : Gaussian blur with sigma = level
      - 'none'     : no-op

    Works for 2D (H,W) and 3D (Z,Y,X). Returns clipped float32 in [0,1].
    """
    img = img.astype(np.float32, copy=False)
    r = _rng(seed)
    k = (kind or "none").lower()

    if k == "gauss":
        noisy = img + r.normal(0.0, float(level), size=img.shape).astype(np.float32)

    elif k == "sp":
        p = float(level)
        noisy = img.copy()
        m = r.random(img.shape)
        noisy[m < p/2] = 0.0
        noisy[(m >= p/2) & (m < p)] = 1.0

    elif k == "speckle":
        noisy = img * (1.0 + r.normal(0.0, float(level), size=img.shape).astype(np.float32))

    elif k == "poisson":
        L = max(1, int(round(level)))
        noisy = r.poisson(img * L).astype(np.float32) / float(L)

    elif k == "blur":
        sigma = float(level)
        if sigma <= 0:
            return img
        if img.ndim == 2:
            noisy = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:  # (Z,Y,X)
            noisy = np.empty_like(img)
            for z in range(img.shape[0]):
                noisy[z] = cv2.GaussianBlur(img[z], (0, 0), sigmaX=sigma, sigmaY=sigma)

    else:  # 'none' or unknown
        return img

    return np.clip(noisy, 0.0, 1.0)


def parse_levels(spec: str) -> list[float]:
    """
    '0.01,0.03,0.05'  -> [0.01, 0.03, 0.05]
    '0:0.05:0.01'     -> [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    """
    spec = spec.strip()
    if ":" in spec:
        a, b, s = map(float, spec.split(":"))
        vals, x = [], a
        while x <= b + 1e-12:
            vals.append(round(x, 10))
            x += s
        return vals
    return [float(x) for x in spec.split(",") if x.strip()]
