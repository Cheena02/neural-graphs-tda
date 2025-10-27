"""Noise generation module for TDA pipeline.

This module provides functions for adding various types of noise to images to test the robustness of persistence homology.

Engineering Features:
- Parameterized noise functions for systematic evaluation.
- Reproducible noise generation with controlled randomness.
- Integration with the TDA logging system.
"""

import numpy as np
import cv2
from typing import List, Dict

class NoiseGenerator:
    """Encapsulates different noise generation methods."""

    def __init__(self, seed: int | None = 42):
        # Use a per-instance RNG for reproducibility; None keeps it random.
        self.rng = np.random.default_rng(seed)

    def add_gaussian_noise(self, image: np.ndarray, variance_levels: List[float]) -> Dict[str, np.ndarray]:
        """
        Adds Gaussian noise to an image at different variance levels.

        Args:
            image: Input image (numpy array).
            variance_levels: List of variances for the Gaussian noise.

        Returns:
            A dictionary of noisy images, with keys indicating the noise level.
        """
        noisy_images = {}
        src_dtype = image.dtype
        is_color = (image.ndim == 3)
        # Work in float32 in [0,255] if image is uint8, else in its native scale
        img = image.astype(np.float32, copy=False)

        h, w = image.shape[:2]
        for var in variance_levels:
            sigma = float(var) ** 0.5
            # Generate 2D noise and broadcast to channels if needed
            n2d = self.rng.normal(0.0, sigma, size=(h, w)).astype(np.float32)
            if is_color:
                n = np.repeat(n2d[:, :, None], image.shape[2], axis=2)
            else:
                n = n2d
            noisy = img + n
            # Clip to valid range if looks like 8-bit imagery
            if src_dtype == np.uint8:
                noisy = np.clip(noisy, 0, 255)
            noisy_images[f"gaussian_var_{var}"] = noisy.astype(src_dtype, copy=False)
        return noisy_images

    def add_salt_pepper_noise(self, image: np.ndarray, amounts: List[float]) -> Dict[str, np.ndarray]:
        """
        Adds salt and pepper noise at multiple amounts. Preserves input dtype/range.
        """
        noisy_images = {}
        src_dtype = image.dtype
        is_color = (image.ndim == 3)
        h, w = image.shape[:2]
        total_pixels = h * w
        s_vs_p = 0.5

        for amount in amounts:
            out = image.copy()
            n_salt = int(np.ceil(amount * total_pixels * s_vs_p))
            n_pepp = int(np.ceil(amount * total_pixels * (1.0 - s_vs_p)))

            # Random unique coordinates in [0,h), [0,w)
            rs = self.rng.integers(0, h, size=n_salt)
            cs = self.rng.integers(0, w, size=n_salt)
            rp = self.rng.integers(0, h, size=n_pepp)
            cp = self.rng.integers(0, w, size=n_pepp)

            if is_color:
                out[rs, cs, :] = 255 if src_dtype == np.uint8 else out.max()
                out[rp, cp, :] = 0
            else:
                out[rs, cs] = 255 if src_dtype == np.uint8 else out.max()
                out[rp, cp] = 0

            noisy_images[f"salt_pepper_amount_{amount}"] = out
        return noisy_images


