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
        for var in variance_levels:
            row, col, *_ = image.shape
            mean = 0
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, 1 if image.ndim == 3 else 1))
            gauss = gauss.reshape(row, col, 1 if image.ndim == 3 else 1)
            noisy = image + gauss.squeeze()
            noisy = np.clip(noisy, 0, 255)
            noisy_images[f"gaussian_var_{var}"] = noisy.astype(np.uint8)
        return noisy_images

    def add_salt_pepper_noise(self, image: np.ndarray, amounts: List[float]) -> Dict[str, np.ndarray]:
        """
        Adds salt and pepper noise to an image at different amounts.

        Args:
            image: Input image (numpy array).
            amounts: List of noise amounts (proportions).

        Returns:
            A dictionary of noisy images, with keys indicating the noise amount.
        """
        noisy_images = {}
        for amount in amounts:
            row, col, *_ = image.shape
            s_vs_p = 0.5
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[tuple(coords)] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[tuple(coords)] = 0
            noisy_images[f"salt_pepper_amount_{amount}"] = out
        return noisy_images

