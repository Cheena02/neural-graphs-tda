#!/usr/bin/env python3
"""
Comprehensive noise mitigation strategies for TDA pipeline.

This module provides a collection of advanced denoising techniques, including
standard image processing filters and topology-aware methods based on
persistence homology. Each strategy is designed to be configurable,
reproducible, and thoroughly logged.

Inspired by research literature:
- "Cubical Persistent Homology-Based Technique for Image Denoising"
- "Topological DeNoising: Strengthening the Topological Signal"
- "Wavelet-based Topological Loss for Low-Light Image Denoising"

Engineering Features:
- Modular and extensible design for adding new strategies
- Parameterized functions for systematic evaluation
- Integration with the TDA logging system for reproducibility
- Performance monitoring for each denoising algorithm
- Error handling and validation for robust execution
"""

import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, denoise_wavelet
from skimage.morphology import closing, opening, disk
import gudhi
from typing import Dict, Any, Optional

from src.utils.logger import TDALogger, log_method_call

class DenoisingStrategies:
    """
    Encapsulates various image denoising strategies for TDA stability analysis.
    
    This class provides a unified interface for applying different denoising
    algorithms, from classic filters to advanced topological methods.
    """
    
    def __init__(self, logger: Optional[TDALogger] = None):
        """
        Initialize the denoising strategies handler.
        
        Args:
            logger: TDA logger instance for tracking.
        """
        self.logger = logger or TDALogger(name="Denoising")
        self.logger.info("🔧 Denoising Strategies initialized")

    @log_method_call
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to denoise an image.
        
        Args:
            image: Input noisy image (numpy array).
            kernel_size: Size of the Gaussian kernel (must be odd).
            
        Returns:
            Denoised image.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
            self.logger.warning(f"Adjusted Gaussian kernel size to {kernel_size}")
            
        self.logger.info(f"Applying Gaussian Blur with kernel size: {kernel_size}")
        denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return denoised_image

    @log_method_call
    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Median filter, effective for salt-and-pepper noise.
        
        Args:
            image: Input noisy image.
            kernel_size: Size of the median filter kernel (must be odd).
            
        Returns:
            Denoised image.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.logger.warning(f"Adjusted Median kernel size to {kernel_size}")

        self.logger.info(f"Applying Median Filter with kernel size: {kernel_size}")
        denoised_image = cv2.medianBlur(image, kernel_size)
        return denoised_image

    @log_method_call
    def apply_bilateral_filter(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply Bilateral filter, which preserves edges while denoising.
        
        Args:
            image: Input noisy image.
            d: Diameter of each pixel neighborhood.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.
            
        Returns:
            Denoised image.
        """
        self.logger.info(f"Applying Bilateral Filter with d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised_image

    @log_method_call
    def apply_non_local_means(self, image: np.ndarray, h: float = 10, patch_size: int = 7, patch_distance: int = 11) -> np.ndarray:
        """
        Apply Non-Local Means denoising, effective for Gaussian noise.
        
        Args:
            image: Input noisy image (uint8).
            h: Strength of the filter.
            patch_size: Size of patches used for comparison.
            patch_distance: Max distance to search for patches.

        Returns:
            Denoised image.
        """
        self.logger.info(f"Applying Non-Local Means with h={h}, patch_size={patch_size}")
        # scikit-image expects float images in [0, 1]
        img_float = image.astype(np.float32) / 255.0
        denoised_float = denoise_nl_means(
            img_float, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True
        )
        denoised_image = (denoised_float * 255).astype(np.uint8)
        return denoised_image

    @log_method_call
    def apply_wavelet_denoising(self, image: np.ndarray, wavelet: str = 'db1', level: int = 1) -> np.ndarray:
        """
        Apply Wavelet denoising.

        Args:
            image: Input noisy image.
            wavelet: Type of wavelet to use.
            level: Decomposition level.

        Returns:
            Denoised image.
        """
        self.logger.info(f"Applying Wavelet Denoising with wavelet={wavelet}, level={level}")
        img_float = image.astype(np.float32) / 255.0
        denoised_float = denoise_wavelet(img_float, wavelet=wavelet, mode='soft', wavelet_levels=level)
        denoised_image = (denoised_float * 255).astype(np.uint8)
        return denoised_image

    @log_method_call
    def apply_morphological_denoising(self, image: np.ndarray, kernel_size: int = 3, operation: str = 'opening') -> np.ndarray:
        """
        Apply morphological operations for denoising.
        'opening' removes small bright spots (salt noise).
        'closing' removes small dark spots (pepper noise).

        Args:
            image: Input noisy image.
            kernel_size: Size of the morphological kernel.
            operation: 'opening' or 'closing'.

        Returns:
            Denoised image.
        """
        self.logger.info(f"Applying Morphological Denoising: {operation} with kernel size {kernel_size}")
        kernel = disk(kernel_size)
        if operation == 'opening':
            denoised_image = opening(image, kernel)
        elif operation == 'closing':
            denoised_image = closing(image, kernel)
        else:
            self.logger.warning(f"Unsupported morphological operation: {operation}. Returning original image.")
            return image
        return denoised_image

    @log_method_call
    def apply_topological_denoising(self, image: np.ndarray, persistence_threshold: float = 10.0, superlevel: bool = False) -> np.ndarray:
        """
        Apply a simplified topological denoising based on persistence.
        This method uses persistence information to guide adaptive filtering.

        Args:
            image: Input noisy image (grayscale).
            persistence_threshold: Persistence value below which features are considered noise.
            superlevel: If True, use superlevel sets filtration. Otherwise, sublevel sets.

        Returns:
            Topologically denoised image.
        """
        self.logger.info(f"Applying Topological Denoising with threshold: {persistence_threshold}")
        if image.ndim > 2:
            self.logger.warning("Topological denoising requires grayscale image. Converting...")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert image for superlevel filtration to match GUDHI's expectation
        vals = (255 - image) if superlevel else image
        
        # Create cubical complex
        cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=vals.flatten())
        
        # Compute persistence
        cubical_complex.persistence()
        
        # Use persistence intervals for simpler implementation
        h0_intervals = cubical_complex.persistence_intervals_in_dimension(0)
        h1_intervals = cubical_complex.persistence_intervals_in_dimension(1)
        
        # Calculate persistence statistics
        all_intervals = np.vstack([h0_intervals, h1_intervals]) if len(h1_intervals) > 0 else h0_intervals
        finite_intervals = all_intervals[np.isfinite(all_intervals[:, 1])]
        
        if len(finite_intervals) == 0:
            self.logger.warning("No finite persistence intervals found. Returning original image.")
            return image
        
        persistences = finite_intervals[:, 1] - finite_intervals[:, 0]
        median_pers = np.median(persistences)
        
        # Use morphological filtering based on persistence threshold
        from scipy.ndimage import median_filter, gaussian_filter
        
        # Adaptive filtering based on persistence threshold
        kernel_size = max(3, min(9, int(persistence_threshold / median_pers * 5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Reshape vals back to image shape
        vals_2d = vals.reshape(image.shape)
        
        denoised_image = median_filter(vals_2d.astype(np.float64), size=kernel_size)
        denoised_image = gaussian_filter(denoised_image, sigma=1.0)
        
        self.logger.info(f"Applied topological-guided filtering with kernel size {kernel_size}")
        self.logger.info(f"Median persistence: {median_pers:.2f}, Threshold: {persistence_threshold}")

        # Revert inversion if superlevel was used
        if superlevel:
            denoised_image = 255 - denoised_image

        return denoised_image.astype(np.uint8)

    def get_all_strategies(self) -> Dict[str, callable]:
        """Returns a dictionary of all available denoising strategies."""
        return {
            "gaussian_blur": self.apply_gaussian_blur,
            "median_filter": self.apply_median_filter,
            "bilateral_filter": self.apply_bilateral_filter,
            "non_local_means": self.apply_non_local_means,
            "wavelet_denoising": self.apply_wavelet_denoising,
            "morphological_opening": lambda img: self.apply_morphological_denoising(img, operation='opening'),
            "morphological_closing": lambda img: self.apply_morphological_denoising(img, operation='closing'),
            "topological_denoising": self.apply_topological_denoising,
        }

