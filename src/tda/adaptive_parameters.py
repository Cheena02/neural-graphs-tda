from pathlib import Path

import numpy as np
from typing import Dict, Any, Optional
from src.utils.logger import TDALogger, log_method_call
from skimage.filters import threshold_otsu



class AdaptiveParameterSelector:
    def __init__(self, logger: Optional[TDALogger] = None):
        self.logger = logger or TDALogger(name="AdaptiveParameters")

    @log_method_call
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyzes basic image properties to guide parameter selection."""
        # If the image is 3D (like a color image), convert it to 2D grayscale.
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        # Ensure image is in 8-bit format for analysis
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        mean_intensity = float(np.mean(image_uint8))

        # This is the core logic: decide if we are looking for bright features or dark ones.
        if mean_intensity < 127:
            superlevel = True  # Assume bright features on a dark background
            reason = f"Image is generally dark (mean={mean_intensity:.1f}) → superlevel filtration for bright features."
        else:
            superlevel = False  # Assume dark features on a bright background
            reason = f"Image is generally bright (mean={mean_intensity:.1f}) → sublevel filtration for dark features."

        return {
            "superlevel": superlevel,
            "reasoning": [reason]
        }

    @log_method_call
    def select_optimal_parameters(self, image: np.ndarray, file_path: Optional[Path] = None,
                                  dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Selects optimal parameters based on image characteristics using Otsu's threshold.
        """
        analysis = self.analyze_image_characteristics(image)

        # --- START OF FIX ---
        # Ensure image is in 8-bit format for Otsu's method
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        # Calculate an automatic, intelligent threshold using Otsu's method.
        otsu_threshold = threshold_otsu(image_uint8)

        # Normalize the threshold to be in the [0, 1] range for the float image.
        calculated_threshold = otsu_threshold / 255.0
        # --- END OF FIX ---

        params = {
            'threshold': calculated_threshold,  # Use the automatically calculated threshold.
            'superlevel': analysis['superlevel'],
            'confidence': 0.9,
            'reasoning': analysis['reasoning']
        }

        params['reasoning'].append(f"Otsu's method selected an automatic threshold of {calculated_threshold:.4f}.")

        return params

    def validate_parameters(self, params: Dict, image: np.ndarray) -> Dict:
        return params
