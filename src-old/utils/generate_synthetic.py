"""
Utility to generate simple synthetic images with known topology for verification.

This script creates a set of simple black and white images (e.g., circles, rings)
that have well-understood topological features (Betti numbers). These images are
crucial for unit testing and verifying the correctness of the persistence homology
computation pipeline.

Generated Images:
- Solid Circle: Betti-0 = 1, Betti-1 = 0
- Annulus (Ring): Betti-0 = 1, Betti-1 = 1
- Two Disjoint Circles: Betti-0 = 2, Betti-1 = 0
- Figure Eight: Betti-0 = 1, Betti-1 = 2
"""

import numpy as np
import cv2
from pathlib import Path

def generate_synthetic_images(output_dir: str):
    """
    Generates a suite of synthetic images and saves them to the specified directory.

    Args:
        output_dir: The directory where images will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_size = (200, 200)
    background_color = 0  # Black
    shape_color = 255      # White

    # 1. Solid Circle (B0=1, B1=0)
    img_circle = np.full(image_size, background_color, dtype=np.uint8)
    cv2.circle(img_circle, (100, 100), 50, shape_color, -1)
    cv2.imwrite(str(output_path / "synthetic_solid_circle.png"), img_circle)

    # 2. Annulus / Ring (B0=1, B1=1)
    img_annulus = np.full(image_size, background_color, dtype=np.uint8)
    cv2.circle(img_annulus, (100, 100), 70, shape_color, -1) # Outer circle
    cv2.circle(img_annulus, (100, 100), 30, background_color, -1) # Inner circle (hole)
    cv2.imwrite(str(output_path / "synthetic_annulus.png"), img_annulus)

    # 3. Two Disjoint Circles (B0=2, B1=0)
    img_two_circles = np.full(image_size, background_color, dtype=np.uint8)
    cv2.circle(img_two_circles, (60, 100), 30, shape_color, -1)
    cv2.circle(img_two_circles, (140, 100), 30, shape_color, -1)
    cv2.imwrite(str(output_path / "synthetic_two_circles.png"), img_two_circles)
    
    # 4. Figure Eight (B0=1, B1=2) - Corrected with thicker lines
    img_figure_eight = np.full(image_size, background_color, dtype=np.uint8)
    # Draw two overlapping circles to form a figure eight
    cv2.circle(img_figure_eight, (100, 75), 40, shape_color, -1)
    cv2.circle(img_figure_eight, (100, 125), 40, shape_color, -1)
    # Carve out the center to ensure two distinct loops
    cv2.rectangle(img_figure_eight, (60, 100-5), (140, 100+5), background_color, -1)
    # Re-draw the central connection point to ensure it's a single component
    cv2.circle(img_figure_eight, (100, 100), 5, shape_color, -1)
    cv2.imwrite(str(output_path / "synthetic_figure_eight.png"), img_figure_eight)

    print(f"Synthetic images generated in {output_path}")

if __name__ == "__main__":
    generate_synthetic_images("synthetic_data")

