"""
Step-by-Step Filtration Visualization (SIMPLIFIED VERSION)

Shows ONLY how the cubical complex builds up during filtration.
NO persistence diagrams, NO Betti curves - those are in separate files.

8 panels showing progressive filtration building.

Author: Cheena Yadav
Date: October 2025
Version: 4.0.0 (SIMPLIFIED)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2


class StepByStepVisualizer:
    """
    Simplified visualizer showing ONLY the filtration building process.
    8 panels: original image + filtration values + 6 building steps.
    """

    def __init__(self, logger=None):
        self.logger = logger

    def _detect_image_state(self, filename):
        """Detect the state of the image from filename."""
        filename_lower = filename.lower()

        # Check for denoising methods
        denoising_methods = {
            "median": "Median Filter",
            "bilateral": "Bilateral Filter",
            "nlm": "Non-Local Means",
            "non_local": "Non-Local Means",
            "topological": "Topological Denoising",
            "morphological": "Morphological Denoising",
        }

        for method_key, method_name in denoising_methods.items():
            if method_key in filename_lower:
                if "gaussian" in filename_lower:
                    noise_type = "on Gaussian"
                elif "salt" in filename_lower or "pepper" in filename_lower:
                    noise_type = "on Salt-Pepper"
                else:
                    noise_type = ""
                return (f"Denoised ({method_name} {noise_type})", "#4CAF50", "🧹")

        # Check for noise types
        if "gaussian" in filename_lower:
            return ("Noisy (Gaussian)", "#FF5722", "⚠️")
        elif "salt" in filename_lower or "pepper" in filename_lower:
            return ("Noisy (Salt-and-Pepper)", "#FF5722", "⚠️")
        else:
            return ("Clean (Original)", "#2196F3", "✨")

    def create_step_by_step_visualization(self, image, params, persistence_dict,
                                          filename, output_dir):
        """
        Creates 8-panel visualization showing ONLY filtration building.

        Args:
            image: Current image state
            params: Analysis parameters
            persistence_dict: Persistence diagrams (for feature counts only)
            filename: Base filename
            output_dir: Output directory
        """
        if self.logger:
            self.logger.info(f"Creating simplified step-by-step for {filename}")

        # Detect image state
        state_name, state_color, state_emoji = self._detect_image_state(filename)
        
        # Detect filtration type
        is_edt = '_edt' in filename.lower() or 'edt_' in filename.lower()

        # Setup figure
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25)

        fig.suptitle(f"Cubical Complex Filtration: {filename}\n"
                     f"{state_emoji} State: {state_name}",
                     fontsize=18, weight='bold', y=0.98,
                     color=state_color)

        # Normalize image
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)

        # Get parameters
        superlevel = params.get('superlevel', False)

        # Set filtration values and label
        if is_edt:
            filtration_values = img_norm
            direction = "EDT (Euclidean Distance Transform)"
        elif superlevel:
            filtration_values = 1.0 - img_norm
            direction = "Superlevel (1 - intensity)"
        else:
            filtration_values = img_norm
            direction = "Sublevel (intensity)"

        # Feature counts
        h0_count = len(persistence_dict.get('H0', []))
        h1_count = len(persistence_dict.get('H1', []))

        # ====================================================================
        # PANEL 1: Original Image
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f"1. Current Image\n{state_name}", 
                     fontsize=12, weight='bold', color=state_color)
        ax1.axis('off')
        
        # Add image stats
        stats_text = f"Size: {image.shape[1]}×{image.shape[0]}\n"
        stats_text += f"Range: [{np.min(image):.2f}, {np.max(image):.2f}]\n"
        stats_text += f"Mean: {np.mean(image):.2f}"
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                fontsize=9, color='cyan',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # ====================================================================
        # PANEL 2: Filtration Values
        # ====================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(filtration_values, cmap='viridis')
        ax2.set_title(f"2. Filtration Values\n{direction}", 
                     fontsize=12, weight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # ====================================================================
        # PANELS 3-8: Filtration Building Steps
        # ====================================================================
        thresholds = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
        
        for idx, t in enumerate(thresholds):
            row = (idx + 2) // 4
            col = (idx + 2) % 4
            ax = fig.add_subplot(gs[row, col])
            
            # Create binary mask at this threshold
            mask = (filtration_values >= t).astype(np.uint8)
            
            # Color the active regions
            colored = np.zeros((*mask.shape, 3))
            colored[mask == 1] = [0, 1, 0]  # Green for active
            
            # Overlay on grayscale
            display = np.stack([img_norm]*3, axis=-1) * 0.3
            display[mask == 1] = colored[mask == 1]
            
            ax.imshow(display)
            ax.set_title(f"{idx+3}. Threshold = {t:.2f}", 
                        fontsize=11, weight='bold')
            ax.axis('off')
            
            # Count components
            num_labels, labels = cv2.connectedComponents(mask)
            component_count = num_labels - 1  # Subtract background
            
            # Add stats
            active_pixels = np.sum(mask)
            total_pixels = mask.size
            percentage = (active_pixels / total_pixels) * 100
            
            stats = f"Components: {component_count}\n"
            stats += f"Active: {percentage:.1f}%"
            
            ax.text(0.02, 0.98, stats,
                   transform=ax.transAxes,
                   fontsize=9, color='lime',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Add overall info at bottom
        info_text = f"Total Features: H₀={h0_count}, H₁={h1_count} | "
        info_text += f"Filtration: {direction} | "
        info_text += f"Green regions = Active at threshold"
        
        fig.text(0.5, 0.02, info_text,
                ha='center', fontsize=11, color='white',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, pad=10))

        # Save
        output_path = Path(output_dir) / f"{filename}_step_by_step.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()

        if self.logger:
            self.logger.info(f"Saved step-by-step to {output_path}")

