"""
Step-by-Step Filtration Visualization (FINAL VERSION)

Shows ONLY the filtration building process - no duplicate PH diagrams or Betti curves.
Those are generated separately by plotter.py.

Focus: Visual clarity of how the cubical complex builds up step-by-step.

Author: Cheena Yadav
Date: October 2025
Version: 3.0.0 (FINAL)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2


class StepByStepVisualizer:
    """
    Final visualizer showing ONLY the filtration building process.
    Clean, focused, publication-ready.
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
            "wavelet": "Wavelet Denoising",
        }

        for method_key, method_name in denoising_methods.items():
            if method_key in filename_lower:
                if "gaussian" in filename_lower:
                    noise_type = "on Gaussian"
                elif "salt" in filename_lower or "pepper" in filename_lower:
                    noise_type = "on Salt-Pepper"
                else:
                    noise_type = ""
                return (f"Denoised: {method_name} {noise_type}".strip(), "#4CAF50")

        if "gaussian" in filename_lower:
            return ("Noisy: Gaussian", "#FF5722")
        elif "salt" in filename_lower or "pepper" in filename_lower:
            return ("Noisy: Salt-and-Pepper", "#FF5722")
        else:
            return ("Clean (Original)", "#2196F3")

    def _clean_title(self, filename):
        """Clean up filename for display."""
        # Remove file extension
        name = filename.replace('.png', '').replace('.jpg', '').replace('.tif', '')
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        # Capitalize properly
        return name.title()

    def _detect_filtration_type(self, filename):
        """
        Detect if this is EDT or intensity filtration from filename.
        
        Returns:
            bool: True if EDT, False if intensity
        """
        filename_lower = filename.lower()
        # Check for EDT indicators in filename
        return '_edt' in filename_lower or 'edt_' in filename_lower

    def create_step_by_step_visualization(self, image, params, persistence_dict,
                                          filename, output_dir):
        """
        Creates a focused 8-panel visualization showing ONLY the filtration building process.
        
        Layout:
        - Panel 1: Original image
        - Panel 2: Filtration values (heatmap)
        - Panels 3-8: Six frames showing filtration at t=0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        
        NO PH diagrams, NO Betti curves - those are separate files!
        """
        if self.logger:
            self.logger.info(f"Creating step-by-step visualization for {filename}")

        # Detect state and clean title
        state_name, state_color = self._detect_image_state(filename)
        clean_title = self._clean_title(filename)

        # Setup figure - white background for publication
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)

        # Title
        fig.suptitle(f"Filtration Building Process: {clean_title}\n{state_name}",
                     fontsize=18, weight='bold', y=0.96, color=state_color)

        # Normalize image
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)

        # Get parameters
        superlevel = params.get('superlevel', False)

        # Detect filtration type
        is_edt = self._detect_filtration_type(filename)

        # Determine filtration type and label
        if is_edt:
            # EDT filtration - always uses the normalized image directly
            filtration_values = img_norm
            direction = "EDT (Euclidean Distance Transform)"
        elif superlevel:
            # Intensity superlevel
            filtration_values = 1.0 - img_norm
            direction = "Superlevel (1 - intensity)"
        else:
            # Intensity sublevel
            filtration_values = img_norm
            direction = "Sublevel (intensity)"

        # Count final Betti numbers for reference
        h0_count = len(persistence_dict.get('H0', [])) if 'H0' in persistence_dict else 0
        h1_count = len(persistence_dict.get('H1', [])) if 'H1' in persistence_dict else 0

        # ====================================================================
        # PANEL 1: Original Image
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f"1. Original Image\n{state_name}", 
                     fontsize=13, weight='bold', color=state_color)
        ax1.axis('off')

        # Add stats box
        stats = (f"Size: {image.shape[0]}×{image.shape[1]}\n"
                f"Range: [{np.min(image):.2f}, {np.max(image):.2f}]\n"
                f"Mean: {np.mean(image):.2f}")
        ax1.text(0.02, 0.98, stats, transform=ax1.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor=state_color, linewidth=2))

        # ====================================================================
        # PANEL 2: Filtration Values
        # ====================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(filtration_values, cmap='viridis')
        ax2.set_title(f"2. Filtration Values\n{direction}", 
                     fontsize=13, weight='bold')
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Filtration Parameter', fontsize=10)

        # ====================================================================
        # PANELS 3-8: Filtration Building Sequence
        # ====================================================================
        levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        positions = [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        for idx, (level, pos) in enumerate(zip(levels, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            
            # Create binary mask at this threshold
            mask = filtration_values <= level
            
            # Count components
            num_components, labeled_image = cv2.connectedComponents(mask.astype(np.uint8))
            
            # Create visualization showing thresholded region
            # Show original image dimmed, with active region highlighted
            vis = np.stack([img_norm * 0.3] * 3, axis=-1)  # Dim background
            
            if np.any(mask):
                # Highlight active region in cyan
                vis[mask, 0] = 0.3 * img_norm[mask]
                vis[mask, 1] = 0.8
                vis[mask, 2] = 0.8
                
                # Draw component boundaries
                contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                # Draw contours on vis
                vis_uint8 = (vis * 255).astype(np.uint8)
                cv2.drawContours(vis_uint8, contours, -1, (255, 255, 0), 2)  # Yellow boundaries
                vis = vis_uint8.astype(np.float32) / 255.0
            
            ax.imshow(vis)
            ax.set_title(f"{idx + 3}. Threshold t = {level:.1f}\n"
                        f"Components: {num_components}",
                        fontsize=12, weight='bold')
            ax.axis('off')
            
            # Add threshold indicator
            indicator_color = 'cyan' if np.any(mask) else 'gray'
            ax.text(0.05, 0.95, f"t ≤ {level:.1f}", 
                   transform=ax.transAxes,
                   fontsize=11, va='top', ha='left', weight='bold',
                   bbox=dict(boxstyle='round', facecolor=indicator_color, 
                            alpha=0.8, edgecolor='white', linewidth=1.5))

        # ====================================================================
        # PANEL 9: Summary Info (bottom right)
        # ====================================================================
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = (
            f"FILTRATION SUMMARY\n\n"
            f"Image: {clean_title}\n"
            f"State: {state_name}\n\n"
            f"Filtration: {direction}\n"
            f"Threshold: {params.get('threshold', 0.5):.4f}\n\n"
            f"Final Topology:\n"
            f"  H₀ (Components): {h0_count}\n"
            f"  H₁ (Holes): {h1_count}\n\n"
            f"See separate files for:\n"
            f"  • Persistence diagram\n"
            f"  • Barcode plot\n"
            f"  • Betti evolution"
        )
        
        ax9.text(0.5, 0.5, summary_text, transform=ax9.transAxes,
                fontsize=11, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor=state_color, linewidth=3))

        # Save
        output_path = Path(output_dir) / f"{filename}_step_by_step.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        if self.logger:
            self.logger.info(f"Saved step-by-step visualization to {output_path}")

        return str(output_path)

