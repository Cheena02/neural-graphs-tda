#!/usr/bin/env python3
"""
Step-by-step visualization module for TDA pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class StepByStepVisualizer:
    def __init__(self, logger=None):
        self.logger = logger

    def create_step_by_step_visualization(self, image, params, filename, output_dir):
        """Generates a comprehensive 6-panel step-by-step visualization of the TDA process."""
        if self.logger:
            self.logger.info(f"Creating step-by-step visualization for {filename}")

        # Use a dark theme for better contrast
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Step-by-Step TDA Analysis: {filename}", fontsize=18, weight='bold')
        axes = axes.ravel()

        # Panel 1: Original Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("1. Original Image", weight='bold')
        axes[0].axis('off')

        # Panel 2: Normalized Image
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        axes[1].imshow(normalized_image, cmap='gray')
        axes[1].set_title("2. Normalized Image (0-1)", weight='bold')
        axes[1].axis('off')

        # Panel 3: Thresholded Binary Image
        threshold = params.get('threshold', 0.5)
        if params.get('superlevel', False):
            binary_image = normalized_image >= threshold
            axes[2].set_title(f"3. Superlevel Set (>{threshold:.3f})", weight='bold')
        else:
            binary_image = normalized_image <= threshold
            axes[2].set_title(f"3. Sublevel Set (<{threshold:.3f})", weight='bold')

        axes[2].imshow(binary_image, cmap='gray', vmin=0, vmax=1)
        axes[2].axis('off')

        # Panel 4: Analysis Parameters (Text)
        stats_text = (
            f"Image Statistics:\n"
            f"  Shape: {image.shape}\n"
            f"  Data Type: {image.dtype}\n"
            f"  Min Value: {np.min(image):.3f}\n"
            f"  Max Value: {np.max(image):.3f}\n"
            f"  Mean: {np.mean(image):.3f}\n"
            f"  Std Dev: {np.std(image):.3f}\n\n"
            f"Selected Parameters:\n"
            f"  Threshold: {params.get('threshold', 0):.6f}\n"
            f"  Superlevel: {params.get('superlevel', False)}\n"
            f"  Confidence: {params.get('confidence', 0):.3f}\n\n"
            f"Reasoning:\n{params['reasoning'][0]}"
        )
        axes[3].text(0.05, 0.95, stats_text, transform=axes[3].transAxes, fontsize=10, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='#444', alpha=0.8))
        axes[3].set_title("4. Analysis Parameters", weight='bold')
        axes[3].axis('off')

        # Panel 5: Threshold Overlay (NEW)
        overlay_img = np.stack([normalized_image] * 3, axis=-1)
        overlay_img[binary_image] = [1, 0, 0]  # Highlight thresholded pixels in red
        axes[4].imshow(overlay_img)
        axes[4].set_title("5. Threshold Overlay", weight='bold')
        axes[4].axis('off')

        # Panel 6: Processing Pipeline (Text)
        pipeline_text = (
            f"Processing Pipeline:\n"
            f"1. Load Image - {image.shape} pixels\n"
            f"2. Normalize - [0, 1] range\n"
            f"3. Select Threshold - {params.get('threshold', 0):.6f}\n"
            f"4. Superlevel Filtration\n"
            f"5. Cubical Complex Construction\n"
            f"6. Persistence Diagram Generation\n"
            f"7. Generate Diagrams & Barcodes\n\n"
            f"Next Steps:\n"
            f"- TDA Analysis\n"
            f"- Betti Number Calculation\n"
            f"- Persistence Statistics\n"
            f"- Visualization Generation"
        )
        axes[5].text(0.05, 0.95, pipeline_text, transform=axes[5].transAxes, fontsize=10, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='#003366', alpha=0.8))
        axes[5].set_title("6. Processing Pipeline", weight='bold')
        axes[5].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = Path(output_dir) / f"{filename}_step_by_step.png"
        plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Step-by-step visualization saved to {output_path}")

    def create_detailed_dashboard(self, image, params, persistence_data, betti_numbers, filename, output_dir):
        """Creates a comprehensive 9-panel TDA analysis dashboard."""
        if self.logger:
            self.logger.info(f"Creating detailed 9-panel dashboard for {filename}")

        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle(f"Comprehensive TDA Analysis: {filename}", fontsize=22, weight='bold', y=0.98)
        axes = axes.ravel()

        # Prepare normalized image for multiple panels
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        threshold = params.get('threshold', 0.5)
        superlevel = params.get('superlevel', False)

        # Panel 1: Original Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("1. Original Image", weight='bold', fontsize=14)
        axes[0].axis('off')

        # Panel 2: Normalized Image with Statistics
        im2 = axes[1].imshow(normalized_image, cmap='gray')
        axes[1].set_title("2. Normalized Image (0-1)", weight='bold', fontsize=14)
        axes[1].axis('off')
        # Add colorbar for reference
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Thresholded Binary Image
        if superlevel:
            binary_image = normalized_image >= threshold
            axes[2].set_title(f"3. Superlevel Set (≥{threshold:.3f})", weight='bold', fontsize=14)
        else:
            binary_image = normalized_image <= threshold
            axes[2].set_title(f"3. Sublevel Set (≤{threshold:.3f})", weight='bold', fontsize=14)
        axes[2].imshow(binary_image, cmap='gray', vmin=0, vmax=1)
        axes[2].axis('off')

        # Panel 4: Persistence Diagram
        axes[3].set_title("4. Persistence Diagram", weight='bold', fontsize=14)
        if persistence_data and len(persistence_data) > 0:
            try:
                import gudhi as gd
                gd.plot_persistence_diagram(persistence_data, axes=axes[3], legend=True)
                axes[3].set_xlabel("Birth", fontsize=12)
                axes[3].set_ylabel("Death", fontsize=12)
            except:
                axes[3].text(0.5, 0.5, "Persistence diagram\nunavailable", ha='center', va='center',
                           transform=axes[3].transAxes, fontsize=12)
        else:
            axes[3].text(0.5, 0.5, "No persistence data", ha='center', va='center',
                       transform=axes[3].transAxes, fontsize=12)
        axes[3].grid(True, alpha=0.3)

        # Panel 5: Threshold Overlay with Legend
        overlay_img = np.stack([normalized_image]*3, axis=-1)
        overlay_img[binary_image] = [1, 0.2, 0.2]  # Red highlight
        axes[4].imshow(overlay_img)
        axes[4].set_title("5. Threshold Overlay", weight='bold', fontsize=14)
        axes[4].axis('off')
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Selected pixels'),
                          Patch(facecolor='gray', label='Background')]
        axes[4].legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Panel 6: Multi-Threshold Comparison
        axes[5].set_title("6. Filtration Evolution", weight='bold', fontsize=14)
        thresholds = [threshold * 0.7, threshold, threshold * 1.3]
        colors = ['blue', 'red', 'green']
        for i, (t, color) in enumerate(zip(thresholds, colors)):
            if superlevel:
                mask = normalized_image >= t
            else:
                mask = normalized_image <= t
            pixel_count = np.sum(mask)
            axes[5].bar(i, pixel_count, color=color, alpha=0.7,
                       label=f'T={t:.3f} ({pixel_count} pixels)')
        axes[5].set_xlabel('Threshold Level', fontsize=12)
        axes[5].set_ylabel('Selected Pixels', fontsize=12)
        axes[5].legend(fontsize=10)
        axes[5].set_xticks([0, 1, 2])
        axes[5].set_xticklabels(['Low', 'Selected', 'High'])

        # Panel 7: Detailed Analysis Parameters
        betti_0 = betti_numbers.get('betti_0', 'N/A')
        betti_1 = betti_numbers.get('betti_1', 'N/A')
        stats_text = (
            f"IMAGE STATISTICS:\n"
            f"• Shape: {image.shape}\n"
            f"• Data Type: {image.dtype}\n"
            f"• Range: [{np.min(image):.3f}, {np.max(image):.3f}]\n"
            f"• Mean ± Std: {np.mean(image):.3f} ± {np.std(image):.3f}\n\n"
            f"TDA RESULTS:\n"
            f"• Connected Components (H₀): {betti_0}\n"
            f"• Loops/Holes (H₁): {betti_1}\n"
            f"• Total Features: {len(persistence_data) if persistence_data else 0}\n\n"
            f"PARAMETERS:\n"
            f"• Threshold: {params.get('threshold', 0):.6f}\n"
            f"• Filtration: {'Superlevel' if superlevel else 'Sublevel'}\n"
            f"• Confidence: {params.get('confidence', 0):.3f}\n\n"
            f"REASONING:\n"
            f"{params['reasoning'][0] if params.get('reasoning') else 'N/A'}"
        )
        axes[6].text(0.05, 0.95, stats_text, transform=axes[6].transAxes, fontsize=11,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9))
        axes[6].set_title("7. Analysis Summary", weight='bold', fontsize=14)
        axes[6].axis('off')

        # Panel 8: Intensity Histogram with Threshold
        axes[7].hist(normalized_image.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[7].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        axes[7].set_xlabel('Pixel Intensity', fontsize=12)
        axes[7].set_ylabel('Frequency', fontsize=12)
        axes[7].set_title("8. Intensity Distribution", weight='bold', fontsize=14)
        axes[7].legend(fontsize=10)
        axes[7].grid(True, alpha=0.3)

        # Panel 9: Processing Pipeline Flowchart
        pipeline_text = (
            f"PROCESSING PIPELINE:\n\n"
            f"1. 📁 Load Image → {image.shape} pixels\n"
            f"2. 🔄 Normalize → [0,1] range\n"
            f"3. 🎯 Auto-Threshold → {threshold:.4f}\n"
            f"4. 🔍 {'Superlevel' if superlevel else 'Sublevel'} Filtration\n"
            f"5. 🏗️  Cubical Complex Construction\n"
            f"6. 📊 Persistence Computation\n"
            f"7. 📈 Homology Analysis\n"
            f"8. 🎨 Visualization Generation\n\n"
            f"NEXT STEPS:\n"
            f"• Statistical Analysis\n"
            f"• Feature Comparison\n"
            f"• Report Generation\n"
            f"• Quality Assessment\n\n"
            f"STATUS: ✅ Complete\n"
            f"Time: {pd.Timestamp.now().strftime('%H:%M:%S')}"
        )
        axes[8].text(0.05, 0.95, pipeline_text, transform=axes[8].transAxes, fontsize=11,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='#1a3d5c', alpha=0.9))
        axes[8].set_title("9. Processing Pipeline", weight='bold', fontsize=14)
        axes[8].axis('off')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96], pad=2.0)
        output_path = Path(output_dir) / f"{filename}_detailed_dashboard.png"
        plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        plt.close(fig)
        if self.logger:
            self.logger.info(f"Detailed dashboard saved to {output_path}")

