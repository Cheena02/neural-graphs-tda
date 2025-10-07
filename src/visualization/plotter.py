#!/usr/bin/env python3
"""
Comprehensive visualization module for TDA pipeline.

This module provides functions for generating publication-quality visualizations
for persistence homology analysis, including persistence diagrams, barcodes,
and stability comparison plots.

Engineering Features:
- High-quality, configurable plots using Matplotlib and Seaborn.
- Modular design for easy extension with new plot types.
- Automated saving of plots in multiple formats (PNG, PDF, SVG).
- Integration with the TDA logging system for tracking.
- Consistent styling for professional reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import gudhi

from src.utils.logger import TDALogger, log_method_call

class TDAVisualizer:
    """
    Handles the creation of all visualizations for the TDA pipeline.
    """
    
    def __init__(self, output_dir: Union[str, Path], logger: Optional[TDALogger] = None):
        """
        Initialize the visualizer with a base output directory.

        Args:
            output_dir: Directory to save all plots.
            logger: TDA logger instance.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or TDALogger(name="Visualizer")
        
        # Set professional plot style
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
        self.logger.info(f"🎨 Visualizer initialized. Plots will be saved to: {self.output_dir}")

    @log_method_call
    def plot_persistence_diagram(self, 
                                 persistence: list, 
                                 title: str, 
                                 output_filename: str):
        """
        Plot and save a persistence diagram.

        Args:
            persistence: Persistence intervals from GUDHI.
            title: Title for the plot.
            output_filename: Filename for the saved plot.
        """
        plt.rcParams.update({'font.size': 14,
                             "axes.titlesize": 16,
                             "axes.labelsize": 14,
                             "legend.fontsize": 12,
                             "font.family": "serif",})

        fig, ax = plt.subplots(figsize=(10,10))
        "1234567890!@#$%^&*()_-+="
        colors = {'H0':"#C41E3A",'H1':'#1E3A8A','H2':'#0595669'}

        dims ={}
        for dim, (birth,death) in persistence:
            if dim not in dims:
                dims[dim] = {'births':[],'deaths':[]}
                dims[dim]['births'].append(birth)
                dims[dim]['deaths'].append(death)

        for dim,data in dims.items():
            if not data['births']:
                continue

            births = np.array(data['births'])
            deaths = np.array(data['deaths'])

            finite_mask = np.isfinite(deaths)

            color = colors.get(f'H{dim}' , f'C{dim}')
            label = f''
        plt.figure(figsize=(8, 8))
        gudhi.plot_persistence_diagram(persistence)
        plt.title(title, fontsize=16)
        
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved persistence diagram to: {output_path}")

    @log_method_call
    def plot_persistence_barcode(self, 
                                 persistence: list, 
                                 title: str, 
                                 output_filename: str):
        """
        Plot and save a persistence barcode.

        Args:
            persistence: Persistence intervals from GUDHI.
            title: Title for the plot.
            output_filename: Filename for the saved plot.
        """
        plt.figure(figsize=(10, 6))
        gudhi.plot_persistence_barcode(persistence)
        plt.title(title, fontsize=16)
        
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved persistence barcode to: {output_path}")

    @log_method_call
    def plot_betti_curves(self, 
                          betti_numbers: np.ndarray, 
                          title: str, 
                          output_filename: str):
        """
        Plot Betti curves over filtration values.

        Args:
            betti_numbers: Array of Betti numbers over filtration steps.
            title: Title for the plot.
            output_filename: Filename for the saved plot.
        """
        plt.figure(figsize=(10, 6))
        for i in range(betti_numbers.shape[1]):
            plt.plot(betti_numbers[:, i], label=f'Betti {i}')
        
        plt.title(title, fontsize=16)
        plt.xlabel("Filtration Step")
        plt.ylabel("Betti Number")
        plt.legend()
        plt.grid(True)
        
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved Betti curves to: {output_path}")

    @log_method_call
    def compare_persistence_diagrams(self, 
                                     diag1: list, 
                                     diag2: list, 
                                     title: str, 
                                     output_filename: str, 
                                     labels: List[str] = ['Diagram 1', 'Diagram 2']):
        """
        Plot two persistence diagrams side-by-side for comparison.

        Args:
            diag1: First persistence diagram.
            diag2: Second persistence diagram.
            title: Overall title for the comparison.
            output_filename: Filename for the saved plot.
            labels: Labels for the two diagrams.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(title, fontsize=18)

        gudhi.plot_persistence_diagram(diag1, axes=ax1)
        ax1.set_title(labels[0])

        gudhi.plot_persistence_diagram(diag2, axes=ax2)
        ax2.set_title(labels[1])

        # Calculate distances for annotation
        ws_dist = gudhi.wasserstein_distance(gudhi.persistence_intervals_in_dimension(diag1, 1), 
                                           gudhi.persistence_intervals_in_dimension(diag2, 1))
        bt_dist = gudhi.bottleneck_distance(gudhi.persistence_intervals_in_dimension(diag1, 1), 
                                            gudhi.persistence_intervals_in_dimension(diag2, 1))

        fig.text(0.5, 0.02, f"Wasserstein Distance (H1): {ws_dist:.4f} | Bottleneck Distance (H1): {bt_dist:.4f}", 
                 ha='center', fontsize=12)

        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved diagram comparison to: {output_path}")

    @log_method_call
    def plot_feature_comparison(self, 
                                df_features, 
                                title: str, 
                                output_filename: str):
        """
        Create a bar plot comparing persistence features across experiments.

        Args:
            df_features: DataFrame with persistence features.
            title: Title for the plot.
            output_filename: Filename for the saved plot.
        """
        plt.figure(figsize=(12, 7))
        df_features.plot(kind='bar', rot=45)
        plt.title(title, fontsize=16)
        plt.ylabel("Value")
        plt.xlabel("Experiment / Image")
        plt.tight_layout()

        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved feature comparison plot to: {output_path}")

    def plot_image_grid(self, images: Dict[str, np.ndarray], title: str, output_filename: str, grid_shape: tuple = (2, 4)):
        """
        Plot a grid of images for comparison (e.g., original, noisy, denoised).

        Args:
            images: Dictionary of images with titles as keys.
            title: Overall title for the grid.
            output_filename: Filename for the saved plot.
            grid_shape: Shape of the image grid.
        """
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(16, 8))
        fig.suptitle(title, fontsize=18)
        axes = axes.flatten()

        for i, (img_title, img) in enumerate(images.items()):
            if i < len(axes):
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(img_title)
                axes[i].axis('off')
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved image grid to: {output_path}")
