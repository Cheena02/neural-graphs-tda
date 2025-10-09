
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

# !/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional
import gudhi
from src.utils.logger import TDALogger, log_method_call



class TDAVisualizer:
    def __init__(self, output_dir: Union[str, Path], color_scheme: str = "vibrant", logger: Optional[TDALogger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or TDALogger(name="Visualizer")

        self.color_schemes = {
            'vibrant': {'H0': '#FF6B6B', 'H1': '#4ECDC4', 'H2': '#45B7D1'},
            'professional': {'H0': '#E74C3C', 'H1': '#3498DB', 'H2': '#2ECC71'},
            'warm': {'H0': '#FF8C42', 'H1': '#FF6B35', 'H2': '#F7931E'},
            'nature': {'H0': '#27AE60', 'H1': '#2ECC71', 'H2': '#58D68D'},
            'sunset': {'H0': '#E67E22', 'H1': '#F39C12', 'H2': '#F4D03F'}
        }
        self.colors = self.color_schemes.get(color_scheme, self.color_schemes['professional'])
        # Create enhanced_colors with ALL required keys for barcode function
        self.enhanced_colors = {
            'H0': self.colors['H0'],
            'H1': self.colors['H1'],
            'H2': self.colors['H2'],
            'background': '#FAFAFA',  # Light background
            'grid': '#E0E0E0'  # Grid color
        }

        # Alpha for transparency
        self.alpha = 0.8

    @log_method_call
    def plot_persistence_diagram(self, persistence: list, title: str, output_filename: str):
        plt.style.use('default')
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
        Plot persistence barcode with self-contained styling to guarantee correct bar colors.
        """
        # Reset style to prevent any possible outside interference.
        plt.style.use("default")

        # --- START OF THE DEFINITIVE FIX ---
        # Define the colors LOCALLY inside the function. This makes it immune to class state issues.
        color_map = {
            0: "#DC143C",  # Crimson Red for H0
            1: "#4682B4",  # Steel Blue for H1
            2: "#32CD32",  # Lime Green for H2 (future-proof)
        }
        # --- END OF THE DEFINITIVE FIX ---

        # Filtering logic to prevent a solid black plot (this part is working correctly)
        max_intervals = 250
        if len(persistence) > max_intervals:
            if self.logger:
                self.logger.info(
                    f"Filtering {len(persistence)} intervals to the most significant {max_intervals} for barcode clarity.")
            persistence.sort(key=lambda x: (x[1][1] - x[1][0]) if np.isfinite(x[1][1]) else np.inf, reverse=True)
            persistence = persistence[:max_intervals]

        if not persistence:
            if self.logger:
                self.logger.warning(f"No persistence data to plot for {output_filename}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Data preparation (this part is fine)
        dims, all_births, all_finite_deaths = {}, [], []
        for dim, (birth, death) in persistence:
            if dim not in dims: dims[dim] = []
            dims[dim].append((birth, death))
            all_births.append(birth)
            if np.isfinite(death): all_finite_deaths.append(death)

        # X-axis limit calculation (this part is fine)
        if all_births and all_finite_deaths:
            min_val, max_val = min(all_births), max(all_finite_deaths)
            margin = max((max_val - min_val) * 0.05, 0.001)
            x_min, x_max = min_val - margin, max_val + margin
        else:
            x_min, x_max = 0, 1
        x_max += margin

        # --- Plotting Loop ---
        y_pos = 0
        dim_labels = {0: "H₀", 1: "H₁", 2: "H₂"}
        for dim in sorted(dims.keys()):
            intervals = sorted(dims[dim], key=lambda x: x[0])

            # Use the LOCAL color_map to get the correct color for the current dimension.
            bar_color = color_map.get(dim, "#333333")

            for birth, death in intervals:
                if np.isfinite(death):
                    bar_length = death - birth
                    ax.barh(y_pos, bar_length, left=birth, height=0.8, color=bar_color, edgecolor=None)
                else:
                    ax.barh(y_pos, x_max - birth, left=birth, height=0.8, color=bar_color, edgecolor=None)
                    ax.text(x_max + (x_max - x_min) * 0.01, y_pos, "∞", va="center", color="black", fontsize=12)
                y_pos += 1

        # --- Labeling Loop ---
        y_pos = 0
        for dim in sorted(dims.keys()):
            if intervals := dims[dim]:
                mid_y = y_pos + len(intervals) / 2 - 0.5
                label_color = color_map.get(dim, "#333333")
                ax.text(x_min - (x_max - x_min) * 0.05, mid_y, dim_labels.get(dim, f"H{dim}"), ha="right", va="center",
                        fontsize=14, weight="bold", color=label_color)
                y_pos += len(intervals)

        # Final styling
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1, len(persistence))
        ax.set_xlabel("Filtration Parameter", fontweight="bold", fontsize=12)
        ax.set_ylabel("Features", fontweight="bold", fontsize=12)
        ax.set_title(title, fontweight="bold", pad=15, fontsize=16)
        ax.set_yticks([])  # Hide the y-axis ticks
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(True, which="major", axis="x", linestyle=":", color="#cccccc")

        plt.tight_layout()
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if self.logger: self.logger.info(f"Persistence barcode saved to: {output_path}")

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

    def _get_persistence_stats(self, dims: dict) -> str:
        """Generate persistence statistics text for annotation."""
        stats = []
        for dim, data in dims.items():
            persistences = [p for p in data['persistences'] if np.isfinite(p) and p > 0]
            if persistences:
                max_pers = max(persistences)
                mean_pers = np.mean(persistences)
                count = len(persistences)
                stats.append(f"H{dim}: {count} features, max={max_pers:.3f}, avg={mean_pers:.3f}")
        return '\n'.join(stats)

