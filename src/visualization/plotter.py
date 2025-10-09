
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
import seaborn as sns
import gudhi
from src.utils.logger import TDALogger, log_method_call
import matplotlib.patheffects as pe


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
        Plot persistence barcode with tight bounds and no white space.
        """
        total_bars = len(persistence)

        # Compact figure size
        if total_bars <= 2:
            fig_height = 2.5
        elif total_bars <= 5:
            fig_height = 4
        else:
            fig_height = 6

        plt.figure(figsize=(8, fig_height))  # Reduced width
        ax = plt.gca()

        # Organize data and find bounds
        dims = {}
        all_births = []
        all_finite_deaths = []

        for dim, (birth, death) in persistence:
            if dim not in dims:
                dims[dim] = []
            dims[dim].append((birth, death))
            all_births.append(birth)
            if np.isfinite(death):
                all_finite_deaths.append(death)

        # Calculate tight x-axis bounds
        if all_births and all_finite_deaths:
            min_val = min(min(all_births), min(all_finite_deaths))
            max_val = max(max(all_births), max(all_finite_deaths))
            range_val = max_val - min_val
            margin = max(range_val * 0.05, 0.001)  # Small margin
            x_min = min_val - margin
            x_max = max_val + margin
        else:
            x_min, x_max = 0, 1

        # extra room on the right so the ∞ glyph isn't clipped
        x_max = x_max + margin


        # Plot bars
        y_pos = 0
        dim_labels = {0: 'H₀', 1: 'H₁', 2: 'H₂'}

        for dim in sorted(dims.keys()):
            intervals = sorted(dims[dim], key=lambda x: x[0])

            # use a dedicated name for bar color (do not call it "color")
            bar_color = self.enhanced_colors.get(f'H{dim}', f'C{dim}')

            for birth, death in intervals:
                if np.isfinite(death):
                    bar_length = death - birth
                    ax.barh(
                        y_pos, bar_length, left=birth, height=0.7,
                        color=bar_color, alpha=self.alpha,  # <-- use bar_color
                        edgecolor='white', linewidth=1
                    )

                    # Label only meaningful bars (optional: tweak 0.02–0.05)
                    if bar_length > (x_max - x_min) * 0.03:
                        mid_point = birth + bar_length / 2
                        txt = ax.text(
                            mid_point, y_pos, f'{bar_length:.4f}',
                            ha='center', va='center', fontsize=9,
                            color='black', weight='bold'
                        )
                        # optional: uncomment if you added patheffects import
                        # txt.set_path_effects([pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()])
                else:
                    # Infinite bar
                    ax.barh(
                        y_pos, x_max - birth, left=birth, height=0.7,
                        color=bar_color, alpha=self.alpha,  # <-- use bar_color
                        edgecolor='white', linewidth=1
                    )
                    txt = ax.text(
                        x_max + (x_max - x_min) * 0.02, y_pos, '∞',
                        va='center', color='black', fontsize=12, weight='bold'
                    )
                    # optional outline:
                    # txt.set_path_effects([pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()])
        # for dim in sorted(dims.keys()):
        #     intervals = sorted(dims[dim], key=lambda x: x[0])
        #     color = self.enhanced_colors.get(f'H{dim}', f'C{dim}')
        #
        #     for birth, death in intervals:
        #         if np.isfinite(death):
        #             bar_length = death - birth
        #             ax.barh(y_pos, bar_length, left=birth, height=0.7,
        #                     color=color, alpha=self.alpha,
        #                     edgecolor='white', linewidth=1)
        #
        #             # Add persistence value
        #             if bar_length > (x_max - x_min) * 0.01:  # Only if significant relative to range
        #                 mid_point = birth + bar_length / 2
        #                 ax.text(mid_point, y_pos, f'{bar_length:.4f}',
        #                         ha='center', va='center', fontsize=9,
        #                         color='black', weight='bold')
        #         else:
        #             # Infinite bar
        #             ax.barh(y_pos, x_max - birth, left=birth, height=0.7,
        #                     color=color, alpha=self.alpha,
        #                     edgecolor='white', linewidth=1)
        #             ax.text(x_max + (x_max - x_min) * 0.02, y_pos, '∞',
        #                     va='center', color='black', fontsize=12, weight='bold')

                y_pos += 1

        # Add dimension labels
        y_pos = 0
        for dim in sorted(dims.keys()):
            intervals = dims[dim]
            if intervals:
                mid_y = y_pos + len(intervals) / 2 - 0.5
                label_color = self.enhanced_colors.get(f'H{dim}', f'C{dim}')
                ax.text(
                    x_min - (x_max - x_min) * 0.05, mid_y, dim_labels.get(dim, f'H{dim}'),
                    ha='right', va='center', fontsize=12, weight='bold', color=label_color
                )
        # for dim in sorted(dims.keys()):
        #     intervals = dims[dim]
        #     if intervals:
        #         mid_y = y_pos + len(intervals) / 2 - 0.5
        #         color = self.enhanced_colors.get(f'H{dim}', f'C{dim}')
        #         ax.text(x_min - (x_max - x_min) * 0.05, mid_y, dim_labels.get(dim, f'H{dim}'),
        #                 ha='right', va='center', fontsize=12, weight='bold', color=color)
                y_pos += len(intervals)

        # Set tight limits
        ax.set_xlim(x_min, x_max)  # TIGHT X-LIMITS
        ax.set_ylim(-0.4, total_bars - 0.6)  # TIGHT Y-LIMITS

        # Styling
        ax.set_facecolor(self.enhanced_colors['background'])
        ax.grid(True, alpha=0.3, axis='x', color=self.enhanced_colors['grid'])
        ax.set_xlabel('Filtration Parameter', fontweight='bold')
        ax.set_ylabel('Features', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_yticks([])

        # Remove extra margins
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

        # Save with minimal padding
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', pad_inches=0.01)
        plt.close()

        if self.logger:
            self.logger.info(f" Persistence barcode saved to: {output_path}")

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

