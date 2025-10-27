
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
            margin = 0.05
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
    
    def plot_persistence_diagram_publication(self, persistence_dict, title, output_filename,
                                            persistence_threshold=0.01):
        """
        Publication-quality persistence diagram with clean styling.
        
        Args:
            persistence_dict: Dict with 'H0' and 'H1' keys containing persistence intervals
            title: Clean title for the plot
            output_filename: Output path
            persistence_threshold: Minimum persistence to display (default: 0.01)
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        
        # Extract data
        h0 = persistence_dict.get('H0', np.array([]))
        h1 = persistence_dict.get('H1', np.array([]))
        
        # Filter and plot
        def plot_dimension(data, color, label, marker):
            if len(data) == 0:
                return
            finite_data = data[np.isfinite(data[:, 1])]
            if len(finite_data) == 0:
                return
            pers = finite_data[:, 1] - finite_data[:, 0]
            mask = pers > persistence_threshold
            filtered = finite_data[mask]
            if len(filtered) > 0:
                ax.scatter(filtered[:, 0], filtered[:, 1], 
                          c=color, label=label, alpha=0.6, s=30, marker=marker,
                          edgecolors='white', linewidth=0.5)
        
        # Find max value for diagonal
        all_data = []
        if len(h0) > 0:
            all_data.append(h0)
        if len(h1) > 0:
            all_data.append(h1)
        
        if len(all_data) > 0:
            all_points = np.vstack(all_data)
            finite_points = all_points[np.isfinite(all_points).all(axis=1)]
            if len(finite_points) > 0:
                max_val = np.max(finite_points)
            else:
                max_val = 1.0
        else:
            max_val = 1.0
        
        # Plot features
        plot_dimension(h0, '#E74C3C', 'H₀ (Components)', 'o')
        plot_dimension(h1, '#3498DB', 'H₁ (Holes)', 's')
        
        # Diagonal line
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5, label='Birth = Death')
        
        # Styling
        ax.set_xlabel('Birth', fontsize=14, weight='bold')
        ax.set_ylabel('Death', fontsize=14, weight='bold')
        ax.set_title(title, fontsize=16, weight='bold', pad=15)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        # Add persistence threshold line
        if persistence_threshold > 0:
            x_thresh = np.linspace(0, max_val - persistence_threshold, 100)
            y_thresh = x_thresh + persistence_threshold
            ax.plot(x_thresh, y_thresh, 'g--', alpha=0.3, linewidth=1.5,
                   label=f'Persistence threshold ({persistence_threshold:.3f})')
            ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        # Save
        output_path = Path(output_filename) if isinstance(output_filename, str) else output_filename
        if not output_path.suffix:
            output_path = output_path.with_suffix('.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Saved publication-quality PH diagram to: {output_path}")
        
        return str(output_path)
    
    def plot_persistence_barcode_publication(self, persistence_dict, title, output_filename,
                                            max_features=200):
        """
        Publication-quality barcode plot showing BOTH H0 and H1.
        
        Args:
            persistence_dict: Dict with 'H0' and 'H1' keys
            title: Clean title
            output_filename: Output path
            max_features: Maximum features to display per dimension (default: 200)
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Extract and filter data
        h0 = persistence_dict.get('H0', np.array([]))
        h1 = persistence_dict.get('H1', np.array([]))
        
        # Filter and sort by persistence
        def filter_and_sort(data, max_count):
            if len(data) == 0:
                return np.array([])
            
            # Remove infinite deaths
            finite_data = data[np.isfinite(data[:, 1])]
            
            # Calculate persistence
            if len(finite_data) == 0:
                return np.array([])
            
            pers = finite_data[:, 1] - finite_data[:, 0]
            
            # Sort by persistence (descending)
            sorted_indices = np.argsort(pers)[::-1]
            filtered = finite_data[sorted_indices]
            
            # Limit to max_count
            if len(filtered) > max_count:
                filtered = filtered[:max_count]
            
            return filtered
        
        h0_filtered = filter_and_sort(h0, max_features)
        h1_filtered = filter_and_sort(h1, max_features)
        
        # Plot bars
        y_pos = 0
        
        # H0 bars (red)
        for i, (birth, death) in enumerate(h0_filtered):
            ax.barh(y_pos, death - birth, left=birth, height=0.8,
                   color='#E74C3C', alpha=0.7, edgecolor='white', linewidth=0.5)
            y_pos += 1
        
        h0_count = len(h0_filtered)
        
        # Add separator
        if h0_count > 0 and len(h1_filtered) > 0:
            ax.axhline(y_pos - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
            y_pos += 1
        
        # H1 bars (blue)
        for i, (birth, death) in enumerate(h1_filtered):
            ax.barh(y_pos, death - birth, left=birth, height=0.8,
                   color='#3498DB', alpha=0.7, edgecolor='white', linewidth=0.5)
            y_pos += 1
        
        h1_count = len(h1_filtered)
        
        # Styling
        ax.set_xlabel('Filtration Parameter', fontsize=14, weight='bold')
        ax.set_ylabel('Topological Features (sorted by persistence)', fontsize=14, weight='bold')
        ax.set_title(title, fontsize=16, weight='bold', pad=15)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.7, 
                  label=f'H₀ Components (top {max_features})'),
            Patch(facecolor='#3498DB', alpha=0.7, 
                  label=f'H₁ Holes (top {max_features})')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12,
                 framealpha=0.95, edgecolor='gray', fancybox=True)
        
        # Grid
        ax.grid(True, axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.5, y_pos)
        
        # Remove y-axis ticks (too many features)
        ax.set_yticks([])
        
        # Add text showing filtering info
        total_h0 = len(h0) if len(h0) > 0 else 0
        total_h1 = len(h1) if len(h1) > 0 else 0
        info_text = (f"Displaying top {max_features} most persistent features per dimension\n"
                    f"Total features: H₀={total_h0}, H₁={total_h1} | "
                    f"Displayed: H₀={h0_count}, H₁={h1_count}")
        ax.text(0.5, -0.08, info_text, transform=ax.transAxes,
               ha='center', fontsize=10, style='italic', color='gray')
        
        # Save
        output_path = Path(output_filename) if isinstance(output_filename, str) else output_filename
        if not output_path.suffix:
            output_path = output_path.with_suffix('.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Saved publication-quality barcode to: {output_path}")
        
        return str(output_path)
    
    def plot_betti_evolution(self, persistence_dict, title, output_filename, num_steps=100):
        """
        Plot how Betti numbers evolve during filtration.
        
        Args:
            persistence_dict: Dict with 'H0' and 'H1' keys
            title: Plot title
            output_filename: Output path
            num_steps: Number of filtration steps to compute
        """
        plt.style.use('default')
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Extract data
        h0 = persistence_dict.get('H0', np.array([]))
        h1 = persistence_dict.get('H1', np.array([]))
        
        # Find filtration range
        all_values = []
        for data in [h0, h1]:
            if len(data) > 0:
                finite_data = data[np.isfinite(data).all(axis=1)]
                if len(finite_data) > 0:
                    all_values.extend(finite_data.flatten())
        
        if len(all_values) == 0:
            if self.logger:
                self.logger.warning("No finite data for Betti evolution plot")
            return
        
        min_val = min(all_values)
        max_val = max(all_values)
        filtration_values = np.linspace(min_val, max_val, num_steps)
        
        # Compute Betti numbers at each filtration value
        betti_0 = []
        betti_1 = []
        
        for t in filtration_values:
            # Count features alive at time t
            b0 = 0
            b1 = 0
            
            if len(h0) > 0:
                for birth, death in h0:
                    if birth <= t < death or (np.isinf(death) and birth <= t):
                        b0 += 1
            
            if len(h1) > 0:
                for birth, death in h1:
                    if birth <= t < death or (np.isinf(death) and birth <= t):
                        b1 += 1
            
            betti_0.append(b0)
            betti_1.append(b1)
        
        # Plot β₀
        ax1.plot(filtration_values, betti_0, color='#E74C3C', linewidth=2, label='β₀ (Components)')
        ax1.set_xlabel('Filtration Parameter', fontsize=14, weight='bold')
        ax1.set_ylabel('β₀ (Components)', fontsize=14, weight='bold', color='#E74C3C')
        ax1.tick_params(axis='y', labelcolor='#E74C3C')
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Create second y-axis for β₁
        ax2 = ax1.twinx()
        ax2.plot(filtration_values, betti_1, color='#3498DB', linewidth=2, label='β₁ (Holes)')
        ax2.set_ylabel('β₁ (Holes)', fontsize=14, weight='bold', color='#3498DB')
        ax2.tick_params(axis='y', labelcolor='#3498DB')
        
        # Title
        ax1.set_title(title, fontsize=16, weight='bold', pad=15)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12, framealpha=0.95)
        
        # Save
        output_path = Path(output_filename) if isinstance(output_filename, str) else output_filename
        if not output_path.suffix:
            output_path = output_path.with_suffix('.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Saved Betti evolution plot to: {output_path}")
        
        return str(output_path)

