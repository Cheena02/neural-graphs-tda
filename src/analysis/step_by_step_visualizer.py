#!/usr/bin/env python3
"""
Step-by-step visualization module for TDA pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class StepByStepVisualizer:
    def __init__(self, logger=None):
        self.logger = logger
    
    def create_step_by_step_visualization(self, image: np.ndarray, params: dict, filename: str, output_dir: Path):
        """Create step-by-step visualization showing image processing stages"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Step-by-Step TDA Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # Step 1: Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('1. Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Step 2: Normalized Image (0-1 range)
        img_normalized = image.astype(np.float32) / 255.0
        axes[0, 1].imshow(img_normalized, cmap='gray')
        axes[0, 1].set_title('2. Normalized Image (0-1)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Step 3: Thresholded Image (based on selected threshold)
        threshold = params['threshold']
        if params['superlevel']:
            img_thresholded = img_normalized > threshold
            thresh_title = f'3. Superlevel Set (>{threshold:.3f})'
        else:
            img_thresholded = img_normalized < threshold
            thresh_title = f'3. Sublevel Set (<{threshold:.3f})'
        
        axes[0, 2].imshow(img_thresholded, cmap='gray')
        axes[0, 2].set_title(thresh_title, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Step 4: Image Statistics
        axes[1, 0].axis('off')
        stats_text = f"""Image Statistics:
        
Shape: {image.shape}
Data Type: {image.dtype}
Min Value: {np.min(image):.3f}
Max Value: {np.max(image):.3f}
Mean: {np.mean(image):.3f}
Std Dev: {np.std(image):.3f}

Selected Parameters:
Threshold: {threshold:.6f}
Superlevel: {params['superlevel']}
Confidence: {params.get('confidence', 0.0):.3f}

Reasoning:
{chr(10).join(params['reasoning'])}"""
        
        axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 0].set_title('4. Analysis Parameters', fontweight='bold')
        
        # Step 5: Histogram
        axes[1, 1].hist(image.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(threshold * 255, color='red', linestyle='--', linewidth=2, 
                           label=f'Threshold: {threshold:.3f}')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('5. Intensity Histogram', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Step 6: Processing Summary
        axes[1, 2].axis('off')
        processing_summary = f"""Processing Pipeline:

1. Load Image → {image.shape} pixels
2. Normalize → [0, 1] range  
3. Apply Threshold → {threshold:.6f}
4. {'Superlevel' if params['superlevel'] else 'Sublevel'} Filtration
5. Cubical Complex Construction
6. Persistence Homology Computation
7. Generate Diagrams & Barcodes

Next Steps:
→ TDA Analysis
→ Betti Number Calculation  
→ Persistence Statistics
→ Visualization Generation"""
        
        axes[1, 2].text(0.05, 0.95, processing_summary, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('6. Processing Pipeline', fontweight='bold')
        
        # Save the step-by-step visualization
        plt.tight_layout()
        step_by_step_path = output_dir / f"{filename}_step_by_step.png"
        plt.savefig(step_by_step_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.logger:
            self.logger.info(f"      📊 Step-by-step visualization saved: {step_by_step_path}")
