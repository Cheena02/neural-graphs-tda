#!/usr/bin/env python3
"""
Summary generation and reporting module for TDA pipeline
"""
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

class SummaryGenerator:
    def __init__(self, logger=None):
        self.logger = logger
    
    def generate_dataset_summary(self, dataset_name: str, results: list, output_dir: Path):
        """Generate comprehensive summary report for a dataset"""
        
        if not results:
            if self.logger:
                self.logger.warning(f"No results to summarize for dataset: {dataset_name}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Generate summary statistics
        summary_stats = {
            'dataset_name': dataset_name,
            'total_images': len(df),
            'processing_date': pd.Timestamp.now().isoformat(),
            'betti_numbers': {
                'betti_0_mean': float(df['betti_0'].mean()),
                'betti_0_std': float(df['betti_0'].std()),
                'betti_0_min': int(df['betti_0'].min()),
                'betti_0_max': int(df['betti_0'].max()),
                'betti_1_mean': float(df['betti_1'].mean()),
                'betti_1_std': float(df['betti_1'].std()),
                'betti_1_min': int(df['betti_1'].min()),
                'betti_1_max': int(df['betti_1'].max())
            },
            'persistence_stats': {
                'h0_total_mean': float(df['h0_total_persistence'].mean()),
                'h0_total_std': float(df['h0_total_persistence'].std()),
                'h1_total_mean': float(df['h1_total_persistence'].mean()),
                'h1_total_std': float(df['h1_total_persistence'].std()),
                'h0_avg_lifespan_mean': float(df['avg_h0_lifespan'].replace([np.inf, -np.inf], np.nan).mean()),
                'h1_avg_lifespan_mean': float(df['avg_h1_lifespan'].mean())
            },
            'parameter_usage': {
                'superlevel_count': int(df['superlevel'].sum()),
                'sublevel_count': int((~df['superlevel']).sum()),
                'avg_threshold': float(df['threshold'].mean()),
                'avg_confidence': float(df['confidence'].mean())
            }
        }
        
        # Save detailed CSV
        csv_path = output_dir / f"{dataset_name}_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary JSON
        summary_path = output_dir / f"{dataset_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate summary visualization
        self.create_summary_plots(df, dataset_name, output_dir)
        
        if self.logger:
            self.logger.info(f"📋 Dataset summary generated for {dataset_name}")
            self.logger.info(f"   📊 CSV: {csv_path}")
            self.logger.info(f"   📋 Summary: {summary_path}")
    
    def create_summary_plots(self, df: pd.DataFrame, dataset_name: str, output_dir: Path):
        """Create summary plots for dataset analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dataset Summary: {dataset_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Betti Numbers Distribution
        axes[0, 0].hist(df['betti_0'], bins=20, alpha=0.7, label='Betti 0', color='blue')
        axes[0, 0].hist(df['betti_1'], bins=20, alpha=0.7, label='Betti 1', color='red')
        axes[0, 0].set_xlabel('Betti Numbers')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Betti Numbers Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Persistence vs Betti Numbers
        axes[0, 1].scatter(df['betti_0'], df['h0_total_persistence'], alpha=0.6, label='H0')
        axes[0, 1].scatter(df['betti_1'], df['h1_total_persistence'], alpha=0.6, label='H1')
        axes[0, 1].set_xlabel('Betti Numbers')
        axes[0, 1].set_ylabel('Total Persistence')
        axes[0, 1].set_title('Persistence vs Betti Numbers')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter Usage
        param_counts = df['superlevel'].value_counts()
        axes[0, 2].pie(param_counts.values, labels=['Sublevel', 'Superlevel'], autopct='%1.1f%%')
        axes[0, 2].set_title('Parameter Usage (Superlevel vs Sublevel)')
        
        # Plot 4: Threshold Distribution
        axes[1, 0].hist(df['threshold'], bins=20, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Threshold Values')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Threshold Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Confidence Scores
        axes[1, 1].hist(df['confidence'], bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Confidence Scores')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Parameter Selection Confidence')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics Table
        axes[1, 2].axis('off')
        summary_text = f"""Summary Statistics:

Total Images: {len(df)}
Avg Betti 0: {df['betti_0'].mean():.1f} ± {df['betti_0'].std():.1f}
Avg Betti 1: {df['betti_1'].mean():.1f} ± {df['betti_1'].std():.1f}

Persistence:
H0 Total: {df['h0_total_persistence'].mean():.3f} ± {df['h0_total_persistence'].std():.3f}
H1 Total: {df['h1_total_persistence'].mean():.3f} ± {df['h1_total_persistence'].std():.3f}

Parameters:
Avg Threshold: {df['threshold'].mean():.6f}
Avg Confidence: {df['confidence'].mean():.3f}
Superlevel Usage: {(df['superlevel'].sum() / len(df) * 100):.1f}%"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_title('Summary Statistics')
        
        # Save summary plots
        plt.tight_layout()
        summary_plot_path = output_dir / f"{dataset_name}_summary_plots.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.logger:
            self.logger.info(f"   📈 Summary plots: {summary_plot_path}")
    
    def generate_overall_summary(self, all_results: list, output_path: Path):
        """Generate overall summary across all datasets"""
        
        if not all_results:
            return
        
        df = pd.DataFrame(all_results)
        
        # Group by dataset/subfolder
        dataset_summary = df.groupby('subfolder').agg({
            'betti_0': ['mean', 'std', 'min', 'max'],
            'betti_1': ['mean', 'std', 'min', 'max'],
            'h0_total_persistence': ['mean', 'std'],
            'h1_total_persistence': ['mean', 'std'],
            'confidence': 'mean',
            'superlevel': 'mean'
        }).round(4)
        
        # Save comprehensive summary
        dataset_summary.to_csv(output_path)
        
        if self.logger:
            self.logger.info(f"📊 Overall summary saved: {output_path}")
            self.logger.info(f"   📈 Processed {len(df)} images across {df['subfolder'].nunique()} datasets")
