#!/usr/bin/env python3
"""
Enhanced TDA Pipeline - Clean and Modular with Comprehensive Analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import os
import glob
import json

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from src.data_io.enhanced_loader import EnhancedDataLoader
from src.tda.oldcubical import cubical_diagrams
from src.noise.generation import NoiseGenerator
from src.noise.mitigation import DenoisingStrategies
from src.visualization.plotter import TDAVisualizer
from src.utils.logger import TDALogger, log_method_call
from src.tda.adaptive_parameters import AdaptiveParameterSelector

# Import the new analysis modules (put these files in src/analysis/)
from src.analysis.comprehensive_analyzer import ComprehensiveAnalyzer

# CONFIGURATION
ONEDRIVE_PATH = r"C:\Users\cheen\OneDrive - The University Of Newcastle\Deriving and Analysing Graphs from Neural Activity\Dataset Analysis\data\raw_data"

DATASETS_TO_RUN = [
    "MOUSEBIRN",        # 8 images - good for testing (JPG files)
    # "synthetic_data",   # 18 images (PNG files)
    # "defungi",          # Will process H1, H2, H3, H5, H6 automatically
    # "nucmm",            # Skip for now - has H5 file issues
]

# Run all datasets option
RUN_ALL_DATASETS = False  # Set to True to process everything
EXCLUDE_DATASETS = ["nucmm"]  # Skip datasets with H5 issues

class EnhancedTDAExperimentPipeline:
    """
    Enhanced TDA experiment pipeline with comprehensive analysis and step-by-step visualization.
    """

    def __init__(self, results_dir: str, onedrive_path: str, datasets_to_run: list):
        self.onedrive_path = onedrive_path
        self.datasets_to_run = datasets_to_run
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TDALogger(name="Enhanced_TDA_Pipeline", log_dir=self.results_dir / "logs", level="INFO")
        self.loader = EnhancedDataLoader(logger=self.logger)
        self.visualizer = TDAVisualizer(output_dir=self.results_dir / "plots", logger=self.logger)
        self.noise_generator = NoiseGenerator()
        self.denoiser = DenoisingStrategies(logger=self.logger)
        self.all_metrics = []
        self.param_selector = AdaptiveParameterSelector(logger=self.logger)
        
        # Add comprehensive analyzer
        self.comprehensive_analyzer = ComprehensiveAnalyzer(logger=self.logger)
        self.last_params = {}  # Store params for step-by-step visualization

    @log_method_call
    def run(self):
        try:
            # Determine which datasets to process
            if RUN_ALL_DATASETS:
                discovered_datasets = self.discover_all_datasets()
                datasets_to_process = [d for d in discovered_datasets if d not in EXCLUDE_DATASETS]
                self.logger.info(f"🌍 Running ALL datasets mode")
                self.logger.info(f"📊 Discovered {len(discovered_datasets)} datasets")
                self.logger.info(f"✅ Processing {len(datasets_to_process)} datasets (excluding: {EXCLUDE_DATASETS})")
            else:
                datasets_to_process = self.datasets_to_run
                self.logger.info(f"🎯 Running SELECTED datasets mode")
                self.logger.info(f"📊 Processing {len(datasets_to_process)} selected datasets")

            self.logger.start_experiment(
                experiment_name=f"Enhanced TDA Analysis: {'ALL' if RUN_ALL_DATASETS else 'SELECTED'} Datasets",
                parameters={
                    "mode": "ALL" if RUN_ALL_DATASETS else "SELECTED",
                    "datasets": datasets_to_process, 
                    "path": self.onedrive_path
                }
            )

            self.logger.info(f"🚀 Starting comprehensive analysis of {len(datasets_to_process)} datasets")
            
            # Process each selected dataset
            for i, dataset_name in enumerate(datasets_to_process):
                self.logger.info(f"📁 [{i+1}/{len(datasets_to_process)}] Processing dataset: {dataset_name}")
                self.process_dataset(dataset_name)

            self.finalize_results()
            self.logger.end_experiment(status="completed")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.logger.end_experiment(status="failed")
            self.finalize_results()

    def discover_all_datasets(self):
        """Discover all available datasets in the OneDrive folder"""
        if not os.path.exists(self.onedrive_path):
            self.logger.error(f"OneDrive path not found: {self.onedrive_path}")
            return []
        
        all_datasets = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.h5', '*.hdf5', '*.npy']
        
        for item in os.listdir(self.onedrive_path):
            item_path = os.path.join(self.onedrive_path, item)
            if os.path.isdir(item_path):
                has_images = False
                for ext in extensions:
                    if glob.glob(os.path.join(item_path, '**', ext), recursive=True):
                        has_images = True
                        break
                
                if has_images:
                    all_datasets.append(item)
        
        return all_datasets

    def discover_dataset_structure(self, dataset_name: str):
        """Discover dataset structure and return all image-containing folders"""
        dataset_path = os.path.join(self.onedrive_path, dataset_name)
        
        if not os.path.exists(dataset_path):
            return []
        
        subfolders_with_images = []
        # Focus on standard image formats, skip H5 for now
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']
        
        for root, dirs, files in os.walk(dataset_path):
            images = []
            for ext in extensions:
                images.extend(glob.glob(os.path.join(root, ext)))
            
            if images:
                relative_path = os.path.relpath(root, self.onedrive_path)
                subfolders_with_images.append({
                    'name': relative_path,
                    'path': root,
                    'images': images,
                    'image_count': len(images)
                })
        
        return subfolders_with_images

    def process_dataset(self, dataset_name: str):
        """Process dataset with automatic subfolder detection and comprehensive analysis"""
        
        self.logger.info(f"🔍 Analyzing dataset structure: {dataset_name}")
        
        subfolders = self.discover_dataset_structure(dataset_name)
        
        if not subfolders:
            self.logger.warning(f"No supported images found in dataset: {dataset_name} (skipping H5 files)")
            return
        
        self.logger.info(f"📊 Dataset structure for {dataset_name}:")
        for subfolder in subfolders:
            self.logger.info(f"   📂 {subfolder['name']}: {subfolder['image_count']} images")
        
        total_images = sum(sf['image_count'] for sf in subfolders)
        self.logger.info(f"📈 Total images to process: {total_images}")
        
        # Process each subfolder with comprehensive analysis
        for i, subfolder in enumerate(subfolders):
            self.logger.info(f"🔍 [{i+1}/{len(subfolders)}] Processing subfolder: {subfolder['name']}")
            self.process_subfolder(subfolder, dataset_name)
        
        # Generate comprehensive dataset summary
        dataset_results = [r for r in self.all_metrics if r.get('subfolder', '').startswith(dataset_name)]
        if dataset_results:
            self.logger.info(f"📋 Generating comprehensive summary for {dataset_name}")
            self.comprehensive_analyzer.finalize_dataset_analysis(
                dataset_name, dataset_results, self.results_dir / "summaries"
            )

    def process_subfolder(self, subfolder_info: dict, dataset_name: str):
        """Process all images in a subfolder with comprehensive analysis"""
        
        subfolder_name = subfolder_info['name']
        images = subfolder_info['images']
        
        # Create organized output structure
        subfolder_output = self.results_dir / "comprehensive_results" / dataset_name / subfolder_name.replace(os.sep, '_')
        subfolder_output.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        (subfolder_output / "diagrams").mkdir(exist_ok=True)
        (subfolder_output / "barcodes").mkdir(exist_ok=True)
        (subfolder_output / "step_by_step").mkdir(exist_ok=True)
        (subfolder_output / "analysis_reports").mkdir(exist_ok=True)
        
        self.logger.info(f"   📁 Output directory: {subfolder_output}")
        
        for i, image_path in enumerate(images):
            try:
                image_name = Path(image_path).stem
                self.logger.info(f"   🖼️  [{i+1}/{len(images)}] Processing: {image_name}")
                
                # Load image
                image, metadata = self.loader.load_image(image_path)
                
                # Create simple metadata if needed
                class SimpleMetadata:
                    def __init__(self, filename):
                        self.filename = filename
                
                if not hasattr(metadata, 'filename'):
                    metadata = SimpleMetadata(image_path)
                
                # Comprehensive image processing
                self.process_single_image_comprehensive(image, metadata, subfolder_output, subfolder_name)
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to process {Path(image_path).name}: {e}")
                continue

    def process_single_image_comprehensive(self, image: np.ndarray, metadata, output_dir: Path, subfolder_name: str):
        """Comprehensive single image processing with all analysis features"""
        
        image_name = Path(metadata.filename).stem
        self.logger.info(f"      📊 Image analysis: {image_name}")
        self.logger.info(f"         Shape: {image.shape}, Type: {image.dtype}")

        # Step 1: Enhanced TDA analysis with detailed logging
        baseline_diags = self.analyze_and_log_comprehensive(image, image_name, "baseline", "clean", subfolder_name)
        
        # Step 2: Create step-by-step visualization using the comprehensive analyzer
        self.logger.info(f"      🎨 Creating step-by-step visualization...")
        self.comprehensive_analyzer.analyze_image_comprehensive(
            image, self.last_params, image_name, output_dir / "step_by_step"
        )
        
        # Step 3: Create standard TDA visualizations with enhanced colors
        self.logger.info(f"      📈 Creating persistence diagrams and barcodes...")
        diagram_viz = TDAVisualizer(output_dir / "diagrams", color_scheme="professional", logger=self.logger)
        barcode_viz = TDAVisualizer(output_dir / "barcodes", color_scheme="professional", logger=self.logger)
        
        # Generate both diagrams and barcodes
        diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_diagram")
        barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_barcode")
        
        # Step 4: Create individual analysis report
        self.create_individual_analysis_report(image, self.last_params, baseline_diags, image_name, output_dir / "analysis_reports")
        
        self.logger.info(f"      ✅ Comprehensive analysis complete for {image_name}")

    def analyze_and_log_comprehensive(self, image: np.ndarray, image_name: str, stage: str, variant: str, subfolder_name: str) -> list:
        """Comprehensive TDA analysis with detailed logging and metrics"""
        
        # Normalize image
        img_float = image.astype(np.float32) / 255.0

        # Adaptive parameter selection with detailed logging
        params = self.param_selector.select_optimal_parameters(image)
        self.last_params = params  # Store for step-by-step visualization
        
        self.logger.info(f"      🎯 Adaptive parameters selected:")
        self.logger.info(f"         - Threshold: {params['threshold']:.6f}")
        self.logger.info(f"         - Superlevel: {params['superlevel']}")
        self.logger.info(f"         - Confidence: {params.get('confidence', 0.0):.3f}")
        self.logger.info(f"         - Reasoning: {', '.join(params['reasoning'])}")

        # Run TDA analysis
        diags_dict = cubical_diagrams(img_float, superlevel=params['superlevel'])
        
        # Convert to persistence list format
        persistence_list = []
        for dim, diags in diags_dict.items():
            if diags.size > 0:
                for interval in diags:
                    persistence_list.append((int(dim[-1]), tuple(interval)))

        # Calculate comprehensive metrics
        h0_intervals = diags_dict.get("H0", np.array([]))
        h1_intervals = diags_dict.get("H1", np.array([]))
        
        betti_0 = len(h0_intervals)
        betti_1 = np.sum(np.isinf(h1_intervals[:, 1]) == False) if h1_intervals.size > 0 else 0

        # Calculate persistence statistics
        h0_lifespans = h0_intervals[:, 1] - h0_intervals[:, 0] if h0_intervals.size > 0 else np.array([])
        h1_finite_intervals = h1_intervals[np.isinf(h1_intervals[:, 1]) == False] if h1_intervals.size > 0 else np.array([])
        h1_lifespans = h1_finite_intervals[:, 1] - h1_finite_intervals[:, 0] if h1_finite_intervals.size > 0 else np.array([])

        total_h0_persistence = np.sum(h0_lifespans[np.isfinite(h0_lifespans)]) if h0_lifespans.size > 0 else 0
        total_h1_persistence = np.sum(h1_lifespans) if h1_lifespans.size > 0 else 0
        avg_h0_lifespan = np.mean(h0_lifespans) if h0_lifespans.size > 0 else 0
        avg_h1_lifespan = np.mean(h1_lifespans) if h1_lifespans.size > 0 else 0

        # Log comprehensive results
        self.logger.info(f"      📈 TDA Analysis Results:")
        self.logger.info(f"         - Betti 0 (connected components): {betti_0}")
        self.logger.info(f"         - Betti 1 (loops/holes): {betti_1}")
        self.logger.info(f"         - Total H0 persistence: {total_h0_persistence:.6f}")
        self.logger.info(f"         - Total H1 persistence: {total_h1_persistence:.6f}")
        self.logger.info(f"         - Average H0 lifespan: {avg_h0_lifespan:.6f}")
        self.logger.info(f"         - Average H1 lifespan: {avg_h1_lifespan:.6f}")

        # Store comprehensive metrics
        metrics = {
            "subfolder": subfolder_name,
            "image_name": image_name,
            "stage": stage,
            "variant": variant,
            "image_shape": str(image.shape),
            "threshold": params['threshold'],
            "superlevel": params['superlevel'],
            "confidence": params.get('confidence', 0.0),
            "betti_0": betti_0,
            "betti_1": betti_1,
            "h0_total_persistence": total_h0_persistence,
            "h1_total_persistence": total_h1_persistence,
            "avg_h0_lifespan": avg_h0_lifespan,
            "avg_h1_lifespan": avg_h1_lifespan,
            "reasoning": '; '.join(params['reasoning'])
        }
        self.all_metrics.append(metrics)
        self.logger.log_tda_results(**metrics)
        
        return persistence_list

    def create_individual_analysis_report(self, image, params, persistence_list, image_name, output_dir):
        """Create individual analysis report for each image"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a detailed text report
        report_content = f"""
TDA Analysis Report: {image_name}
{'='*50}

Image Information:
- Filename: {image_name}
- Shape: {image.shape}
- Data Type: {image.dtype}
- Min Value: {np.min(image):.6f}
- Max Value: {np.max(image):.6f}
- Mean Value: {np.mean(image):.6f}
- Standard Deviation: {np.std(image):.6f}

Selected Parameters:
- Threshold: {params['threshold']:.6f}
- Superlevel Filtration: {params['superlevel']}
- Confidence Score: {params.get('confidence', 0.0):.3f}

Parameter Selection Reasoning:
{chr(10).join(f"- {reason}" for reason in params['reasoning'])}

TDA Results:
- Total Persistence Features: {len(persistence_list)}
- H0 Features (Connected Components): {len([p for p in persistence_list if p[0] == 0])}
- H1 Features (Loops/Holes): {len([p for p in persistence_list if p[0] == 1])}

Persistence Intervals:
H0 (Connected Components):
{chr(10).join(f"  [{interval[0]:.6f}, {interval[1]:.6f})" for dim, interval in persistence_list if dim == 0)}

H1 (Loops/Holes):
{chr(10).join(f"  [{interval[0]:.6f}, {interval[1]:.6f})" for dim, interval in persistence_list if dim == 1)}

Analysis Timestamp: {pd.Timestamp.now().isoformat()}
"""
        
        # Save the report
        report_path = output_dir / f"{image_name}_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        if self.logger:
            self.logger.info(f"      📄 Individual report saved: {report_path}")

    def finalize_results(self):
        """Generate comprehensive final reports and summaries"""
        if not self.all_metrics:
            self.logger.warning("No metrics were generated during analysis.")
            return

        # Save comprehensive CSV
        df = pd.DataFrame(self.all_metrics)
        csv_path = self.results_dir / "comprehensive_experiment_metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"📊 Comprehensive metrics saved to: {csv_path}")
        
        # Generate overall comprehensive summary
        self.logger.info(f"📋 Generating overall comprehensive summary...")
        self.comprehensive_analyzer.finalize_overall_analysis(
            self.all_metrics, self.results_dir / "overall_comprehensive_summary.csv"
        )
        
        # Generate final summary statistics
        summary_stats = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_images_processed': len(df),
            'unique_datasets': df['subfolder'].nunique(),
            'unique_subfolders': df['subfolder'].nunique(),
            'betti_number_stats': {
                'avg_betti_0': float(df['betti_0'].mean()),
                'std_betti_0': float(df['betti_0'].std()),
                'max_betti_0': int(df['betti_0'].max()),
                'avg_betti_1': float(df['betti_1'].mean()),
                'std_betti_1': float(df['betti_1'].std()),
                'max_betti_1': int(df['betti_1'].max())
            },
            'parameter_usage_stats': {
                'avg_confidence': float(df['confidence'].mean()),
                'superlevel_usage_percent': float((df['superlevel'] == True).sum() / len(df) * 100),
                'avg_threshold': float(df['threshold'].mean()),
                'threshold_std': float(df['threshold'].std())
            },
            'persistence_stats': {
                'avg_h0_total_persistence': float(df['h0_total_persistence'].mean()),
                'avg_h1_total_persistence': float(df['h1_total_persistence'].mean()),
                'avg_h0_lifespan': float(df['avg_h0_lifespan'].mean()),
                'avg_h1_lifespan': float(df['avg_h1_lifespan'].mean())
            }
        }
        
        # Save final summary
        summary_path = self.results_dir / "final_comprehensive_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        self.logger.info(f"📋 Final comprehensive summary saved to: {summary_path}")
        self.logger.info(f"🎉 ANALYSIS COMPLETE!")
        self.logger.info(f"   📊 Processed {summary_stats['total_images_processed']} images")
        self.logger.info(f"   📁 Across {summary_stats['unique_datasets']} datasets")
        self.logger.info(f"   📈 Average Betti 0: {summary_stats['betti_number_stats']['avg_betti_0']:.1f}")
        self.logger.info(f"   📈 Average Betti 1: {summary_stats['betti_number_stats']['avg_betti_1']:.1f}")
        self.logger.info(f"   🎯 Average Confidence: {summary_stats['parameter_usage_stats']['avg_confidence']:.3f}")


if __name__ == "__main__":
    print("🚀 Enhanced TDA Pipeline with Comprehensive Step-by-Step Analysis")
    print(f"📁 Data source: {ONEDRIVE_PATH}")
    print(f"🎯 Selected datasets: {DATASETS_TO_RUN}")
    print(f"🔧 Run all datasets: {RUN_ALL_DATASETS}")
    if EXCLUDE_DATASETS:
        print(f"⚠️  Excluding datasets: {EXCLUDE_DATASETS}")

    pipeline = EnhancedTDAExperimentPipeline(
        results_dir="comprehensive_experiment_results",
        onedrive_path=ONEDRIVE_PATH,
        datasets_to_run=DATASETS_TO_RUN
    )
    
    pipeline.run()
    print("🎉 All datasets processed with comprehensive detailed analysis and visualizations!")
    print("📁 Check the 'comprehensive_experiment_results' folder for:")
    print("   📊 Step-by-step visualizations")
    print("   📈 Persistence diagrams and barcodes") 
    print("   📋 Individual analysis reports")
    print("   📊 Dataset summaries")
    print("   📈 Overall comprehensive summary")
