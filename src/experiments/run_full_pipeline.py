#!/usr/bin/env python3
"""
Enhanced TDA Pipeline with Dataset Selection and Detailed Analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import os
import glob

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from src.data_io.enhanced_loader import EnhancedDataLoader
from src.tda.oldcubical import cubical_diagrams
from src.noise.generation import NoiseGenerator
from src.noise.mitigation import DenoisingStrategies
from src.visualization.plotter import TDAVisualizer
from src.utils.logger import TDALogger, log_method_call
from src.tda.adaptive_parameters import AdaptiveParameterSelector
from src.tda.thresholds import auto_min_persistence
from src.analysis.comprehensive_analyzer import ComprehensiveAnalyzer


ONEDRIVE_PATH = r"C:\Users\cheen\OneDrive - The University Of Newcastle\Deriving and Analysing Graphs from Neural Activity\Dataset Analysis\data\raw_data"

DATASETS_TO_RUN = [
     "MOUSEBIRN",  # 8 images - good for testing
    #  "synthetic_data",   # 18 images
    #  "defungi",          # Will process H1, H2, H3, H5, H6 automatically
    #  "nucmm",            # Will process Mouse, Zebrafish subfolders
]
# CONFIGURATION
RUN_ALL_DATASETS = False  # Set to True to process everything
EXCLUDE_DATASETS = [" "]

# Add this after your existing DATASETS_TO_RUN configuration

# ENHANCED NOISE EXPERIMENT CONFIGURATION
RUN_NOISE_EXPERIMENTS = True
RUN_DENOISING_EXPERIMENTS = True  # Also test denoising strategies

# Configuration at the top of your file
EXPERIMENT_STAGES = {
    "run_clean": True,        # Stage 1: Process clean images
    "run_noise": True,        # Stage 2: Add noise variants
    "run_denoising": True     # Stage 3: Apply denoising to noisy images
}

# Noise configurations
NOISE_VARIANTS = [
    {"type": "gaussian", "param": 0.05, "suffix": "gaussian_0.05"},
    {"type": "gaussian", "param": 0.1, "suffix": "gaussian_0.1"},
    {"type": "salt_pepper", "param": 0.05, "suffix": "salt_pepper_0.05"},
    {"type": "salt_pepper", "param": 0.1, "suffix": "salt_pepper_0.1"},
]

DENOISING_METHODS = ["median_filter", "bilateral_filter", "non_local_means"]



class TDAExperimentPipeline:
    """
    Enhanced TDA experiment pipeline with dataset selection and detailed analysis.
    """

    def __init__(self, results_dir: str, onedrive_path: str, datasets_to_run: list):
        self.onedrive_path = onedrive_path
        self.datasets_to_run = datasets_to_run
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TDALogger(name="TDA_Pipeline", log_dir=self.results_dir / "logs", level="INFO")
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
                experiment_name=f" TDA Analysis: {'ALL' if RUN_ALL_DATASETS else 'SELECTED'} Datasets",
                parameters={
                    "mode": "ALL" if RUN_ALL_DATASETS else "SELECTED",
                    "datasets": datasets_to_process,
                    "path": self.onedrive_path
                }
            )

            self.logger.info(f"🚀 Starting comprehensive analysis of {len(datasets_to_process)} datasets")

            # Process each selected dataset
            for i, dataset_name in enumerate(datasets_to_process):
                self.logger.info(f"📁 [{i + 1}/{len(datasets_to_process)}] Processing dataset: {dataset_name}")
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
        """Process dataset with automatic subfolder detection"""

        self.logger.info(f"📁 Analyzing dataset structure: {dataset_name}")

        # Discover dataset structure
        subfolders = self.discover_dataset_structure(dataset_name)

        if not subfolders:
            self.logger.error(f"No images found in dataset: {dataset_name}")
            return

        self.logger.info(f"📊 Dataset structure for {dataset_name}:")
        for subfolder in subfolders:
            self.logger.info(f"   📂 {subfolder['name']}: {subfolder['image_count']} images")

        total_images = sum(sf['image_count'] for sf in subfolders)
        self.logger.info(f"📈 Total images to process: {total_images}")

        # Process each subfolder
        for subfolder in subfolders:
            self.logger.info(f"🔍 Processing subfolder: {subfolder['name']}")
            self.process_subfolder(subfolder, dataset_name)

    def process_subfolder(self, subfolder_info: dict, dataset_name: str):
        """Process all images in a subfolder"""

        subfolder_name = subfolder_info['name']
        images = subfolder_info['images']

        # Process each image
        for i, image_path in enumerate(images):
            try:
                image_name = Path(image_path).stem
                self.logger.info(f"   🖼️  [{i + 1}/{len(images)}] Processing: {image_name}")

                # ADD THIS LINE to create a unique directory for each image.
                image_output_dir = self.results_dir / "organized_results" / dataset_name / subfolder_name.replace(
                    os.sep, "_") / image_name
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # Load image
                image, metadata = self.loader.load_image(image_path)

                # Create metadata object if needed
                class SimpleMetadata:
                    def __init__(self, filename):
                        self.filename = filename

                if not hasattr(metadata, "filename"):
                    metadata = SimpleMetadata(image_path)

                # CHANGE THIS LINE to pass the new image-specific directory.
                self.process_single_image_enhanced(image, metadata, image_output_dir, subfolder_name)

            except Exception as e:
                self.logger.error(f"   ❌ Failed to process {Path(image_path).name}: {e}")
                continue

    # def process_single_image_enhanced(self, image: np.ndarray, metadata, output_dir: Path, subfolder_name: str):
    #     """Enhanced single image processing with detailed analysis and organized output."""
    #
    #     image_name = Path(metadata.filename).stem
    #     self.logger.info(f"      📊 Image details: shape={image.shape}, dtype={image.dtype}")
    #
    #     # Enhanced analysis to get persistence data
    #     baseline_diags = self.analyze_and_log_enhanced(image, image_name, "baseline", "clean", subfolder_name)
    #
    #     # --- START OF FINAL FIX ---
    #
    #     # 1. Create specific subdirectories inside the image's unique folder
    #     diagram_dir = output_dir / "diagrams"
    #     diagram_dir.mkdir(exist_ok=True)
    #
    #     barcode_dir = output_dir / "barcodes"
    #     barcode_dir.mkdir(exist_ok=True)
    #
    #     step_by_step_dir = output_dir / "step_by_step"
    #     step_by_step_dir.mkdir(exist_ok=True)
    #
    #     # 2. Call the analyzer to save the step-by-step plots
    #     self.comprehensive_analyzer.analyze_image_comprehensive(
    #         image, self.last_params, image_name, step_by_step_dir
    #     )
    #
    #     # 3. Initialize visualizers to use the new, specific directories
    #     diagram_viz = TDAVisualizer(diagram_dir, color_scheme="professional", logger=self.logger)
    #     barcode_viz = TDAVisualizer(barcode_dir, color_scheme="professional", logger=self.logger)
    #
    #     # --- END OF FINAL FIX ---
    #
    #     # This part remains the same, but now saves to the correct folders
    #     diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_diagram")
    #     barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_barcode")
    #
    #     self.logger.info(f"      ✅ All visualizations saved for {image_name} in: {output_dir}")

    def process_single_image_enhanced(self, image: np.ndarray, metadata, output_dir: Path, subfolder_name: str):
        """Enhanced processing with comprehensive noise and denoising experiments."""

        image_name = Path(metadata.filename).stem
        self.logger.info(f"      📊 Image details: shape={image.shape}, dtype={image.dtype}")

        # 1. Process CLEAN version (preserve existing results)
        clean_output_dir = output_dir / "clean"
        clean_output_dir.mkdir(parents=True, exist_ok=True)
        self._process_single_variant(image, image_name, clean_output_dir, subfolder_name, "clean")

        # 2. Process NOISE variants
        if RUN_NOISE_EXPERIMENTS:
            for noise_config in NOISE_EXPERIMENTS:
                noise_name = noise_config["name"]
                self.logger.info(f"        🔊 Adding {noise_name} noise to {image_name}")

                # Generate noisy version using your NoiseGenerator
                noisy_image = self.noise_generator.add_noise(
                    image,
                    noise_type=noise_config["type"],
                    **noise_config["params"]
                )

                # Create directory for this noise type
                noise_output_dir = output_dir / f"noise_{noise_name}"
                noise_output_dir.mkdir(parents=True, exist_ok=True)

                # Process the noisy variant
                self._process_single_variant(noisy_image, f"{image_name}_{noise_name}",
                                             noise_output_dir, subfolder_name, f"noise_{noise_name}")

                # 3. Process DENOISED variants (if enabled)
                if RUN_DENOISING_EXPERIMENTS:
                    for strategy in DENOISING_STRATEGIES:
                        self.logger.info(f"          🧹 Applying {strategy} to {noise_name} noise")

                        # Apply denoising strategy
                        try:
                            denoised_image = self.denoiser.denoise(noisy_image, method=strategy)

                            # Create directory for denoised variant
                            denoised_output_dir = output_dir / f"denoised_{noise_name}_{strategy}"
                            denoised_output_dir.mkdir(parents=True, exist_ok=True)

                            # Process the denoised variant
                            self._process_single_variant(
                                denoised_image,
                                f"{image_name}_{noise_name}_{strategy}",
                                denoised_output_dir,
                                subfolder_name,
                                f"denoised_{noise_name}_{strategy}"
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to apply {strategy} to {noise_name}: {e}")

    def _process_single_variant(self, image: np.ndarray, image_name: str, output_dir: Path,
                                subfolder_name: str, variant_type: str):
        """Process a single image variant with full analysis pipeline."""

        # Run your existing analysis
        analysis_results = self.analyze_and_log_enhanced(image, image_name, "baseline", variant_type, subfolder_name)
        baseline_diags = analysis_results["persistence_list"]
        betti_numbers = analysis_results["betti_numbers"]

        # Create subdirectories
        diagram_dir = output_dir / "diagrams"
        diagram_dir.mkdir(exist_ok=True)

        barcode_dir = output_dir / "barcodes"
        barcode_dir.mkdir(exist_ok=True)

        step_by_step_dir = output_dir / "step_by_step"
        step_by_step_dir.mkdir(exist_ok=True)

        # Generate all visualizations
        self.comprehensive_analyzer.analyze_image_comprehensive(
            image, self.last_params, baseline_diags, betti_numbers, image_name, step_by_step_dir
        )

        # Create plots
        diagram_viz = TDAVisualizer(diagram_dir, color_scheme="professional", logger=self.logger)
        barcode_viz = TDAVisualizer(barcode_dir, color_scheme="professional", logger=self.logger)

        diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_diagram")
        barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {image_name}", f"{image_name}_barcode")

        self.logger.info(f"      ✅ {variant_type.title()} variant complete for {image_name}")

    def analyze_and_log_enhanced(self, image: np.ndarray, image_name: str, stage: str, variant: str,
                                 subfolder_name: str) -> list:
        """Enhanced analysis with comprehensive logging and metrics"""

        # Convert image to proper format
        img_float = image.astype(np.float32)
        mp = auto_min_persistence(img_float)
        # AUTO-SELECT PARAMETERS with detailed logging
        params = self.param_selector.select_optimal_parameters(image)
        self.last_params = params
        self.logger.info(f"      🎯 Parameters selected:")
        self.logger.info(f"         - Threshold: {params['threshold']:.6f}")
        self.logger.info(f"         - Superlevel: {params['superlevel']}")
        self.logger.info(f"         - Confidence: {params.get('confidence', 0.0):.3f}")
        self.logger.info(f"         - Reasoning: {', '.join(params['reasoning'])}")
        self.logger.info(f"      🔎 Auto min_persistence set to {mp:.4f}")

        # Run TDA analysis
        diags_dict = cubical_diagrams(img_float, superlevel=params['superlevel'])
        for dim in ["H0", "H1"]:
            if diags_dict[dim].size > 0:
                lifespans = diags_dict[dim][:, 1] - diags_dict[dim][:, 0]
                diags_dict[dim] = diags_dict[dim][lifespans >= mp]

        # Convert to persistence format
        persistence_list = []
        for dim, diags in diags_dict.items():
            if diags.size > 0:
                for interval in diags:
                    persistence_list.append((int(dim[-1]), tuple(interval)))
        # Calculate comprehensive metrics
        h0_intervals = diags_dict.get("H0", np.array([]))
        h1_intervals = diags_dict.get("H1", np.array([]))

        # ----- Betti numbers -----
        # H0: count all components (includes the one infinite component)
        betti_0 = len(h0_intervals)

        # H1: only count finite loops (ignore those that never die)
        if h1_intervals.size > 0:
            h1_finite_intervals = h1_intervals[np.isfinite(h1_intervals[:, 1])]
            betti_1 = int(h1_finite_intervals.shape[0])
        else:
            h1_finite_intervals = np.array([])
            betti_1 = 0

        # ----- Persistence statistics (finite only for means/sums) -----
        # H0 lifespans
        if h0_intervals.size > 0:
            h0_lifespans = h0_intervals[:, 1] - h0_intervals[:, 0]
            h0_lifespans_finite = h0_lifespans[np.isfinite(h0_lifespans)]
        else:
            h0_lifespans = np.array([])
            h0_lifespans_finite = np.array([])

        # H1 lifespans (finite only)
        if h1_finite_intervals.size > 0:
            h1_lifespans = h1_finite_intervals[:, 1] - h1_finite_intervals[:, 0]
        else:
            h1_lifespans = np.array([])

        total_h0_persistence = float(np.sum(h0_lifespans_finite)) if h0_lifespans_finite.size else 0.0
        total_h1_persistence = float(np.sum(h1_lifespans))        if h1_lifespans.size        else 0.0
        avg_h0_lifespan      = float(np.mean(h0_lifespans_finite)) if h0_lifespans_finite.size else 0.0
        avg_h1_lifespan      = float(np.mean(h1_lifespans))        if h1_lifespans.size        else 0.0


        # # Calculate comprehensive metrics
        # h0_intervals = diags_dict.get("H0", np.array([]))
        # h1_intervals = diags_dict.get("H1", np.array([]))
        #
        # # Betti numbers
        # betti_0 = len(h0_intervals)
        # betti_1 = np.sum(np.isinf(h1_intervals[:, 1]) == False) if h1_intervals.size > 0 else 0
        #
        # # Persistence statistics
        # h0_lifespans = h0_intervals[:, 1] - h0_intervals[:, 0] if h0_intervals.size > 0 else np.array([])
        # h1_finite_intervals = h1_intervals[
        #     np.isinf(h1_intervals[:, 1]) == False] if h1_intervals.size > 0 else np.array([])
        # h1_lifespans = h1_finite_intervals[:, 1] - h1_finite_intervals[
        #     :, 0] if h1_finite_intervals.size > 0 else np.array([])
        #
        # total_h0_persistence = np.sum(h0_lifespans[np.isfinite(h0_lifespans)]) if h0_lifespans.size > 0 else 0
        # total_h1_persistence = np.sum(h1_lifespans) if h1_lifespans.size > 0 else 0
        # avg_h0_lifespan = np.mean(h0_lifespans) if h0_lifespans.size > 0 else 0
        # avg_h1_lifespan = np.mean(h1_lifespans) if h1_lifespans.size > 0 else 0

        # Log comprehensive results
        self.logger.info(f"      📈 TDA Results:")
        self.logger.info(f"         - Betti 0 (components): {betti_0}")
        self.logger.info(f"         - Betti 1 (holes): {betti_1}")
        self.logger.info(f"         - Total H0 persistence: {total_h0_persistence:.4f}")
        self.logger.info(f"         - Total H1 persistence: {total_h1_persistence:.4f}")
        self.logger.info(f"         - Average H0 lifespan: {avg_h0_lifespan:.4f}")
        self.logger.info(f"         - Average H1 lifespan: {avg_h1_lifespan:.4f}")

        # Store comprehensive metrics
        metrics = {
            "subfolder": subfolder_name,
            "image_name": image_name,
            "stage": stage,
            "variant": variant,
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

        return {
            "persistence_list": persistence_list,
            "betti_numbers": {"betti_0": betti_0, "betti_1": betti_1}
        }

    def create_individual_analysis_report(self, image, params, persistence_list, image_name, output_dir):
            """Create individual analysis report for each image"""

            output_dir.mkdir(parents=True, exist_ok=True)

            # Create a detailed text report
            report_content = f"""
    TDA Analysis Report: {image_name}
    {'=' * 50}

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
        """Generate comprehensive final reports"""
        if not self.all_metrics:
            self.logger.warning("No metrics were generated.")
            return

        # Create comprehensive CSV report
        df = pd.DataFrame(self.all_metrics)
        csv_path = self.results_dir / "comprehensive_experiment_metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"📊 Comprehensive metrics saved to: {csv_path}")

        # Generate summary statistics
        summary_stats = {
            'total_images_processed': len(df),
            'unique_datasets': df['subfolder'].nunique(),
            'avg_betti_0': df['betti_0'].mean(),
            'avg_betti_1': df['betti_1'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'superlevel_usage': (df['superlevel'] == True).sum() / len(df) * 100
        }

        summary_path = self.results_dir / "analysis_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        self.logger.info(f"📋 Final comprehensive summary saved to: {summary_path}")
        self.logger.info(f"🎉 ANALYSIS COMPLETE!")
        self.logger.info(f"   📊 Processed {summary_stats['total_images_processed']} images")
        self.logger.info(f"   📁 Across {summary_stats['unique_datasets']} datasets")
        self.logger.info(f"   📈 Average Betti 0: {summary_stats['avg_betti_0']:.1f}")
        self.logger.info(f"   📈 Average Betti 1: {summary_stats['avg_betti_1']:.1f}")
        self.logger.info(f"   🎯 Average Confidence: {summary_stats['avg_confidence']:.3f}")


if __name__ == "__main__":
    print("🚀  TDA Pipeline with Comprehensive Step-by-Step Analysis")
    print(f"📁 Data source: {ONEDRIVE_PATH}")
    print(f"🎯 Selected datasets: {DATASETS_TO_RUN}")
    print(f"🔧 Run all datasets: {RUN_ALL_DATASETS}")
    if EXCLUDE_DATASETS:
        print(f"⚠️  Excluding datasets: {EXCLUDE_DATASETS}")

    pipeline = TDAExperimentPipeline(
        results_dir="TDA_Analysis_Results",
        onedrive_path=ONEDRIVE_PATH,
        datasets_to_run=DATASETS_TO_RUN
    )

    pipeline.run()
    print("🎉 All datasets processed with comprehensive detailed analysis and visualizations!")
    print("📁 Check the '' folder for:")
    print("   📊 Step-by-step visualizations")
    print("   📈 Persistence diagrams and barcodes")
    print("   📋 Individual analysis reports")
    print("   📊 Dataset summaries")
    print("   📈 Overall comprehensive summary")
