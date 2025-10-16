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
    # "MOUSEBIRN",  # 8 images - good for testing
     "synthetic_data",   # 18 images
    # "defungi",          # Will process H1, H2, H3, H5, H6 automatically
    # "nucmm",            # Will process Mouse, Zebrafish subfolders
]

# CONFIGURATION
RUN_ALL_DATASETS = False  # Set to True to process everything
EXCLUDE_DATASETS = [" "]

# NOISE EXPERIMENT CONFIGURATION
RUN_NOISE_EXPERIMENTS = True
RUN_DENOISING_EXPERIMENTS = True

# Noise experiments - using your exact structure
NOISE_EXPERIMENTS = [
    {"type": "gaussian", "param": 0.05, "name": "gaussian_0.05"},
    {"type": "gaussian", "param": 0.1, "name": "gaussian_0.1"},
    {"type": "salt_pepper", "param": 0.05, "name": "salt_pepper_0.05"},
    {"type": "salt_pepper", "param": 0.1, "name": "salt_pepper_0.1"},
]

DENOISING_METHODS = [
    {"method": "median_filter", "name": "median_filter"},
    {"method": "bilateral_filter", "name": "bilateral_filter"},
    {"method": "non_local_means", "name": "non_local_means"},
]


class TDAExperimentPipeline:
    """Enhanced TDA experiment pipeline with dataset selection and detailed analysis."""

    def __init__(self, results_dir: str, onedrive_path: str, datasets_to_run: list):
        self.onedrive_path = onedrive_path
        self.datasets_to_run = datasets_to_run
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TDALogger(name="TDA_Pipeline", log_dir=self.results_dir / "logs", level="INFO")
        self.loader = EnhancedDataLoader(logger=self.logger)
        self.visualizer = TDAVisualizer(output_dir=self.results_dir / "plots", logger=self.logger)

        # Initialize noise modules
        self.noise_generator = NoiseGenerator()
        self.denoising_strategies = DenoisingStrategies(logger=self.logger)

        self.all_metrics = []
        self.param_selector = AdaptiveParameterSelector(logger=self.logger)
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
                experiment_name=f"TDA Analysis: {'ALL' if RUN_ALL_DATASETS else 'SELECTED'} Datasets",
                parameters={
                    "mode": "ALL" if RUN_ALL_DATASETS else "SELECTED",
                    "datasets": datasets_to_process,
                    "path": self.onedrive_path,
                    "noise_experiments": RUN_NOISE_EXPERIMENTS,
                    "denoising_experiments": RUN_DENOISING_EXPERIMENTS
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

                # Create unique directory for each image
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

                # Process with your desired structure
                self.process_single_image_enhanced(image, metadata, image_output_dir, subfolder_name)

            except Exception as e:
                self.logger.error(f"   ❌ Failed to process {Path(image_path).name}: {e}")
                continue

    def process_single_image_enhanced(self, image: np.ndarray, metadata, output_dir: Path, subfolder_name: str):
        """Process image with baseline → noised → denoised structure"""

        image_name = Path(metadata.filename).stem
        self.logger.info(f"      📊 Processing {image_name}")

        # Create the three main directories
        baseline_dir = output_dir / "baseline"
        noised_dir = output_dir / "noised"
        denoised_dir = output_dir / "denoised"

        for dir_path in [baseline_dir, noised_dir, denoised_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # === BASELINE (CLEAN) ===
        self.logger.info(f"        🧹 Processing baseline (clean) for {image_name}")
        self._process_and_save_to_folder(
            image, f"{image_name}_clean", baseline_dir, subfolder_name, "baseline"
        )

        # === NOISED VARIANTS ===
        if RUN_NOISE_EXPERIMENTS:
            self.logger.info(f"        🔊 Processing noised variants for {image_name}")

            # Convert to uint8 for noise generation
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

            # Store noisy images for denoising stage
            noisy_images_store = {}

            for noise_exp in NOISE_EXPERIMENTS:
                try:
                    # Generate noise using your modules
                    if noise_exp["type"] == "gaussian":
                        noisy_dict = self.noise_generator.add_gaussian_noise(image_uint8, [noise_exp["param"]])
                    elif noise_exp["type"] == "salt_pepper":
                        noisy_dict = self.noise_generator.add_salt_pepper_noise(image_uint8, [noise_exp["param"]])
                    else:
                        self.logger.warning(f"Unknown noise type: {noise_exp['type']}")
                        continue

                    noisy_uint8 = list(noisy_dict.values())[0]
                    noisy_float = noisy_uint8.astype(np.float32) / 255.0

                    # Store for denoising
                    noisy_images_store[noise_exp["name"]] = {
                        "float": noisy_float,
                        "uint8": noisy_uint8
                    }

                    # Process and save to noised folder
                    variant_name = f"{image_name}_{noise_exp['name']}"
                    self._process_and_save_to_folder(
                        noisy_float, variant_name, noised_dir, subfolder_name, f"noised_{noise_exp['name']}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to generate {noise_exp['type']} noise: {e}")

            # === DENOISED VARIANTS ===
            if RUN_DENOISING_EXPERIMENTS:
                self.logger.info(f"        🧹 Processing denoised variants for {image_name}")

                for noise_name, noisy_data in noisy_images_store.items():
                    for denoising in DENOISING_METHODS:
                        try:
                            # Apply denoising using your modules
                            denoising_func = getattr(self.denoising_strategies, f"apply_{denoising['method']}")
                            denoised_uint8 = denoising_func(noisy_data["uint8"])
                            denoised_float = denoised_uint8.astype(np.float32) / 255.0

                            # Process and save to denoised folder
                            denoised_name = f"{image_name}_{noise_name}_{denoising['name']}"
                            self._process_and_save_to_folder(
                                denoised_float, denoised_name, denoised_dir,
                                subfolder_name, f"denoised_{noise_name}_{denoising['name']}"
                            )

                        except Exception as e:
                            self.logger.error(f"Failed {denoising['method']} on {noise_name}: {e}")

    def _process_and_save_to_folder(self, image: np.ndarray, variant_name: str,
                                    save_dir: Path, subfolder_name: str, variant_type: str):
        """Process image and save all results to a single folder"""

        # Run TDA analysis
        analysis_results = self.analyze_and_log_enhanced(image, variant_name, "baseline", variant_type, subfolder_name)

        # analysis_results is the persistence dict from cubical_diagrams
        persistence_dict = analysis_results
        # Convert to list format for visualizer
        baseline_diags = []
        for dim in [0, 1]:
            if f"H{dim}" in persistence_dict:
                for interval in persistence_dict[f"H{dim}"]:
                    baseline_diags.append((dim, interval))

        # Create visualizers that save to the same directory
        diagram_viz = TDAVisualizer(save_dir, color_scheme="professional", logger=self.logger)
        barcode_viz = TDAVisualizer(save_dir, color_scheme="professional", logger=self.logger)

        # Save plots with descriptive names
        diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_ph_diagram")
        barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_ph_barcode")

        # Save step-by-step
        try:
            self.comprehensive_analyzer.analyze_image_comprehensive(
                image, self.last_params, baseline_diags, variant_name, save_dir
            )
        except Exception as e:
            self.logger.error(f"Failed step-by-step for {variant_name}: {e}")

        self.logger.info(f"          ✅ {variant_name} saved to {save_dir.name}/")

    def analyze_and_log_enhanced(self, image: np.ndarray, image_name: str, stage: str, variant: str,
                                 subfolder_name: str) -> list:
        """Enhanced analysis with comprehensive logging and metrics"""

        # Convert image to proper format
        img_float = image.astype(np.float32)
        mp = auto_min_persistence(img_float)

        # AUTO-SELECT PARAMETERS with detailed logging
        params = self.param_selector.select_optimal_parameters(image)
        self.last_params = params  # Store for step-by-step visualization

        self.logger.info(f"      🎯 Parameters selected:")
        self.logger.info(f"         Threshold: {params['threshold']:.6f}")
        self.logger.info(f"         Superlevel: {params['superlevel']}")
        self.logger.info(f"         Confidence: {params.get('confidence', 0.0):.3f}")

        # COMPUTE PERSISTENCE DIAGRAMS
        persistence_dict = cubical_diagrams(
            img_float,
            superlevel=params['superlevel']
        )

        # CALCULATE BETTI NUMBERS
        betti_0 = len(persistence_dict.get("H0", []))
        betti_1 = len(persistence_dict.get("H1", []))

        self.logger.info(f"      📊 TDA Results:")
        self.logger.info(f"         Total features: {len(persistence_dict)}")
        self.logger.info(f"         Betti 0 (components): {betti_0}")
        self.logger.info(f"         Betti 1 (holes): {betti_1}")

        # STORE METRICS
        metrics = {
            'image_name': image_name,
            'subfolder': subfolder_name,
            'stage': stage,
            'variant': variant,
            'threshold': params['threshold'],
            'superlevel': params['superlevel'],
            'confidence': params.get('confidence', 0.0),
            'total_features': len(persistence_dict),
            'betti_0': betti_0,
            'betti_1': betti_1,
            'min_persistence': mp,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.all_metrics.append(metrics)

        return persistence_dict

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
    print("🚀 TDA Pipeline with Comprehensive Noise Analysis")
    print(f"📁 Data source: {ONEDRIVE_PATH}")
    print(f"🎯 Selected datasets: {DATASETS_TO_RUN}")
    print(f"🔧 Run all datasets: {RUN_ALL_DATASETS}")
    print(f"🔊 Noise experiments: {RUN_NOISE_EXPERIMENTS}")
    print(f"🧹 Denoising experiments: {RUN_DENOISING_EXPERIMENTS}")
    if EXCLUDE_DATASETS:
        print(f"⚠️  Excluding datasets: {EXCLUDE_DATASETS}")

    pipeline = TDAExperimentPipeline(
        results_dir="TDA_Analysis_Results",
        onedrive_path=ONEDRIVE_PATH,
        datasets_to_run=DATASETS_TO_RUN
    )

    pipeline.run()
    print("🎉 All datasets processed with comprehensive noise analysis!")
    print("📁 Check the 'TDA_Analysis_Results' folder for:")
    print("   📂 baseline/ - Clean image results")
    print("   📂 noised/ - All noise variant results")
    print("   📂 denoised/ - All denoised variant results")
    print("   📊 Comprehensive metrics and summaries")
