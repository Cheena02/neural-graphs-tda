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
from src.tda.cubical import cubical_diagrams
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
    # "synthetic_data",   # 18 images
    # "defungi/H1",          # Will process H1, H2, H3, H5, H6 automatically
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

        # ORIGINAL WORKING LOGGER INITIALIZATION
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

    def analyze_and_log_enhanced(self, image: np.ndarray, image_name: str, stage: str, variant: str,
                                 subfolder_name: str) -> dict:
        """Enhanced analysis with comprehensive logging and metrics - FIXED"""

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

        # COMPUTE PERSISTENCE DIAGRAMS - FIXED: Remove threshold parameter
        persistence_dict = cubical_diagrams(
            img_float,
            superlevel=params['superlevel']
        )

        # CALCULATE BETTI NUMBERS - FIXED: Use dictionary format
        betti_0 = len(persistence_dict.get("H0", []))
        betti_1 = len(persistence_dict.get("H1", []))
        total_features = betti_0 + betti_1

        self.logger.info(f"      📊 TDA Results:")
        self.logger.info(f"         Total features: {total_features}")
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
            'total_features': total_features,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'min_persistence': mp,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.all_metrics.append(metrics)

        return persistence_dict

    def _process_and_save_to_folder(self, image: np.ndarray, variant_name: str,
                                    save_dir: Path, subfolder_name: str, variant_type: str):
        """Process image and save all results to a single folder - FIXED"""

        # Run TDA analysis
        persistence_dict = self.analyze_and_log_enhanced(image, variant_name, "baseline", variant_type, subfolder_name)

        # FIXED: Convert dictionary format to list format for visualizer
        baseline_diags = []
        betti_numbers = {"betti_0": 0, "betti_1": 0}

        if isinstance(persistence_dict, dict):
            # Convert from {"H0": [...], "H1": [...]} to [(dim, (birth, death)), ...]
            for dim in [0, 1]:
                if f"H{dim}" in persistence_dict:
                    for interval in persistence_dict[f"H{dim}"]:
                        baseline_diags.append((dim, interval))
                    betti_numbers[f"betti_{dim}"] = len(persistence_dict[f"H{dim}"])

        # Create visualizers that save to the same directory
        diagram_viz = TDAVisualizer(output_dir=save_dir, logger=self.logger)
        barcode_viz = TDAVisualizer(output_dir=save_dir, logger=self.logger)

        # Save plots with descriptive names
        diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_ph_diagram")
        barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_ph_barcode")

        # FIXED: Save step-by-step with correct parameters
        try:
            self.comprehensive_analyzer.analyze_image_comprehensive(
                image,  # image
                self.last_params,  # params
                variant_name,  # filename
                save_dir  # output_dir
            )
        except Exception as e:
            self.logger.error(f"Failed step-by-step for {variant_name}: {e}")

        self.logger.info(f"          ✅ {variant_name} saved to {save_dir.name}/")

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

            experiment_id = self.logger.start_experiment(
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

            for i, dataset_name in enumerate(datasets_to_process, 1):
                self.logger.info(f"📊 [{i}/{len(datasets_to_process)}] Processing dataset: {dataset_name}")
                self.process_dataset(dataset_name)

            self.finalize_results()
            self.logger.end_experiment(experiment_id, "completed")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def discover_all_datasets(self):
        """Discover all available datasets in the OneDrive path."""
        base_path = Path(self.onedrive_path)
        if not base_path.exists():
            self.logger.error(f"OneDrive path does not exist: {base_path}")
            return []

        datasets = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                datasets.append(item.name)

        self.logger.info(f"🔍 Discovered datasets: {datasets}")
        return datasets

    def process_dataset(self, dataset_name: str):
        """Process a single dataset with comprehensive analysis."""

        self.logger.info(f"📊 Processing dataset: {dataset_name}")

        # Get dataset path
        dataset_path = Path(self.onedrive_path) / dataset_name

        if not dataset_path.exists():
            self.logger.error(f"Dataset path not found: {dataset_path}")
            return

        # Load images
        images_data = self.loader.load_dataset_images(dataset_path, dataset_name)

        if not images_data:
            self.logger.warning(f"No images found in {dataset_name}")
            return

        # Process each image
        for i, (subfolder_name, image_name, image_path, image) in enumerate(images_data, 1):
            self.logger.info(f"      [{i}/{len(images_data)}] Processing: {image_name}")

            try:
                # Create organized results directory
                organized_dir = self.results_dir / "organized_results" / dataset_name / subfolder_name / image_name

                # Process baseline (clean image)
                baseline_dir = organized_dir / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)

                self.logger.info(f"        Processing {image_name}")
                self.logger.info(f"          Processing baseline (clean) for {image_name}")

                self._process_and_save_to_folder(image, f"{image_name}_clean", baseline_dir, subfolder_name, "baseline")

                # Process noised variants if enabled
                if RUN_NOISE_EXPERIMENTS:
                    self.logger.info(f"          Processing noised variants for {image_name}")

                    noised_dir = organized_dir / "noised"
                    noised_dir.mkdir(parents=True, exist_ok=True)

                    for noise_config in NOISE_EXPERIMENTS:
                        noise_type = noise_config["type"]
                        noise_param = noise_config["param"]
                        noise_name = noise_config["name"]

                        # Generate noisy image
                        if noise_type == "gaussian":
                            noisy_image = self.noise_generator.add_gaussian_noise(
                                (image * 255).astype(np.uint8), sigma=noise_param * 255
                            )
                        elif noise_type == "salt_pepper":
                            noisy_image = self.noise_generator.add_salt_pepper_noise(
                                (image * 255).astype(np.uint8), amount=noise_param
                            )

                        noisy_image_float = noisy_image.astype(np.float32) / 255.0

                        variant_name = f"{image_name}_{noise_name}"
                        self._process_and_save_to_folder(noisy_image_float, variant_name, noised_dir, subfolder_name,
                                                         "noised")

                # Process denoised variants if enabled
                if RUN_DENOISING_EXPERIMENTS:
                    self.logger.info(f"          Processing denoised variants for {image_name}")

                    denoised_dir = organized_dir / "denoised"
                    denoised_dir.mkdir(parents=True, exist_ok=True)

                    # Apply denoising to each noised variant
                    for noise_config in NOISE_EXPERIMENTS:
                        noise_type = noise_config["type"]
                        noise_param = noise_config["param"]
                        noise_name = noise_config["name"]

                        # Generate the same noisy image
                        if noise_type == "gaussian":
                            noisy_image = self.noise_generator.add_gaussian_noise(
                                (image * 255).astype(np.uint8), sigma=noise_param * 255
                            )
                        elif noise_type == "salt_pepper":
                            noisy_image = self.noise_generator.add_salt_pepper_noise(
                                (image * 255).astype(np.uint8), amount=noise_param
                            )

                        # Apply each denoising method
                        for denoise_config in DENOISING_METHODS:
                            method_name = denoise_config["method"]

                            # Apply denoising
                            if method_name == "median_filter":
                                denoised_image = self.denoising_strategies.median_filter(noisy_image, kernel_size=5)
                            elif method_name == "bilateral_filter":
                                denoised_image = self.denoising_strategies.bilateral_filter(noisy_image, d=9,
                                                                                            sigma_color=75,
                                                                                            sigma_space=75)
                            elif method_name == "non_local_means":
                                denoised_image = self.denoising_strategies.non_local_means_denoising(noisy_image, h=10,
                                                                                                     patch_size=7)

                            denoised_image_float = denoised_image.astype(np.float32) / 255.0

                            variant_name = f"{image_name}_{noise_name}_{method_name}"
                            self._process_and_save_to_folder(denoised_image_float, variant_name, denoised_dir,
                                                             subfolder_name, "denoised")

            except Exception as e:
                self.logger.error(f"    Failed to process {image_name}: {e}")
                continue

    def finalize_results(self):
        """Generate comprehensive final reports"""
        if not self.all_metrics:
            self.logger.warning("No metrics were generated.")
            return

        # Create comprehensive CSV report
        df = pd.DataFrame(self.all_metrics)
        csv_path = self.results_dir / "full_experiment_metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"📊 Saved comprehensive metrics to: {csv_path}")

        # Log summary statistics
        self.logger.info("📈 EXPERIMENT SUMMARY:")
        self.logger.info(f"   Total images processed: {len(df)}")
        self.logger.info(f"   Average Betti 0: {df['betti_0'].mean():.2f}")
        self.logger.info(f"   Average Betti 1: {df['betti_1'].mean():.2f}")


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
