"""
Enhanced TDA Pipeline with Dataset Selection and Detailed Analysis
ENHANCED VERSION: Added comprehensive comparative analysis while preserving original structure
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import seaborn as sns

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io.enhanced_loader import EnhancedDataLoader
from src.tda.cubical import cubical_diagrams
from src.noise.generation import NoiseGenerator
from src.noise.mitigation import DenoisingStrategies
from src.visualization.plotter import TDAVisualizer
from src.utils.logger import TDALogger, log_method_call
from src.tda.adaptive_parameters import AdaptiveParameterSelector
from src.tda.thresholds import auto_min_persistence
from src.analysis.comprehensive_analyzer import ComprehensiveAnalyzer
from src.tda.distances import compute_all_distances
from src.tda.edt import edt_diagrams, compare_filtrations


ONEDRIVE_PATH = r"C:\Users\cheen\OneDrive - The University Of Newcastle\Deriving and Analysing Graphs from Neural Activity\Dataset Analysis\data\raw_data"

DATASETS_TO_RUN = [
      # "MOUSEBIRN",  # 8 images - good for testing
     "synthetic_data",  # 18 images
    #  "defungi",          # Will process H1, H2, H3, H5, H6 automatically
    #  "nucmm",            Will process Mouse, Zebrafish subfolders
     'ReportImages'
]

# CONFIGURATION
RUN_ALL_DATASETS = False# Set to True to process everything
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
    {"method": "median_filter", "name": "median_filter", "window_size": 5},
    {"method": "bilateral_filter", "name": "bilateral_filter", "sigma_color": 75, "sigma_spatial": 75},
    {"method": "non_local_means", "name": "non_local_means", "h": 10},
]

# ADDED: Expected Betti numbers for verification (synthetic images)
EXPECTED_BETTI = {
    'A_disk': {'betti_0': 1, 'betti_1': 0},
    'B_two_disks': {'betti_0': 2, 'betti_1': 0},
    'C_single_ring': {'betti_0': 1, 'betti_1': 1},
    'D_many_rings': {'betti_0': 1, 'betti_1': 4},
    'E_tree_branch': {'betti_0': 1, 'betti_1': 0},
    'synthetic_solid_circle': {'betti_0': 1, 'betti_1': 0},
    'synthetic_annulus': {'betti_0': 1, 'betti_1': 1},
    'synthetic_two_circles': {'betti_0': 2, 'betti_1': 0},
    'synthetic_figure_eight': {'betti_0': 1, 'betti_1': 2}
}
# FILTRATION CONFIGURATION
USE_EDT_FILTRATION = False # Set to True to use EDT instead of intensity
COMPARE_FILTRATIONS = True
COMPUTE_DISTANCES = True # Set to False to skip slow distance calculations
# REPRODUCIBILITY CONFIGURATION
RANDOM_SEED = 42

def _gauss_param_to_variance(p: float, is_uint8: bool = True) -> float:
    """
    If p < 1.0, interpret it as sigma fraction of dynamic range (0..1 or 0..255),
    and convert to variance. Otherwise assume `p` is already a variance.
    """
    if p < 1.0:
        sigma = p * (255.0 if is_uint8 else 1.0)
        return float(sigma * sigma)
    return float(p)



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

        # ADDED: Storage for comparative analysis
        self.three_stage_comparisons = []  # Clean → Noisy → Denoised comparisons
        self.verification_results = []  # Verification for synthetic images
        self.is_synthetic_dataset = any('synthetic' in dataset.lower() for dataset in datasets_to_run)

        # ADDED: Set random seed for reproducibility
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED)
            import random
            random.seed(RANDOM_SEED)
            self.logger.info(f"🎲 Random seed set to: {RANDOM_SEED}")
        else:
            self.logger.info("🎲 Using random seed (non-reproducible)")



    # ADDED: Verification system for synthetic images
    def verify_betti_numbers(self, image_name: str, computed_betti: dict) -> dict:
        """Verify computed Betti numbers against expected values for synthetic images."""

        if image_name not in EXPECTED_BETTI:
            return {"verified": False, "reason": "No expected values available"}

        expected = EXPECTED_BETTI[image_name]

        betti_0_correct = computed_betti['betti_0'] == expected['betti_0']
        betti_1_correct = computed_betti['betti_1'] == expected['betti_1']

        verification_result = {
            "verified": betti_0_correct and betti_1_correct,
            "image_name": image_name,
            "expected_betti_0": expected['betti_0'],
            "expected_betti_1": expected['betti_1'],
            "computed_betti_0": computed_betti['betti_0'],
            "computed_betti_1": computed_betti['betti_1'],
            "betti_0_correct": betti_0_correct,
            "betti_1_correct": betti_1_correct
        }

        self.verification_results.append(verification_result)

        if verification_result["verified"]:
            self.logger.info(f"        ✅ VERIFICATION PASSED for {image_name}")
        else:
            self.logger.warning(f"        ❌ VERIFICATION FAILED for {image_name}")
            self.logger.warning(f"           Expected: β₀={expected['betti_0']}, β₁={expected['betti_1']}")
            self.logger.warning(f"           Computed: β₀={computed_betti['betti_0']}, β₁={computed_betti['betti_1']}")

        return verification_result

    # ADDED: Three-stage comparative analysis
    def perform_three_stage_comparison(self, clean_image: np.ndarray, image_name: str,
                                       noise_config: dict, denoise_config: dict) -> dict:
        """Perform three-stage comparison: Clean → Noisy → Denoised."""

        # Stage 1: Clean image analysis
        clean_results = self.analyze_single_image_stage(clean_image, f"{image_name}_clean", "clean")

        # Stage 2: Add noise and analyze
        if noise_config["type"] == "gaussian":
            noisy_dict = self.noise_generator.add_gaussian_noise(
                (clean_image * 255).astype(np.uint8),
                [_gauss_param_to_variance(noise_config["param"], is_uint8=True)]
            )

        elif noise_config["type"] == "salt_pepper":
            noisy_dict = self.noise_generator.add_salt_pepper_noise(
                (clean_image * 255).astype(np.uint8), [noise_config["param"]]
            )

        noisy_image = list(noisy_dict.values())[0].astype(np.float32) / 255.0
        noisy_results = self.analyze_single_image_stage(noisy_image, f"{image_name}_{noise_config['name']}", "noisy")

        # Stage 3: Apply denoising and analyze
        if denoise_config["method"] == "median_filter":
            denoised_uint8 = self.denoising_strategies.apply_median_filter(
                (noisy_image * 255).astype(np.uint8)
            )
        elif denoise_config["method"] == "bilateral_filter":
            denoised_uint8 = self.denoising_strategies.apply_bilateral_filter(
                (noisy_image * 255).astype(np.uint8)
            )
        elif denoise_config["method"] == "non_local_means":
            denoised_uint8 = self.denoising_strategies.apply_non_local_means(
                (noisy_image * 255).astype(np.uint8)
            )

        denoised_image = denoised_uint8.astype(np.float32) / 255.0
        denoised_results = self.analyze_single_image_stage(
            denoised_image, f"{image_name}_{noise_config['name']}_{denoise_config['name']}", "denoised"
        )

        # Calculate comparative metrics
        comparison = {
            'image_name': image_name,
            'noise_type': noise_config['type'],
            'noise_level': noise_config['param'],
            'denoise_method': denoise_config['method'],

            # Betti number changes
            'clean_betti_0': clean_results['betti_0'],
            'clean_betti_1': clean_results['betti_1'],
            'noisy_betti_0': noisy_results['betti_0'],
            'noisy_betti_1': noisy_results['betti_1'],
            'denoised_betti_0': denoised_results['betti_0'],
            'denoised_betti_1': denoised_results['betti_1'],

            # Impact metrics
            'noise_impact_betti_0': noisy_results['betti_0'] - clean_results['betti_0'],
            'noise_impact_betti_1': noisy_results['betti_1'] - clean_results['betti_1'],
            'noise_impact_total': (noisy_results['betti_0'] + noisy_results['betti_1']) -
                                  (clean_results['betti_0'] + clean_results['betti_1']),

            # Recovery metrics
            'recovery_betti_0': abs(denoised_results['betti_0'] - clean_results['betti_0']),
            'recovery_betti_1': abs(denoised_results['betti_1'] - clean_results['betti_1']),
            'recovery_total': abs((denoised_results['betti_0'] + denoised_results['betti_1']) -
                                  (clean_results['betti_0'] + clean_results['betti_1'])),

            # Relative changes (as requested by supervisor)
            'relative_noise_impact': ((noisy_results['betti_0'] + noisy_results['betti_1']) /
                                      max(clean_results['betti_0'] + clean_results['betti_1'], 1)) - 1,
            'relative_recovery': 1 - (abs((denoised_results['betti_0'] + denoised_results['betti_1']) -
                                          (clean_results['betti_0'] + clean_results['betti_1'])) /
                                      max(abs((noisy_results['betti_0'] + noisy_results['betti_1']) -
                                              (clean_results['betti_0'] + clean_results['betti_1'])), 1)),

            'timestamp': pd.Timestamp.now().isoformat()
        }
        if COMPUTE_DISTANCES:
            self.logger.info(f"        Computing Wasserstein distances for {image_name}")
            try:
                diagram_distances = compute_all_distances(
                    clean_results['persistence_dict'],
                    noisy_results['persistence_dict'],
                    denoised_results['persistence_dict']
                )

                # Log the distances for verification
                self.logger.info(
                    f"        Distance H0 clean->noisy: {diagram_distances.get('wasserstein_clean_noisy_H0', 'N/A')}")
                self.logger.info(
                    f"        Distance H1 clean->noisy: {diagram_distances.get('wasserstein_clean_noisy_H1', 'N/A')}")

                # Add distances to comparison metrics
                comparison.update(diagram_distances)

            except Exception as e:
                self.logger.error(f"        Failed to compute distances: {e}")
                # Add default distance values
                for dim in [0, 1]:
                    comparison[f'wasserstein_clean_noisy_H{dim}'] = 0.0
                    comparison[f'wasserstein_clean_denoised_H{dim}'] = 0.0
                    comparison[f'wasserstein_noisy_denoised_H{dim}'] = 0.0
        else:
            self.logger.info(f"        Skipping distance calculations (COMPUTE_DISTANCES=False)")
            # Add placeholder values
            for dim in [0, 1]:
                comparison[f'wasserstein_clean_noisy_H{dim}'] = 0.0
                comparison[f'wasserstein_clean_denoised_H{dim}'] = 0.0
                comparison[f'wasserstein_noisy_denoised_H{dim}'] = 0.0



        self.three_stage_comparisons.append(comparison)

        # Log comparison results
        self.logger.info(f"      🔄 Three-stage comparison: {image_name}")
        self.logger.info(f"         Clean: β₀={clean_results['betti_0']}, β₁={clean_results['betti_1']}")
        self.logger.info(f"         Noisy: β₀={noisy_results['betti_0']}, β₁={noisy_results['betti_1']} "
                         f"(Δ={comparison['noise_impact_total']:+d})")
        self.logger.info(f"         Denoised: β₀={denoised_results['betti_0']}, β₁={denoised_results['betti_1']} "
                         f"(Recovery: {comparison['relative_recovery']:.2%})")

        return comparison

    # ADDED: Single stage analysis helper
    def analyze_single_image_stage(self, image: np.ndarray, variant_name: str, stage: str) -> dict:
        """Analyze a single image stage and return Betti numbers."""

        img_float = image.astype(np.float32)
        params = self.param_selector.select_optimal_parameters(image)

        # ADD THIS LOGGING TO VERIFY CONFIGURATION
        self.logger.info(f"        Using EDT filtration: {USE_EDT_FILTRATION}")
        # FIXED: Use threshold appropriately for each filtration type
        if USE_EDT_FILTRATION:
            persistence_dict = edt_diagrams(img_float, bin_thresh=params['threshold'])
            filtration_type = "edt"
            self.logger.info(f"        Applied EDT filtration with threshold: {params['threshold']}")
        else:
            # FIXED: DO NOT pre-threshold for intensity filtration
            # Let cubical complex analyze the full intensity landscape
            persistence_dict = cubical_diagrams(img_float, superlevel=params['superlevel'])
            filtration_type = "intensity_raw"
            self.logger.info(f"         Applied intensity filtration (superlevel={params['superlevel']})")

        betti_0 = len(persistence_dict.get("H0", []))
        betti_1 = len(persistence_dict.get("H1", []))

        return {
            'variant_name': variant_name,
            'stage': stage,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'total_features': betti_0 + betti_1,
            'superlevel': params['superlevel'],
            'threshold': params['threshold'],
            'threshold_used': 'edt' if USE_EDT_FILTRATION else ('yes' if 0 < params['threshold'] < 1 else 'no'),
            # ADDED
            'filtration_type': filtration_type,
            'persistence_dict': persistence_dict
        }

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

        # COMPUTE PERSISTENCE DIAGRAMS - FIXED to use EDT configuration
        # Skip if we're doing comparison (comparison function handles both)
        if COMPARE_FILTRATIONS:
            # Just return empty dict, comparison will be done separately
            self.logger.info(f"         Skipping individual filtration (comparison mode enabled)")
            return {"H0": np.array([]), "H1": np.array([])}

        # COMPUTE PERSISTENCE DIAGRAMS - FIXED to use EDT configuration
        self.logger.info(f"         Using EDT filtration: {USE_EDT_FILTRATION}")
        if USE_EDT_FILTRATION:
            persistence_dict = edt_diagrams(img_float, bin_thresh=params['threshold'])
            filtration_type = "edt"
            self.logger.info(f"         Applied EDT filtration with threshold: {params['threshold']}")
        else:
            # FIXED: DO NOT pre-threshold for intensity filtration
            # Let cubical complex analyze the full intensity landscape
            persistence_dict = cubical_diagrams(img_float, superlevel=params['superlevel'])
            filtration_type = "intensity_raw"
            self.logger.info(f"         Applied intensity filtration (superlevel={params['superlevel']})")

        # CALCULATE BETTI NUMBERS
        betti_0 = len(persistence_dict.get("H0", []))
        betti_1 = len(persistence_dict.get("H1", []))
        total_features = betti_0 + betti_1

        self.logger.info(f"      📊 TDA Results:")
        self.logger.info(f"         Total features: {total_features}")
        self.logger.info(f"         Betti 0 (components): {betti_0}")
        self.logger.info(f"         Betti 1 (holes): {betti_1}")

        # ADDED: Verification for synthetic images
        computed_betti = {'betti_0': betti_0, 'betti_1': betti_1}
        if stage == "baseline" and self.is_synthetic_dataset and image_name in EXPECTED_BETTI:
            self.verify_betti_numbers(image_name, computed_betti)

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

        # FIXED: Convert dictionary format to list format for visualizer and calculate Betti numbers
        baseline_diags = []
        betti_numbers = {"betti_0": 0, "betti_1": 0}


        # Perform filtration comparison if enabled
        if COMPARE_FILTRATIONS:
            self.logger.info(f"      🔄 Running filtration comparison for {variant_name}")
            self.perform_filtration_comparison_analysis(image, variant_name)

        for dim in [0, 1]:
            if f"H{dim}" in persistence_dict:
                for interval in persistence_dict[f"H{dim}"]:
                    baseline_diags.append((dim, interval))
                betti_numbers[f"betti_{dim}"] = len(persistence_dict[f"H{dim}"])

        # Create visualizers that save to the same directory
        diagram_viz = TDAVisualizer(save_dir, color_scheme="professional", logger=self.logger)
        barcode_viz = TDAVisualizer(save_dir, color_scheme="professional", logger=self.logger)

        # Save plots with descriptive names
        filtration_method = "edt" if USE_EDT_FILTRATION else "intensity"
        diagram_viz.plot_persistence_diagram(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_{filtration_method}_ph_diagram")
        barcode_viz.plot_persistence_barcode(baseline_diags, f"{subfolder_name}: {variant_name}",
                                             f"{variant_name}_{filtration_method}_ph_barcode")

        # FIXED: Save step-by-step - Now betti_numbers is defined
        try:
            self.comprehensive_analyzer.analyze_image_comprehensive(
                image, self.last_params,persistence_dict, variant_name, save_dir
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
                    "denoising_experiments": RUN_DENOISING_EXPERIMENTS,
                    "synthetic_dataset": self.is_synthetic_dataset,
                    "random_seed": RANDOM_SEED  # ADDED
                }
            )

            self.logger.info(f"🚀 Starting comprehensive analysis of {len(datasets_to_process)} datasets")

            for i, dataset_name in enumerate(datasets_to_process, 1):
                self.logger.info(f"📊 [{i}/{len(datasets_to_process)}] Processing dataset: {dataset_name}")
                self.process_dataset(dataset_name)

            # ADDED: Generate comparative analysis
            self.generate_comparative_analysis()

            self.finalize_results()
            self.logger.end_experiment(experiment_id)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def discover_all_datasets(self):
        """Discover all available datasets in the OneDrive path."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']
        all_datasets = []

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


        selected_images = images

        # Process each image
        for i, image_path in enumerate(selected_images):
            try:
                image_name = Path(image_path).stem
                self.logger.info(f"   🖼️  [{i + 1}/{len(selected_images)}] Processing: {image_name}")

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


                for noise_config in NOISE_EXPERIMENTS:
                    for denoise_config in DENOISING_METHODS:
                        self.perform_three_stage_comparison(
                            image, image_name, noise_config, denoise_config
                        )

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

            # FIXED: Ensure consistent single-channel format for noise generation
            if len(image.shape) == 3:
                # Convert to grayscale if multi-channel
                image_gray = np.mean(image, axis=2)
            else:
                image_gray = image

            # Convert to uint8 for noise generation
            image_uint8 = (image_gray * 255).astype(np.uint8) if image_gray.max() <= 1.0 else image_gray.astype(
                np.uint8)

            # Store noisy images for denoising stage
            noisy_images_store = {}

            for noise_exp in NOISE_EXPERIMENTS:
                try:
                    # FIXED: Generate noise with proper shape handling
                    if noise_exp["type"] == "gaussian":
                        noisy_dict = self.noise_generator.add_gaussian_noise(image_uint8, [noise_exp["param"]])
                        noisy_image = list(noisy_dict.values())[0]
                    elif noise_exp["type"] == "salt_pepper":
                        noisy_dict = self.noise_generator.add_salt_pepper_noise(image_uint8, [noise_exp["param"]])
                        noisy_image = list(noisy_dict.values())[0]

                    # Ensure output is 2D
                    if len(noisy_image.shape) > 2:
                        noisy_image = noisy_image[:, :, 0] if noisy_image.shape[2] == 1 else np.mean(noisy_image,
                                                                                                     axis=2)

                    # Convert back to float
                    noisy_float = noisy_image.astype(np.float32) / 255.0
                    noisy_images_store[noise_exp["name"]] = noisy_image  # Store uint8 for denoising

                    # Process noisy variant
                    noisy_name = f"{image_name}_{noise_exp['name']}"
                    self._process_and_save_to_folder(
                        noisy_float, noisy_name, noised_dir, subfolder_name, f"noised_{noise_exp['name']}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed noise {noise_exp['name']}: {e}")

            # === DENOISED VARIANTS ===
            if RUN_DENOISING_EXPERIMENTS:
                self.logger.info(f"        🧽 Processing denoised variants for {image_name}")

                for noise_name, noisy_uint8 in noisy_images_store.items():
                    for denoising in DENOISING_METHODS:
                        try:
                            # Apply denoising
                            if denoising["method"] == "median_filter":
                                denoised_image = self.denoising_strategies.apply_median_filter(noisy_uint8)
                            elif denoising["method"] == "bilateral_filter":
                                denoised_image = self.denoising_strategies.apply_bilateral_filter(noisy_uint8)
                            elif denoising["method"] == "non_local_means":
                                denoised_image = self.denoising_strategies.apply_non_local_means(noisy_uint8)

                            # Convert to float
                            denoised_float = denoised_image.astype(np.float32) / 255.0

                            # Process denoised variant
                            denoised_name = f"{image_name}_{noise_name}_{denoising['name']}"
                            self._process_and_save_to_folder(
                                denoised_float, denoised_name, denoised_dir,
                                subfolder_name, f"denoised_{noise_name}_{denoising['name']}"
                            )

                        except Exception as e:
                            self.logger.error(f"Failed {denoising['method']} on {noise_name}: {e}")

    # ADDED: Generate comprehensive comparative analysis
    def generate_comparative_analysis(self):


        self.logger.info("📈 Generating comprehensive comparative analysis...")

        # Analyze noise impact patterns
        if self.three_stage_comparisons:
            df = pd.DataFrame(self.three_stage_comparisons)

            # Noise impact analysis
            self.logger.info("🔊 NOISE IMPACT ANALYSIS:")
            noise_impact = df.groupby(['noise_type', 'noise_level'])['noise_impact_total'].agg(['mean', 'std'])
            for (noise_type, level), stats in noise_impact.iterrows():
                self.logger.info(f"   {noise_type} {level}: {stats['mean']:.1f} ± {stats['std']:.1f} features")

            # Denoising effectiveness analysis
            self.logger.info("🧽 DENOISING EFFECTIVENESS ANALYSIS:")
            denoise_effectiveness = df.groupby('denoise_method')['relative_recovery'].agg(['mean', 'std'])
            for method, stats in denoise_effectiveness.iterrows():
                self.logger.info(f"   {method}: {stats['mean']:.2%} ± {stats['std']:.2%} recovery")

            # Save comparative analysis
            comparison_path = self.results_dir / "three_stage_comparisons.csv"
            df.to_csv(comparison_path, index=False)
            self.logger.info(f"📊 Three-stage comparisons saved to: {comparison_path}")

            # Generate visualization
            self.generate_comparative_visualization(df)

    # ADDED: Generate comparative visualization
    def generate_comparative_visualization(self, df: pd.DataFrame):
        """Generate comparative visualization plots."""

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('TDA Comparative Analysis: Clean → Noisy → Denoised', fontsize=14, fontweight='bold')

            # 1. Noise Impact by Type
            ax1 = axes[0, 0]
            noise_impact = df.groupby('noise_type')['noise_impact_total'].mean()
            bars1 = ax1.bar(noise_impact.index, noise_impact.values, alpha=0.7, color=['skyblue', 'lightcoral'])
            ax1.set_title('Average Noise Impact by Type')
            ax1.set_ylabel('Feature Change')
            ax1.set_xlabel('Noise Type')

            # 2. Recovery Effectiveness by Method
            ax2 = axes[0, 1]
            recovery_eff = df.groupby('denoise_method')['relative_recovery'].mean()
            bars2 = ax2.bar(recovery_eff.index, recovery_eff.values, alpha=0.7,
                            color=['lightgreen', 'orange', 'purple'])
            ax2.set_title('Recovery Effectiveness by Method')
            ax2.set_ylabel('Relative Recovery')
            ax2.set_xlabel('Denoising Method')
            ax2.tick_params(axis='x', rotation=45)

            # 3. Noise Level vs Impact
            ax3 = axes[1, 0]
            for noise_type in df['noise_type'].unique():
                type_data = df[df['noise_type'] == noise_type]
                ax3.scatter(type_data['noise_level'], type_data['noise_impact_total'],
                            label=noise_type, alpha=0.6, s=50)
            ax3.set_xlabel('Noise Level')
            ax3.set_ylabel('Total Feature Impact')
            ax3.set_title('Noise Level vs Impact')
            ax3.legend()

            # 4. Recovery vs Original Impact
            ax4 = axes[1, 1]
            scatter = ax4.scatter(df['noise_impact_total'], df['relative_recovery'],
                                  c=df['noise_level'], cmap='viridis', alpha=0.6, s=50)
            ax4.set_xlabel('Original Noise Impact')
            ax4.set_ylabel('Relative Recovery')
            ax4.set_title('Recovery vs Original Impact')
            plt.colorbar(scatter, ax=ax4, label='Noise Level')

            plt.tight_layout()

            # Save visualization
            viz_path = self.results_dir /"plots"/ "comparative_analysis_visualization.png"
            viz_path.parent.mkdir(exist_ok=True)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"📊 Comparative visualization saved to: {viz_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")

    def _generate_analysis_summary_md(self, output_path):
        """Generate human-readable analysis summary in Markdown format."""

        if not self.three_stage_comparisons:
            return

        df = pd.DataFrame(self.three_stage_comparisons)

        summary_content = f"""# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: {USE_EDT_FILTRATION}
    - **Compare Filtrations**: {COMPARE_FILTRATIONS}
    - **Random Seed**: {RANDOM_SEED}
    - **Datasets Processed**: {', '.join(self.datasets_to_run)}

    ## Results Overview
    - **Total Images Analyzed**: {len(df['image_name'].unique())}
    - **Total Comparisons**: {len(df)}
    - **Average Noise Impact**: {df['noise_impact_total'].mean():.2f} features
    - **Average Recovery Rate**: {df['relative_recovery'].mean():.2f}%

    ## Noise Type Analysis
    {df.groupby('noise_type')['noise_impact_total'].agg(['mean', 'std']).to_string()}

    ## Denoising Method Effectiveness
    {df.groupby('denoise_method')['relative_recovery'].agg(['mean', 'std']).to_string()}

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: {df['wasserstein_clean_noisy_H0'].replace([np.inf, -np.inf], np.nan).mean():.4f}
    - **Average Wasserstein H1 (Clean->Noisy)**: {df['wasserstein_clean_noisy_H1'].replace([np.inf, -np.inf], np.nan).mean():.4f}
    """

        with open(output_path, 'w') as f:
            f.write(summary_content)

    def _generate_method_comparison_matrix(self, output_path):
        """Generate method comparison matrix CSV."""

        if not self.three_stage_comparisons:
            return

        df = pd.DataFrame(self.three_stage_comparisons)

        # Create comparison matrix
        comparison_matrix = df.pivot_table(
            values='relative_recovery',
            index=['noise_type', 'noise_level'],
            columns='denoise_method',
            aggfunc='mean'
        )

        comparison_matrix.to_csv(output_path)

    def _generate_enhanced_visualization_with_context(self):
        """Generate visualization with proper dataset context and labels."""

        # Modify your existing generate_comparative_visualization() method
        # to include dataset information and configuration details

        viz_path = self.results_dir / "plots" / "comprehensive_comparative_analysis_with_context.png"
        viz_path.parent.mkdir(exist_ok=True)

        # Add dataset and configuration info to the plot title
        title = f"TDA Comparative Analysis: {', '.join(self.datasets_to_run)}\n"
        title += f"EDT: {USE_EDT_FILTRATION}, Compare: {COMPARE_FILTRATIONS}, Seed: {RANDOM_SEED}"

        # Use this title in your plotting function
        # ... rest of visualization code with enhanced context

    def perform_filtration_comparison_analysis(self, image: np.ndarray, image_name: str):
        """Perform dedicated comparison between intensity and EDT filtrations."""

        if not COMPARE_FILTRATIONS:
            return

        self.logger.info(f"      Performing filtration comparison for {image_name}")

        img_float = image.astype(np.float32)
        params = self.param_selector.select_optimal_parameters(image)

        # Prepare parameter dictionaries
        intensity_params = {
            'superlevel': params['superlevel'],
            'coeff': 2
        }

        edt_params = {
            'bin_thresh': params['threshold'],
            'invert': False,
            'coeff': 2
        }

        # Run comparison using the actual method signature
        comparison_results = compare_filtrations(
            img_float,
            intensity_params=intensity_params,
            edt_params=edt_params,
            save_path = str(self.results_dir / "filtration_comparison_results.csv"),
            image_name = image_name
        )

        # Save comparison results
        comparison_data = {
            'image_name': image_name,
            'intensity_betti_0': comparison_results['intensity_betti_0'],
            'intensity_betti_1': comparison_results['intensity_betti_1'],
            'edt_betti_0': comparison_results['edt_betti_0'],
            'edt_betti_1': comparison_results['edt_betti_1'],
            'method_difference_total': comparison_results.get('method_difference_total', 0),
            'better_method': comparison_results.get('better_method', 'unknown'),
            'threshold': params['threshold'],
            'superlevel': params['superlevel']
        }

        # Store for later analysis
        if not hasattr(self, 'filtration_comparisons'):
            self.filtration_comparisons = []
        self.filtration_comparisons.append(comparison_data)

        self.logger.info(
            f"        Intensity: β₀={comparison_results['intensity_betti_0']}, β₁={comparison_results['intensity_betti_1']}")
        self.logger.info(f"        EDT: β₀={comparison_results['edt_betti_0']}, β₁={comparison_results['edt_betti_1']}")
        self.logger.info(f"        Better method: {comparison_results.get('better_method', 'unknown')}")

    def finalize_results(self):
        """Generate comprehensive final reports with verification results"""
        if not self.all_metrics:
            self.logger.warning("No metrics were generated.")
            return

        # Create comprehensive CSV report
        df = pd.DataFrame(self.all_metrics)
        csv_path = self.results_dir / "full_experiment_metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"📊 Saved comprehensive metrics to: {csv_path}")

        # ADDED: Create verification report if we have verification results
        if self.verification_results and self.is_synthetic_dataset:
            verification_df = pd.DataFrame(self.verification_results)
            verification_csv = self.results_dir / "verification_results.csv"
            verification_df.to_csv(verification_csv, index=False)

            # Log verification summary
            total_verified = len(self.verification_results)
            passed_verified = sum(1 for r in self.verification_results if r['verified'])



            self.logger.info("🧪 VERIFICATION SUMMARY:")
            self.logger.info(f"   Total verified images: {total_verified}")
            self.logger.info(f"   Verification passed: {passed_verified}")
            self.logger.info(f"   Verification failed: {total_verified - passed_verified}")
            if total_verified > 0:
                self.logger.info(f"   Success rate: {passed_verified / total_verified * 100:.1f}%")

        elif not self.is_synthetic_dataset:
            self.logger.info("ℹ️  Verification skipped for real datasets (no ground truth available)")
        # Log summary statistics
        self.logger.info("📈 EXPERIMENT SUMMARY:")
        self.logger.info(f"   Total images processed: {len(df)}")
        self.logger.info(f"   Average Betti 0: {df['betti_0'].mean():.2f}")
        self.logger.info(f"   Average Betti 1: {df['betti_1'].mean():.2f}")

        # ADDED: Noise robustness analysis
        baseline_df = df[df['stage'] == 'baseline']
        if len(baseline_df) > 0:
            self.logger.info("🔊 NOISE ROBUSTNESS ANALYSIS:")
            self.logger.info(f"   Baseline avg features: {baseline_df['total_features'].mean():.1f}")

            noised_df = df[df['variant'].str.contains('gaussian|salt_pepper', na=False)]
            if len(noised_df) > 0:
                self.logger.info(f"   Noised avg features: {noised_df['total_features'].mean():.1f}")
                if baseline_df['total_features'].mean() > 0:
                    feature_increase = (noised_df['total_features'].mean() / baseline_df[
                        'total_features'].mean() - 1) * 100
                    self.logger.info(f"   Feature increase due to noise: {feature_increase:.1f}%")

        # Generate verification results (even for real datasets)
        self.logger.info("  Generating verification and summary reports...")

        # Create comprehensive analysis summary
        summary_md_path = self.results_dir / "comprehensive_analysis_summary.md"
        self._generate_analysis_summary_md(summary_md_path)

        # Create method comparison matrix
        comparison_matrix_path = self.results_dir / "method_comparison_matrix.csv"
        self._generate_method_comparison_matrix(comparison_matrix_path)

        # Enhanced visualization with dataset context
        self._generate_enhanced_visualization_with_context()

        self.logger.info(f"  Generated comprehensive analysis summary: {summary_md_path}")
        self.logger.info(f"  Generated method comparison matrix: {comparison_matrix_path}")

        if hasattr(self, 'filtration_comparisons') and self.filtration_comparisons:
            comparison_df = pd.DataFrame(self.filtration_comparisons)
            comparison_csv = self.results_dir / "filtration_comparison_results.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            self.logger.info(f"  Filtration comparison results saved to: {comparison_csv}")

            # Log summary
            better_method_counts = comparison_df['better_method'].value_counts()
            self.logger.info(f"  Filtration method effectiveness: {better_method_counts.to_dict()}")

if __name__ == "__main__":
    print("🚀 TDA Pipeline with Comprehensive Noise Analysis")
    print(f"📁 Data source: {ONEDRIVE_PATH}")
    print(f"🎯 Selected datasets: {DATASETS_TO_RUN}")
    print(f"🔧 Run all datasets: {RUN_ALL_DATASETS}")
    print(f"🔊 Noise experiments: {RUN_NOISE_EXPERIMENTS}")
    print(f"🧹 Denoising experiments: {RUN_DENOISING_EXPERIMENTS}")
    print(f"⚠️  Excluding datasets: {EXCLUDE_DATASETS}")
    print("✨ ENHANCED: Added comprehensive comparative analysis")

    pipeline = TDAExperimentPipeline(
        results_dir=str(PROJECT_ROOT / "TDA_Analysis_Results_syntehtic_data"),
        onedrive_path=ONEDRIVE_PATH,
        datasets_to_run=DATASETS_TO_RUN
    )

    pipeline.run()
