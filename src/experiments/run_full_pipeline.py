#!/usr/bin/env python3
"""
Main experiment pipeline for TDA noise robustness analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from src.io.enhanced_loader import EnhancedDataLoader
from src.tda.cubical import cubical_diagrams
from src.noise.generation import NoiseGenerator
from src.noise.mitigation import DenoisingStrategies
from src.visualization.plotter import TDAVisualizer
from src.utils.logger import TDALogger, log_method_call

class TDAExperimentPipeline:
    """
    Orchestrates a full TDA experiment, from data loading to analysis and reporting.
    """

    def __init__(self, config_path: str, results_dir: str):
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TDALogger(name="TDA_Pipeline", log_dir=self.results_dir / "logs", level="INFO")
        self.loader = EnhancedDataLoader(logger=self.logger)
        self.visualizer = TDAVisualizer(output_dir=self.results_dir / "plots", logger=self.logger)
        self.noise_generator = NoiseGenerator()
        self.denoiser = DenoisingStrategies(logger=self.logger)
        self.all_metrics = []

    @log_method_call
    def run(self):
        try:
            config = self.loader.load_config(self.config_path)
            self.logger.start_experiment(
                experiment_name=f"TDA Analysis: {config['name']}",
                parameters=config
            )

            for image_batch, metadata_batch in self.loader.stream_dataset(config, batch_size=1):
                for image, metadata in zip(image_batch, metadata_batch):
                    self.process_single_image(image, metadata)

            self.finalize_results()
            self.logger.end_experiment(status="completed")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.logger.end_experiment(status="failed")
            self.finalize_results()

    def process_single_image(self, image: np.ndarray, metadata):
        image_name = Path(metadata.filename).stem
        self.logger.info(f"--- Processing image: {image_name} ---")

        # Baseline Analysis
        baseline_diags = self.analyze_and_log(image, image_name, "baseline", "clean")
        self.visualizer.plot_persistence_diagram(baseline_diags, f"Baseline: {image_name}", f"{image_name}_baseline_diag")
        self.visualizer.plot_persistence_barcode(baseline_diags, f"Barcode: {image_name}", f"{image_name}_baseline_barcode")

    def analyze_and_log(self, image: np.ndarray, image_name: str, stage: str, variant: str) -> list:
        # Invert the image for sublevel set filtration of bright features
        img_float = (255 - image).astype(np.float32) / 255.0

        diags_dict = cubical_diagrams(img_float, superlevel=False)
        persistence_list = []
        for dim, diags in diags_dict.items():
            if diags.size > 0:
                for interval in diags:
                    persistence_list.append((int(dim[-1]), tuple(interval)))

        h0_intervals = diags_dict.get("H0", np.array([]))
        h1_intervals = diags_dict.get("H1", np.array([]))
        
        # Correct Betti number calculation
        betti_0 = len(h0_intervals) # Number of H0 intervals is the number of components
        betti_1 = np.sum(np.isinf(h1_intervals[:, 1]) == False) # Number of finite H1 intervals is the number of loops

        h0_lifespans = h0_intervals[:, 1] - h0_intervals[:, 0]
        h1_finite_intervals = h1_intervals[np.isinf(h1_intervals[:, 1]) == False]
        h1_lifespans = h1_finite_intervals[:, 1] - h1_finite_intervals[:, 0] if h1_finite_intervals.size > 0 else np.array([])

        metrics = {
            "image_name": image_name,
            "stage": stage,
            "variant": variant,
            "betti_0": betti_0,
            "betti_1": betti_1,
            "h0_total_persistence": np.sum(h0_lifespans[np.isfinite(h0_lifespans)]),
            "h1_total_persistence": np.sum(h1_lifespans),
        }
        self.all_metrics.append(metrics)
        self.logger.log_tda_results(**metrics)
        return persistence_list

    def finalize_results(self):
        if not self.all_metrics:
            self.logger.warning("No metrics were generated.")
            return

        df = pd.DataFrame(self.all_metrics)
        csv_path = self.results_dir / "full_experiment_metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved all metrics to: {csv_path}")

if __name__ == "__main__":
    pipeline = TDAExperimentPipeline(
        config_path="datasets/synthetic_verification.yaml",
        results_dir="experiment_results/synthetic_verification_run"
    )
    pipeline.run()

