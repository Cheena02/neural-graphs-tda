#!/usr/bin/env python3
"""
Comprehensive TDA analysis orchestrator
"""
from .step_by_step_visualizer import StepByStepVisualizer
from .summary_generator import SummaryGenerator

class ComprehensiveAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger
        self.step_visualizer = StepByStepVisualizer(logger)
        self.summary_generator = SummaryGenerator(logger)
    
    def analyze_image_comprehensive(self, image, params, filename, output_dir):
        """Complete analysis with all visualizations"""
        # Step-by-step visualization
        self.step_visualizer.create_step_by_step_visualization(
            image, params, filename, output_dir
        )
        
        # Return analysis results
        return {
            'step_by_step_created': True,
            'filename': filename
        }
    
    def finalize_dataset_analysis(self, dataset_name, results, output_dir):
        """Generate all summary reports for a dataset"""
        self.summary_generator.generate_dataset_summary(dataset_name, results, output_dir)
    
    def finalize_overall_analysis(self, all_results, output_path):
        """Generate overall summary"""
        self.summary_generator.generate_overall_summary(all_results, output_path)
