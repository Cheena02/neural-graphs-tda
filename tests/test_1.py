# Create a test script: test_my_pipeline.py
import sys
from pathlib import Path
sys.path.append('./src')

from visualization.plotter import TDAVisualizer
from tda.adaptive_parameters import AdaptiveParameterSelector
from tda.cubical import cubical_diagrams
import numpy as np

# Test the components
print("Testing enhanced pipeline...")

# 1. Test parameter selector
selector = AdaptiveParameterSelector()
test_image = np.random.rand(100, 100)
params = selector.select_optimal_parameters(test_image)
print(f"✅ Parameters: {params}")

# 2. Test visualizer
viz = TDAVisualizer("test_plots", color_scheme="professional")
print("✅ Visualizer created")

# 3. Test TDA analysis
tda_results = cubical_diagrams(test_image, superlevel=params['superlevel'])
print(f"✅ TDA analysis: {len(tda_results['H0'])} H0, {len(tda_results['H1'])} H1")

print("🎉 Everything works!")
