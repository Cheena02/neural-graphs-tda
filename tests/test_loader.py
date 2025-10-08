# test_loader.py - JUST for testing what datasets are found
import sys
sys.path.append('./src')

from data_io.data_loader import UniversalDataLoader

loader = UniversalDataLoader()
onedrive_path = r"C:\Users\cheen\OneDrive - The University Of Newcastle\Deriving and Analysing Graphs from Neural Activity\Dataset Analysis\data\raw_data"

# JUST show what datasets exist
datasets = loader.discover_all_datasets(onedrive_path)
print(f"Found {len(datasets)} datasets:")
for folder, images in datasets.items():
    print(f"  📁 {folder}: {len(images)} images")

# That's it! No processing.

