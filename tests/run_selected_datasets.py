import sys
sys.path.append('./src')

from data_io.data_loader import UniversalDataLoader

loader = UniversalDataLoader()
onedrive_path = r"C:\Users\cheen\OneDrive - The University Of Newcastle\Deriving and Analysing Graphs from Neural Activity\Dataset Analysis\data\raw_data"

# Choose which datasets to run
datasets_to_run = [
    #"synthetic_data",    # 18 images - good for testing
    #"sample_images",     # 3 images - very quick test
     "MOUSEBIRN",       # 8 images
    # "mouse",           # 8 images
    # "spider_web",      # 431 images
]

print("🎯 Selected datasets:")
for name in datasets_to_run:
    print(f"  - {name}")

# Process only selected datasets
loader.process_selected_datasets(onedrive_path, datasets_to_run, "my_selected_results")
