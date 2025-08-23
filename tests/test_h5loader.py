import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add 'preprocessing' folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))

from preprocessing.load_h5 import load_h5_file

# Base input folders
datasets = {
    "mouse": "data/raw_data/nucmm/nucmm/Mouse (NucMM-M)/Image/train",
    "zebrafish": "data/raw_data/nucmm/nucmm/Zebrafish (NucMM-Z)/Image/train"
}

# Base output folders
npy_output_base = "preprocessed_npy"
png_output_base = "png_images"

# Loop through each dataset
for name, folder_path in datasets.items():
    npy_output_dir = os.path.join(npy_output_base, name)
    png_output_dir = os.path.join(png_output_base, name)

    os.makedirs(npy_output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    h5_files = [f for f in os.listdir(folder_path) if f.endswith(".h5")]

    for file_name in h5_files:
        full_path = os.path.join(folder_path, file_name)

        try:
            image = load_h5_file(full_path)

            # Save .npy
            npy_name = os.path.splitext(file_name)[0] + ".npy"
            np.save(os.path.join(npy_output_dir, npy_name), image)

            # Get 2D slice for .png
            if image.ndim == 3:
                image_to_save = image[image.shape[0] // 2]
            else:
                image_to_save = image

            # Save .png
            png_name = os.path.splitext(file_name)[0] + ".png"
            plt.imsave(os.path.join(png_output_dir, png_name), image_to_save, cmap='gray')

            print(f"[{name.upper()}] Saved: {npy_name}, {png_name}")

        except Exception as e:
            print(f"[{name.upper()}] Error processing {file_name}: {e}")
