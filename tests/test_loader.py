import os
import tempfile
import numpy as np
import cv2
import yaml
import pytest

from src.io import loader


def test_load_config_and_list_images(tmp_path):
    # Create fake config YAML
    config_path = tmp_path / "config.yaml"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create a fake image
    fake_img = np.ones((10, 10), dtype=np.uint8) * 255
    img_file = img_dir / "test.png"
    cv2.imwrite(str(img_file), fake_img)

    # Write config
    with open(config_path, "w") as f:
        yaml.dump({
            "path": str(img_dir),
            "file_extension": ".png",
            "color_mode": "grayscale"
        }, f)

    # Run loader functions
    cfg = loader.load_config(config_path)
    img_list = loader.list_images(cfg)
    imgs, paths = loader.load_dataset(config_path)

    # Assertions
    assert isinstance(cfg, dict)
    assert len(img_list) == 1
    assert paths[0].endswith("test.png")
    assert imgs[0].shape == (10, 10)
    assert imgs[0].dtype == np.uint8

def test_load_dataset_sample_yaml():
    cfg_path = "config/datasets/sample_dataset.yaml"  # <— uses tiny test dataset
    images, paths = loader.load_dataset(cfg_path)
    assert len(images) >= 3
    assert all(img.ndim == 2 for img in images)  # grayscale
    assert all(p.lower().endswith(".png") for p in paths)


