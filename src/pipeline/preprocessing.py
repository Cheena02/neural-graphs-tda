from __future__ import annotations
from pathlib import Path
import argparse
import cv2
import numpy as np
import os
import sys

from src.io.loader import load_config, list_images, load_image


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img, np.float32)


def preprocess_for_ph(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal change: return TWO images
      - gray_out: enhanced grayscale in uint8 [0,255] (no threshold)
      - bin_out:  Otsu-binary in uint8 {0,255}
    """
    # Normalize + light denoise + CLAHE
    img01 = normalize01(img)
    img01 = cv2.GaussianBlur(img01, (3, 3), 0.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray01 = clahe.apply((img01 * 255).astype(np.uint8)) / 255.0

    gray_out = (gray01 * 255).astype(np.uint8)

    # Otsu binarization on the enhanced grayscale
    _, bin_out = cv2.threshold(gray_out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray_out, bin_out


def preprocess_dataset(config_yaml: str, out_dir: Path, downsample: int = 768) -> int:
    """Run preprocessing for one dataset; save *_preproc_gray.png and *_preproc_bin.png. Return #files written."""
    cfg = load_config(config_yaml)
    # Split outputs
    gray_dir = out_dir / "gray"
    bin_dir = out_dir / "bin"
    gray_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for p in list_images(cfg):
        im = load_image(p, cfg.get("color_mode", "grayscale"))
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        gray_img, bin_img = preprocess_for_ph(im)

        if downsample:
            h, w = gray_img.shape
            m = max(h, w)
            if m > downsample:
                s = downsample / m
                new_size = (int(w * s), int(h * s))
                gray_img = cv2.resize(gray_img, new_size, interpolation=cv2.INTER_AREA)
                bin_img  = cv2.resize(bin_img,  new_size, interpolation=cv2.INTER_AREA)

        stem = Path(p).stem
        cv2.imwrite(str(gray_dir / f"{stem}_preproc_gray.png"), gray_img)
        cv2.imwrite(str(bin_dir / f"{stem}_preproc_bin.png"),  bin_img)
        written += 2

    print(f"[preprocess] wrote {written} files to\n gray:{gray_dir}\n bin:{bin_dir}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dataset YAML (original images)")
    ap.add_argument("--out_dir", required=True, help="folder to save preprocessed PNGs")
    ap.add_argument("--downsample", type=int, default=768)
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Resolve out_dir: if relative -> anchor at project root; if absolute -> use as-is
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    preprocess_dataset(args.config, out_dir, args.downsample)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if len(sys.argv) == 1:
        # No args → run BOTH datasets by default
        defaults = [
            ("mouse",     "config/datasets/nucmm_mouse.yaml",     PROJECT_ROOT / "my_results" / "preproc" / "nucmm_mouse"),
            ("zebrafish", "config/datasets/nucmm_zebrafish.yaml", PROJECT_ROOT / "my_results" / "preproc" / "nucmm_zebrafish"),
        ]
        total = 0
        for name, cfg, outdir in defaults:
            try:
                print(f"[preprocess] Running {name}: cfg={cfg} → {outdir}")
                total += preprocess_dataset(cfg, outdir, downsample=768)
            except Exception as e:
                print(f"[preprocess][WARN] {name} failed: {e}")
        print(f"[preprocess] DONE. Total files written: {total}")
    else:
        main()
