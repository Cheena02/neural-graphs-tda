# src/io/loader.py
from __future__ import annotations

import os
import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Dict, Any
import imageio.v3 as iio
from PIL import Image, ImageSequence
import cv2
import yaml

def _pil_to_gray_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.uint8)

def _load_pic(path: str, mode: str = "single", slice_sel: str | int = "middle") -> np.ndarray:
    try:
        im = Image.open(path)
        # multi-frame PIC? (some PICs are stacks)
        frames = [frame.copy() for frame in ImageSequence.Iterator(im)] or [im]
        if len(frames) == 1:
            return _pil_to_gray_np(frames[0])               # (H,W) uint8
        # stack: (Z,Y,X), choose slice or return whole stack later if you add 3D PH
        if mode == "single":
            z = len(frames)//2 if slice_sel == "middle" else int(slice_sel)
            z = max(0, min(z, len(frames)-1))
            return _pil_to_gray_np(frames[z])
        stack = np.stack([_pil_to_gray_np(f) for f in frames], axis=0)  # (Z,Y,X)
        return stack
    except Exception:
        # Fallback via imageio
        try:
            frames = iio.mimread(path)
            if not frames:
                arr = iio.imread(path)
                return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if len(frames) == 1:
                arr = np.asarray(frames[0])
                return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            # multi-frame
            stack = np.stack([f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)
            return stack if mode != "single" else stack[stack.shape[0]//2]
        except Exception as e:
            raise ValueError(f"Could not read .pic file {path}: {e}")


def _repo_root() -> Path:
    # .../Dataset Analysis/src/io/loader.py -> go up to project root
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(config_path: str | os.PathLike) -> Path:
    p = Path(config_path)
    candidates = [
        p,
        Path.cwd() / p,
        _repo_root() / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Config not found: {config_path}\nTried:\n  - {tried}")


# NEW: expand helpers ----------------------------------------------------------
def expand_path(s: str) -> Path:
    """
    Expand ${ENV} / %ENV%, ~ and return absolute Path.
    """
    return Path(os.path.expanduser(os.path.expandvars(s))).resolve()


def deep_expand(x: Any) -> Any:
    """
    Recursively expand strings that contain env vars or ~ inside a dict/list tree.
    Leaves other strings alone (fast path).
    """
    if isinstance(x, dict):
        return {k: deep_expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [deep_expand(v) for v in x]
    if isinstance(x, str):
        # Only expand strings that look like paths with env vars or ~
        if "${" in x or "%" in x or x.startswith("~"):
            return str(expand_path(x))
        return x
    return x
# -----------------------------------------------------------------------------


def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    """Load a dataset YAML configuration file (robust path resolution + env expansion)."""
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Normalize keys and defaults
    if "extensions" in cfg and isinstance(cfg["extensions"], list):
        exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in cfg["extensions"]]
    else:
        ext = cfg.get("file_extension", ".png")
        if isinstance(ext, str):
            exts = [ext.lower() if ext.startswith(".") else f".{ext.lower()}"]
        else:
            exts = [".png"]
    cfg["extensions"] = exts
    cfg["color_mode"] = cfg.get("color_mode", "grayscale")
    cfg["recursive"] = bool(cfg.get("recursive", False))

    # Expand env vars (DATA_PATH / RESULTS_PATH) and ~ across the whole config
    cfg = deep_expand(cfg)

    # Resolve base path (after expansion)
    if "path" not in cfg:
        raise KeyError("Config missing required key 'path'.")
    base = Path(cfg["path"])

    # Try as-is, then relative to CWD and repo root
    base_candidates = [
        base,
        Path.cwd() / base,
        _repo_root() / base,
    ]
    for c in base_candidates:
        if c.exists():
            cfg["path"] = str(c.resolve())
            break
    else:
        tried = "\n  - ".join(str(x) for x in base_candidates)
        raise FileNotFoundError(f"Dataset path not found for 'path' in config:\n  - {tried}")

    return cfg


def list_images(dataset_config: Dict[str, Any]) -> List[str]:
    """Return absolute paths to all images in a dataset."""
    base_path = Path(dataset_config["path"])
    exts = tuple(dataset_config["extensions"])
    recursive = bool(dataset_config.get("recursive", False))

    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset folder does not exist: {base_path}")

    if recursive:
        files = [p for p in base_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in base_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    files = sorted(p.resolve().as_posix() for p in files)
    if not files:
        raise FileNotFoundError(
            f"No images found in {base_path} with extensions {exts} "
            f"(recursive={recursive})."
        )
    return files


def _load_h5_volume(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        # try common dataset keys first
        for key in ("image", "raw", "data", "img", "volume"):
            if key in f and isinstance(f[key], h5py.Dataset):
                dset = f[key]; break
        else:
            # fallback: largest numeric dataset
            best, best_sz = None, -1
            for k, v in f.items():
                if isinstance(v, h5py.Dataset) and np.issubdtype(v.dtype, np.number):
                    sz = int(np.prod(v.shape))
                    if sz > best_sz:
                        best, best_sz = v, sz
            if best is None:
                raise ValueError(f"No numeric dataset found in {h5_path}")
            dset = best

        vol = dset[()]  # load to RAM

    vol = np.asarray(vol)
    vol = np.squeeze(vol)
    if vol.ndim not in (2, 3):
        raise ValueError(f"Unsupported H5 shape {vol.shape} in {h5_path}")

    # ensure (Z, Y, X) if 3D: move the smallest axis to Z (common for microscopy)
    if vol.ndim == 3:
        z_axis = int(np.argmin(vol.shape))
        if z_axis != 0:
            vol = np.moveaxis(vol, z_axis, 0)

    if np.issubdtype(vol.dtype, np.floating):
        vol = np.nan_to_num(vol, copy=False)
    return vol

def load_image(image_path: str, color_mode: str = "grayscale"):

    p = Path(image_path)
    if p.suffix.lower() == ".h5":
        return _load_h5_volume(image_path)
    # raster branch same as before:
    if color_mode == "grayscale":
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def load_dataset(config_path: str) -> Tuple[list, list]:
    """
    High-level loader:
    Reads the config, lists all images, loads them into memory.
    Returns (images, paths)
    """
    cfg = load_config(config_path)
    image_paths = list_images(cfg)
    images = [load_image(p, cfg.get("color_mode", "grayscale")) for p in image_paths]
    return images, image_paths
