# src/io/loader.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import yaml


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


def load_image(image_path: str, color_mode: str = "grayscale"):
    """Load an image in grayscale or color as a numpy array."""
    if color_mode == "grayscale":
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cvIMREAD_COLOR)  # keep your original if correct
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
