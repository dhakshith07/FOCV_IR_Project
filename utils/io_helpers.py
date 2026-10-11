"""
utils/io_helpers.py
File I/O, JSON read/write, image loading utilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def read_json(path: Path) -> Any:
    """Read and return data from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """
    Load an image as a BGR numpy array using OpenCV.
    Returns None if the file does not exist.
    """
    try:
        import cv2
        path = Path(path)
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        return img
    except ImportError:
        raise ImportError("opencv-python is required for load_image_bgr()")
