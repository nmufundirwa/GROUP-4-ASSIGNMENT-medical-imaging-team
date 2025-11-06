"""
Utilities for loading and preprocessing images.
"""
from pathlib import Path
import cv2
import numpy as np

def list_images(input_dir: Path) -> list[Path]:
    """Return a sorted list of image files in a directory."""
    return sorted(input_dir.glob("*.jpeg"))

def read_image(path: Path) -> np.ndarray:
    """Read an image and convert it to a float32 NumPy array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Unable to read image at {path}")
    return img.astype(np.float32) / 255.0

def preprocess_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize and pad an image to a target size."""
    h, w = img.shape[:2]
    th, tw = target_size
    if h / w != th / tw:
        # Pad to square
        max_dim = max(h, w)
        ph = (max_dim - h) // 2
        pw = (max_dim - w) // 2
        padded = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = img
   
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    return resized
