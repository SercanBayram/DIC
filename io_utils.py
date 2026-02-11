from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import csv


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FrameSequence:
    paths: List[Path]
    frames: List[np.ndarray]
    delta_t: float


def _natural_sort_key(path: Path) -> List[object]:
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    key: List[object] = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p)
    return key


def load_images(folder: Path, delta_t: float = 2.0) -> FrameSequence:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Goruntu klasoru bulunamadi: {folder}")

    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort(key=_natural_sort_key)
    if len(paths) < 2:
        raise ValueError("DIC icin en az iki goruntu gerekli.")

    frames: List[np.ndarray] = []
    for p in paths:
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Goruntu okunamadi: {p}")
        frames.append(image)

    return FrameSequence(paths=paths, frames=frames, delta_t=delta_t)


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_pair_csv(
    output_path: Path,
    pair_index: int,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    score: np.ndarray,
    delta_t: float,
) -> Path:
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "x", "y", "u", "v", "vx", "vy", "score"])
        for xi, yi, ui, vi, si in zip(x, y, u, v, score):
            writer.writerow([pair_index, float(xi), float(yi), float(ui), float(vi), float(ui / delta_t), float(vi / delta_t), float(si)])
    return output_path


def save_npy(output_path: Path, array: np.ndarray) -> Path:
    output_path = Path(output_path)
    np.save(output_path, array)
    return output_path
