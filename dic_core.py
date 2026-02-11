from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DICParams:
    subset_size: int = 31
    step: int = 10
    search_radius: int = 20
    min_score: float = 0.5
    use_subpixel: bool = True

    def validate(self) -> None:
        if self.subset_size < 3 or self.subset_size % 2 == 0:
            raise ValueError("subset_size tek ve >=3 olmali.")
        if self.step < 1:
            raise ValueError("step >=1 olmali.")
        if self.search_radius < 1:
            raise ValueError("search_radius >=1 olmali.")


def snap_roi_to_grid(
    roi: Tuple[int, int, int, int],
    subset_size: int,
    step: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x, y, w, h = map(int, roi)
    h_img, w_img = image_shape[:2]
    r = subset_size // 2

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)

    safe_x0 = x0 + r
    safe_y0 = y0 + r
    safe_x1 = x1 - r
    safe_y1 = y1 - r
    if safe_x1 <= safe_x0 or safe_y1 <= safe_y0:
        raise ValueError("ROI subset boyutuna gore cok kucuk.")

    span_x = safe_x1 - safe_x0
    span_y = safe_y1 - safe_y0

    usable_x = (span_x // step) * step
    usable_y = (span_y // step) * step
    if usable_x < step or usable_y < step:
        raise ValueError("ROI, step ve subset kombinasyonu icin yetersiz.")

    safe_x1 = safe_x0 + usable_x
    safe_y1 = safe_y0 + usable_y

    snapped_x0 = safe_x0 - r
    snapped_y0 = safe_y0 - r
    snapped_x1 = safe_x1 + r
    snapped_y1 = safe_y1 + r

    return snapped_x0, snapped_y0, snapped_x1 - snapped_x0, snapped_y1 - snapped_y0


def generate_grid_points(
    roi: Tuple[int, int, int, int],
    subset_size: int,
    step: int,
) -> np.ndarray:
    x, y, w, h = map(int, roi)
    r = subset_size // 2
    x_start = x + r
    y_start = y + r
    x_end = x + w - r
    y_end = y + h - r

    xs = np.arange(x_start, x_end + 1, step)
    ys = np.arange(y_start, y_end + 1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.int32)


def _subpixel_peak_offset(corr_map: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    if (
        peak_x <= 0
        or peak_y <= 0
        or peak_x >= corr_map.shape[1] - 1
        or peak_y >= corr_map.shape[0] - 1
    ):
        return 0.0, 0.0

    def axis_offset(v_minus: float, v0: float, v_plus: float) -> float:
        denom = v_minus - 2.0 * v0 + v_plus
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (v_minus - v_plus) / denom

    dx = axis_offset(corr_map[peak_y, peak_x - 1], corr_map[peak_y, peak_x], corr_map[peak_y, peak_x + 1])
    dy = axis_offset(corr_map[peak_y - 1, peak_x], corr_map[peak_y, peak_x], corr_map[peak_y + 1, peak_x])
    return float(dx), float(dy)


def compute_displacement_pair(
    frame0: np.ndarray,
    frame1: np.ndarray,
    grid_points: np.ndarray,
    params: DICParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params.validate()
    r = params.subset_size // 2

    displacements = np.full((len(grid_points), 2), np.nan, dtype=np.float32)
    scores = np.zeros((len(grid_points),), dtype=np.float32)

    for i, (cx, cy) in enumerate(grid_points):
        x0, y0 = int(cx - r), int(cy - r)
        x1, y1 = x0 + params.subset_size, y0 + params.subset_size
        template = frame0[y0:y1, x0:x1]

        sx0 = max(0, int(cx - r - params.search_radius))
        sy0 = max(0, int(cy - r - params.search_radius))
        sx1 = min(frame1.shape[1], int(cx + r + params.search_radius + 1))
        sy1 = min(frame1.shape[0], int(cy + r + params.search_radius + 1))

        search_img = frame1[sy0:sy1, sx0:sx1]
        if search_img.shape[0] < params.subset_size or search_img.shape[1] < params.subset_size:
            continue

        corr = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)
        peak_x, peak_y = max_loc

        sub_dx = sub_dy = 0.0
        if params.use_subpixel:
            sub_dx, sub_dy = _subpixel_peak_offset(corr, peak_y, peak_x)

        best_x = sx0 + peak_x + r + sub_dx
        best_y = sy0 + peak_y + r + sub_dy

        displacements[i, 0] = best_x - cx
        displacements[i, 1] = best_y - cy
        scores[i] = max_val

    low_conf_mask = scores < params.min_score
    displacements[low_conf_mask] = np.nan

    return displacements[:, 0], displacements[:, 1], scores


def compute_strain(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x = np.unique(x)
    grid_y = np.unique(y)
    nx, ny = len(grid_x), len(grid_y)

    uu = u.reshape(ny, nx)
    vv = v.reshape(ny, nx)

    valid_u = np.nan_to_num(uu, nan=0.0)
    valid_v = np.nan_to_num(vv, nan=0.0)
    if sigma > 0:
        valid_u = cv2.GaussianBlur(valid_u.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        valid_v = cv2.GaussianBlur(valid_v.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)

    dy = float(np.mean(np.diff(grid_y))) if ny > 1 else 1.0
    dx = float(np.mean(np.diff(grid_x))) if nx > 1 else 1.0

    du_dy, du_dx = np.gradient(valid_u, dy, dx)
    dv_dy, dv_dx = np.gradient(valid_v, dy, dx)

    exx = du_dx
    eyy = dv_dy
    gxy = du_dy + dv_dx
    return exx.ravel(), eyy.ravel(), gxy.ravel()
