from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from dic_core import DICParams, compute_displacement_pair, compute_strain, generate_grid_points, snap_roi_to_grid
from io_utils import ensure_output_dir, load_images, save_npy, save_pair_csv
from viz import save_quiver_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basit DIC prototipi (MVP)")
    parser.add_argument("--input", type=Path, required=True, help="Ardisik goruntu klasoru")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Cikti klasoru")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Ayar dosyasi")
    parser.add_argument("--roi", type=int, nargs=4, help="ROI x y w h (interaktif secim yerine)")
    parser.add_argument("--no-interactive", action="store_true", help="Interaktif ROI secimini kapat")
    parser.add_argument("--max-pairs", type=int, default=None, help="Islenecek maksimum frame cifti")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def select_roi_interactive(image: np.ndarray) -> Tuple[int, int, int, int]:
    roi = cv2.selectROI("ROI secimi", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("Gecerli ROI secilmedi.")
    return tuple(map(int, roi))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    delta_t = float(cfg.get("delta_t", 2.0))
    params = DICParams(
        subset_size=int(cfg.get("subset_size", 31)),
        step=int(cfg.get("step", 10)),
        search_radius=int(cfg.get("search_radius", 20)),
        min_score=float(cfg.get("min_score", 0.5)),
        use_subpixel=bool(cfg.get("use_subpixel", True)),
    )

    seq = load_images(args.input, delta_t=delta_t)
    output_dir = ensure_output_dir(args.output)

    first = seq.frames[0]
    if args.roi is not None:
        roi = tuple(args.roi)
    elif args.no_interactive:
        h, w = first.shape[:2]
        roi = (w // 5, h // 5, (3 * w) // 5, (3 * h) // 5)
    else:
        roi = select_roi_interactive(first)

    snapped_roi = snap_roi_to_grid(roi, params.subset_size, params.step, first.shape)
    grid_points = generate_grid_points(snapped_roi, params.subset_size, params.step)
    print(f"ROI={roi} -> snapped ROI={snapped_roi}, grid noktasi={len(grid_points)}")

    num_pairs = len(seq.frames) - 1
    if args.max_pairs is not None:
        num_pairs = min(num_pairs, args.max_pairs)

    for i in range(num_pairs):
        f0, f1 = seq.frames[i], seq.frames[i + 1]
        u, v, score = compute_displacement_pair(f0, f1, grid_points, params)
        x = grid_points[:, 0].astype(np.float32)
        y = grid_points[:, 1].astype(np.float32)

        csv_path = output_dir / f"pair_{i:04d}.csv"
        save_pair_csv(csv_path, i, x, y, u, v, score, seq.delta_t)

        exx, eyy, gxy = compute_strain(x, y, u, v, sigma=float(cfg.get("strain_smoothing_sigma", 1.0)))
        save_npy(output_dir / f"pair_{i:04d}_strain_exx.npy", exx)
        save_npy(output_dir / f"pair_{i:04d}_strain_eyy.npy", eyy)
        save_npy(output_dir / f"pair_{i:04d}_strain_gxy.npy", gxy)

        plot_path = output_dir / f"pair_{i:04d}_quiver.png"
        save_quiver_plot(
            image=f0,
            x=x,
            y=y,
            u=u,
            v=v,
            score=score,
            output_path=plot_path,
            title=f"Pair {i}->{i+1} | dt={seq.delta_t}s",
        )
        print(f"Kaydedildi: {csv_path.name}, {plot_path.name}")


if __name__ == "__main__":
    main()
