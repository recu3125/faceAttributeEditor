#!/usr/bin/env python
"""
Validate landmark->attribute model on validation images.

Uses raw_validation_data.csv trial-level responses and computes Pearson
correlation per attribute (condition).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_EYE_LOOP = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_LOOP = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reverse model.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("omi") / "corr" / "landmark_to_attr_ridge.npz",
        help="Path to landmark_to_attr_ridge.npz",
    )
    parser.add_argument(
        "--landmarks_jsonl",
        type=Path,
        default=Path("omi") / "landmarks" / "validation_face_landmarks.jsonl",
        help="Path to validation landmarks JSONL.",
    )
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=Path("omi") / "raw_validation_data.csv",
        help="Path to raw_validation_data.csv",
    )
    return parser.parse_args()


def compute_center(points: np.ndarray, indices: List[int]) -> np.ndarray:
    return points[indices].mean(axis=0)


def normalize_face(vec: np.ndarray) -> np.ndarray:
    points = vec.reshape(-1, 3)
    left = compute_center(points, LEFT_EYE_LOOP)
    right = compute_center(points, RIGHT_EYE_LOOP)
    mouth = compute_center(points, LIPS_OUTER)
    mid_eyes = (left + right) * 0.5

    x_axis = right - left
    y_axis = mouth - mid_eyes
    x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)
    y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) or 1.0)
    if z_axis[2] < 0:
        z_axis = -z_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

    rot = np.stack([x_axis, y_axis, z_axis], axis=0)
    return (points @ rot.T).reshape(-1)


def compute_ranges(vec: np.ndarray) -> Tuple[float, float, float, float]:
    pts = vec.reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    range_x = max(1e-6, float(maxs[0] - mins[0]))
    range_y = max(1e-6, float(maxs[1] - mins[1]))
    range_z = max(1e-6, float(maxs[2] - mins[2]))
    return range_x, range_y, range_z, (range_x + range_y) * 0.5


def align_vector(vec: np.ndarray, aspect: float, mean_ranges: Tuple[float, float, float, float]) -> np.ndarray:
    out = vec.copy()
    out[0::3] *= aspect
    out = normalize_face(out)
    range_x, range_y, range_z, xy_avg = compute_ranges(out)
    mean_range_x, mean_range_y, mean_range_z, mean_xy_avg = mean_ranges
    mean_ratio = mean_range_z / mean_xy_avg
    upload_ratio = range_z / xy_avg
    z_scale = mean_ratio / upload_ratio if upload_ratio > 0 else 1.0
    out[2::3] *= z_scale
    return out


def load_validation_landmarks(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    data: Dict[str, np.ndarray] = {}
    aspect: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            image_name = rec.get("image", "")
            stem = Path(image_name).stem
            faces = rec.get("faces", [])
            if not faces:
                continue
            landmarks = np.array(faces[0]["landmarks"], dtype=np.float64)
            if landmarks.ndim != 2 or landmarks.shape[1] != 3:
                continue
            data[stem] = landmarks.reshape(-1)
            width = rec.get("width") or 0
            height = rec.get("height") or 0
            if width and height:
                aspect[stem] = float(width) / float(height)
    return data, aspect


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing model npz: {args.npz}")
    if not args.landmarks_jsonl.exists():
        raise SystemExit(f"Missing landmarks jsonl: {args.landmarks_jsonl}")
    if not args.raw_csv.exists():
        raise SystemExit(f"Missing raw validation csv: {args.raw_csv}")

    model = np.load(args.npz, allow_pickle=True)
    coef = model["coef"].astype(np.float32)  # [A, P]
    attrs = [str(a) for a in model["attrs"]]
    y_mean = model["y_mean"].astype(np.float32).reshape(-1)
    y_std = model["y_std"].astype(np.float32).reshape(-1)
    x_mean = model["x_mean"].astype(np.float32).reshape(-1)
    x_std = model["x_std"].astype(np.float32).reshape(-1)
    normalize_flag = int(model.get("normalize", np.array([0]))[0])
    mean_ranges = tuple(model.get("mean_ranges", np.array([0.0, 0.0, 0.0, 0.0])).tolist())

    lm_map, aspect_map = load_validation_landmarks(args.landmarks_jsonl)

    by_attr_pred: Dict[str, List[float]] = {a: [] for a in attrs}
    by_attr_true: Dict[str, List[float]] = {a: [] for a in attrs}
    by_key_sum: Dict[Tuple[str, str, str], float] = {}
    by_key_count: Dict[Tuple[str, str, str], int] = {}

    with args.raw_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"].strip()
            stim = row["stimulus_index"].strip()
            level = row["stimulus_level"].strip()
            key = (stim, cond, level)
            try:
                resp = float(row["response"])
            except ValueError:
                continue
            by_key_sum[key] = by_key_sum.get(key, 0.0) + resp
            by_key_count[key] = by_key_count.get(key, 0) + 1

    for (stim, cond, level), total in by_key_sum.items():
        if cond not in by_attr_pred:
            continue
        stem = f"{stim}_{cond}_{level}"
        vec = lm_map.get(stem)
        if vec is None:
            continue
        aspect = aspect_map.get(stem, 1.0)
        if normalize_flag:
            vec_use = align_vector(vec, aspect, mean_ranges)
        else:
            vec_use = vec
        y_z = (vec_use - y_mean) / np.where(y_std == 0, 1.0, y_std)
        attr_idx = attrs.index(cond)
        pred_z = coef[attr_idx] @ y_z
        pred = pred_z * (x_std[attr_idx] or 1.0) + x_mean[attr_idx]
        avg_resp = total / max(1, by_key_count[(stim, cond, level)])
        by_attr_pred[cond].append(float(pred))
        by_attr_true[cond].append(float(avg_resp))

    print("Pearson correlation by attribute:")
    vals = []
    for attr in attrs:
        preds = np.array(by_attr_pred[attr], dtype=np.float64)
        trues = np.array(by_attr_true[attr], dtype=np.float64)
        if len(preds) < 2:
            print(f"{attr}\t(n={len(preds)})\tNA")
            continue
        r = pearson_corr(preds, trues)
        vals.append(r)
        print(f"{attr}\t(n={len(preds)})\t{r:.4f}")

    if vals:
        mean_r = float(np.nanmean(vals))
        print(f"\nMean Pearson r across attributes: {mean_r:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
