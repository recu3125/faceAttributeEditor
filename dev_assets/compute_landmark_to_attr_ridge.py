#!/usr/bin/env python
"""
Compute ridge regression from landmarks -> attributes.

Inputs:
- omi/attribute_means.csv
- omi/landmarks/face_landmarks.jsonl (from extract_omi_facemesh.py)

Outputs:
- omi/corr/landmark_to_attr_ridge.npz
  * coef: [num_attrs, num_landmarks*3]  (trained on standardized landmarks)
  * attrs: list of attribute names
  * feature_names: list like "lm_0_x", "lm_0_y", "lm_0_z", ...
  * mean_landmarks: [468, 3]
  * used_ids: list of stimulus ids used
  * y_mean, y_std: landmark normalization stats (flattened)
  * x_mean, x_std: attribute normalization stats
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ridge regression from landmarks to attributes."
    )
    parser.add_argument(
        "--attr_csv",
        type=Path,
        default=Path("omi") / "attribute_means.csv",
        help="Path to attribute_means.csv",
    )
    parser.add_argument(
        "--landmarks_jsonl",
        type=Path,
        default=Path("omi") / "landmarks" / "face_landmarks.jsonl",
        help="Path to face_landmarks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("omi") / "corr" / "landmark_to_attr_ridge.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--exclude_ids",
        type=Path,
        default=None,
        help="Optional text file with stimulus ids to exclude (one per line).",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="Ridge regularization strength (lambda).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable alignment/normalization step.",
    )
    return parser.parse_args()


def load_attributes(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    attrs = [c for c in rows[0].keys() if c != "stimulus"]
    data: Dict[str, np.ndarray] = {}
    for r in rows:
        stim = r["stimulus"].strip()
        data[stim] = np.array([float(r[a]) for a in attrs], dtype=np.float64)
    return attrs, data


def load_landmarks(path: Path) -> Tuple[Dict[str, np.ndarray], int, Dict[str, float]]:
    data: Dict[str, np.ndarray] = {}
    aspect: Dict[str, float] = {}
    n_landmarks = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            image_name = rec.get("image", "")
            stem = Path(image_name).stem
            faces = rec.get("faces", [])
            if not faces:
                continue
            landmarks = np.array(faces[0]["landmarks"], dtype=np.float64)  # [N,3]
            if landmarks.ndim != 2 or landmarks.shape[1] != 3:
                continue
            if n_landmarks is None:
                n_landmarks = landmarks.shape[0]
            if landmarks.shape[0] != n_landmarks:
                continue
            data[stem] = landmarks
            width = rec.get("width") or 0
            height = rec.get("height") or 0
            if width and height:
                aspect[stem] = float(width) / float(height)
    return data, int(n_landmarks or 0), aspect


def build_feature_names(n_landmarks: int = 468) -> List[str]:
    names = []
    for i in range(n_landmarks):
        names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
    return names


LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_EYE_LOOP = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_LOOP = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]


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


def main() -> int:
    args = parse_args()
    if not args.attr_csv.exists():
        raise SystemExit(f"Missing attributes file: {args.attr_csv}")
    if not args.landmarks_jsonl.exists():
        raise SystemExit(f"Missing landmarks file: {args.landmarks_jsonl}")

    attrs, attr_map = load_attributes(args.attr_csv)
    lm_map, n_landmarks, aspect_map = load_landmarks(args.landmarks_jsonl)

    common_ids = sorted(set(attr_map.keys()) & set(lm_map.keys()))
    if args.exclude_ids and args.exclude_ids.exists():
        exclude = {
            line.strip()
            for line in args.exclude_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        common_ids = [i for i in common_ids if i not in exclude]
    if not common_ids:
        raise SystemExit("No overlapping ids between attributes and landmarks.")

    if n_landmarks == 0:
        raise SystemExit("No valid landmarks found.")

    X = np.stack([attr_map[i] for i in common_ids], axis=0)  # [N, A]
    Y_raw = np.stack([lm_map[i].reshape(-1) for i in common_ids], axis=0)  # [N, P]

    mean_ranges = None
    if args.no_normalize:
        Y = Y_raw
        mean_landmarks = Y_raw.reshape(len(common_ids), -1, 3).mean(axis=0)
    else:
        mean_landmarks_raw = Y_raw.reshape(len(common_ids), -1, 3).mean(axis=0)
        mean_aligned = normalize_face(mean_landmarks_raw.reshape(-1))
        mean_ranges = compute_ranges(mean_aligned)
        aligned = []
        for idx, key in enumerate(common_ids):
            aspect = aspect_map.get(key, 1.0)
            aligned.append(align_vector(Y_raw[idx], aspect, mean_ranges))
        Y = np.stack(aligned, axis=0)
        mean_landmarks = Y.reshape(len(common_ids), -1, 3).mean(axis=0)

    # Standardize Y (landmarks)
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, ddof=1, keepdims=True)
    y_std = np.where(y_std == 0, 1.0, y_std)
    Yz = (Y - y_mean) / y_std

    # Standardize X (attributes)
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, ddof=1, keepdims=True)
    x_std = np.where(x_std == 0, 1.0, x_std)
    Xz = (X - x_mean) / x_std

    # Ridge regression: beta = (Y'Y + lambda I)^-1 Y'X
    n_features = Yz.shape[1]
    ridge = float(args.ridge)
    gram = Yz.T @ Yz
    gram.flat[:: n_features + 1] += ridge
    beta = np.linalg.solve(gram, Yz.T @ Xz)  # [P, A]
    coef = beta.T.astype(np.float32)  # [A, P]

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coef=coef,
        attrs=np.array(attrs),
        feature_names=np.array(build_feature_names(n_landmarks)),
        mean_landmarks=mean_landmarks.astype(np.float32),
        used_ids=np.array(common_ids),
        y_mean=y_mean.astype(np.float32).reshape(-1),
        y_std=y_std.astype(np.float32).reshape(-1),
        x_mean=x_mean.astype(np.float32).reshape(-1),
        x_std=x_std.astype(np.float32).reshape(-1),
        ridge=np.array([ridge], dtype=np.float32),
        normalize=np.array([0 if args.no_normalize else 1], dtype=np.int32),
        mean_ranges=np.array(mean_ranges or (0.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )

    print(
        f"Saved {output} with coef shape {coef.shape} "
        f"(attrs={len(attrs)}, features={Y.shape[1]}), samples={len(common_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
