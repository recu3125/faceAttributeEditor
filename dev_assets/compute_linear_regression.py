#!/usr/bin/env python
"""
Compute linear regression between OMI attribute means and face mesh landmarks.

Inputs:
- omi/attribute_means.csv
- omi/landmarks/face_landmarks.jsonl (from extract_omi_facemesh.py)

Outputs:
- omi/corr/linear_regression.npz
  * coef: [num_attrs, num_landmarks*3]
  * intercept: [num_landmarks*3]
  * attrs: list of attribute names
  * feature_names: list like "lm_0_x", "lm_0_y", "lm_0_z", ...
  * mean_landmarks: [468, 3]
  * used_ids: list of stimulus ids used
  * x_mean, x_std, y_mean, y_std: z-score stats
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
        description="Compute linear regression between attributes and landmarks."
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
        default=Path("omi") / "corr" / "linear_regression.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Ridge regularization strength (0 = OLS).",
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


def load_landmarks(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    data: Dict[str, np.ndarray] = {}
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
    return data, int(n_landmarks or 0)


def build_feature_names(n_landmarks: int = 468) -> List[str]:
    names = []
    for i in range(n_landmarks):
        names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
    return names


def main() -> int:
    args = parse_args()
    if not args.attr_csv.exists():
        raise SystemExit(f"Missing attributes file: {args.attr_csv}")
    if not args.landmarks_jsonl.exists():
        raise SystemExit(f"Missing landmarks file: {args.landmarks_jsonl}")

    attrs, attr_map = load_attributes(args.attr_csv)
    lm_map, n_landmarks = load_landmarks(args.landmarks_jsonl)

    common_ids = sorted(set(attr_map.keys()) & set(lm_map.keys()))
    if not common_ids:
        raise SystemExit("No overlapping ids between attributes and landmarks.")

    if n_landmarks == 0:
        raise SystemExit("No valid landmarks found.")

    # X: attributes, Y: landmarks (flattened)
    X = np.stack([attr_map[i] for i in common_ids], axis=0)  # [N, A]
    Y = np.stack([lm_map[i].reshape(-1) for i in common_ids], axis=0)  # [N, P]

    # Z-score X and Y
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, ddof=1, keepdims=True)
    x_std = np.where(x_std == 0, 1.0, x_std)
    Xz = (X - x_mean) / x_std

    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, ddof=1, keepdims=True)
    y_std = np.where(y_std == 0, 1.0, y_std)
    Yz = (Y - y_mean) / y_std

    # Solve linear regression: Yz = Xz @ B + e
    # Using closed form with optional ridge
    ridge = float(args.ridge)
    XtX = Xz.T @ Xz
    if ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0])
    XtY = Xz.T @ Yz
    coef = np.linalg.solve(XtX, XtY)  # [A, P]

    # Intercept in original space (predict Y from raw X)
    # Y_hat = ((X - x_mean)/x_std) @ coef; then unscale to Y
    # => Y_hat_raw = (X - x_mean)/x_std @ coef * y_std + y_mean
    # => Y_hat_raw = X @ (diag(1/x_std) @ coef @ diag(y_std)) + (y_mean - x_mean/x_std @ coef * y_std)
    coef_raw = (coef * y_std).astype(np.float32)
    coef_raw = (coef_raw / x_std.T).astype(np.float32)
    intercept = (y_mean - (x_mean / x_std) @ (coef * y_std)).astype(np.float32).ravel()

    mean_landmarks = np.stack([lm_map[i] for i in common_ids], axis=0).mean(axis=0)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coef=coef.astype(np.float32),
        coef_raw=coef_raw.astype(np.float32),
        intercept=intercept.astype(np.float32),
        attrs=np.array(attrs),
        feature_names=np.array(build_feature_names(n_landmarks)),
        mean_landmarks=mean_landmarks.astype(np.float32),
        used_ids=np.array(common_ids),
        x_mean=x_mean.astype(np.float32).ravel(),
        x_std=x_std.astype(np.float32).ravel(),
        y_mean=y_mean.astype(np.float32).ravel(),
        y_std=y_std.astype(np.float32).ravel(),
    )

    print(
        f"Saved {output} with coef shape {coef.shape} (attrs={len(attrs)}, features={Y.shape[1]}), "
        f"samples={len(common_ids)}, ridge={ridge}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
