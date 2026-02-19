#!/usr/bin/env python
"""
Compute partial regression (partial correlation) between OMI attributes and face mesh landmarks.

For each attribute:
  1) Residualize the attribute against all other attributes (with intercept).
  2) Residualize each landmark coordinate against all other attributes (with intercept).
  3) Compute correlation between residualized attribute and residualized coordinates.

Outputs:
- omi/corr/partial_regression.npz
  * coef: [num_attrs, num_landmarks*3]  (partial correlation style, z-scored residuals)
  * attrs: list of attribute names
  * feature_names: list like "lm_0_x", "lm_0_y", "lm_0_z", ...
  * mean_landmarks: [468, 3]
  * used_ids: list of stimulus ids used
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
        description="Compute partial regression between attributes and landmarks."
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
        default=Path("omi") / "corr" / "partial_regression.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--exclude_ids",
        type=Path,
        default=None,
        help="Optional text file with stimulus ids to exclude (one per line).",
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


def residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residualize target on covariates (with intercept)."""
    n = covariates.shape[0]
    X = np.concatenate([np.ones((n, 1), dtype=np.float64), covariates], axis=1)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ beta


def main() -> int:
    args = parse_args()
    if not args.attr_csv.exists():
        raise SystemExit(f"Missing attributes file: {args.attr_csv}")
    if not args.landmarks_jsonl.exists():
        raise SystemExit(f"Missing landmarks file: {args.landmarks_jsonl}")

    attrs, attr_map = load_attributes(args.attr_csv)
    lm_map, n_landmarks = load_landmarks(args.landmarks_jsonl)

    common_ids = sorted(set(attr_map.keys()) & set(lm_map.keys()))
    if args.exclude_ids and args.exclude_ids.exists():
        exclude = {line.strip() for line in args.exclude_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        common_ids = [i for i in common_ids if i not in exclude]
    if not common_ids:
        raise SystemExit("No overlapping ids between attributes and landmarks.")

    if n_landmarks == 0:
        raise SystemExit("No valid landmarks found.")

    X = np.stack([attr_map[i] for i in common_ids], axis=0)  # [N, A]
    Y = np.stack([lm_map[i].reshape(-1) for i in common_ids], axis=0)  # [N, P]

    n_samples, n_attrs = X.shape
    n_features = Y.shape[1]

    coef = np.zeros((n_attrs, n_features), dtype=np.float64)

    for a in range(n_attrs):
        mask = np.ones(n_attrs, dtype=bool)
        mask[a] = False
        X_other = X[:, mask]

        x_res = residualize(X[:, a], X_other)
        y_res = residualize(Y, X_other)

        x_mean = x_res.mean()
        x_std = x_res.std(ddof=1)
        if x_std == 0:
            x_std = 1.0
        x_z = (x_res - x_mean) / x_std

        y_mean = y_res.mean(axis=0)
        y_std = y_res.std(axis=0, ddof=1)
        y_std = np.where(y_std == 0, 1.0, y_std)
        y_z = (y_res - y_mean) / y_std

        coef[a] = (x_z[:, None] * y_z).sum(axis=0) / (n_samples - 1)

    mean_landmarks = np.stack([lm_map[i] for i in common_ids], axis=0).mean(axis=0)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coef=coef.astype(np.float32),
        attrs=np.array(attrs),
        feature_names=np.array(build_feature_names(n_landmarks)),
        mean_landmarks=mean_landmarks.astype(np.float32),
        used_ids=np.array(common_ids),
    )

    print(
        f"Saved {output} with coef shape {coef.shape} "
        f"(attrs={len(attrs)}, features={Y.shape[1]}), samples={len(common_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
