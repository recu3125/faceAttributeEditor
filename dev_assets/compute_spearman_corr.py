#!/usr/bin/env python
"""
Compute Spearman correlations between OMI attribute means and face mesh landmarks.

Inputs:
- omi/attribute_means.csv
- omi/landmarks/face_landmarks.jsonl (from extract_omi_facemesh.py)

Outputs:
- omi/corr/spearman_corr.npz
  * corr: [num_attrs, num_landmarks*3]
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
        description="Compute Spearman correlations between attributes and landmarks."
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
        default=Path("omi") / "corr" / "spearman_corr.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--exclude_ids",
        type=Path,
        default=None,
        help="Optional text file with stimulus ids to exclude (one per line).",
    )
    return parser.parse_args()


def rankdata_1d(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for ties, 1..n."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    # Handle ties: average ranks for equal values
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def rankdata_2d(matrix: np.ndarray) -> np.ndarray:
    """Rank each column with average-tie handling."""
    n_rows, n_cols = matrix.shape
    ranked = np.empty((n_rows, n_cols), dtype=np.float64)
    for col in range(n_cols):
        ranked[:, col] = rankdata_1d(matrix[:, col])
    return ranked


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
    if args.exclude_ids and args.exclude_ids.exists():
        exclude = {line.strip() for line in args.exclude_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        common_ids = [i for i in common_ids if i not in exclude]
    if not common_ids:
        raise SystemExit("No overlapping ids between attributes and landmarks.")

    if n_landmarks == 0:
        raise SystemExit("No valid landmarks found.")

    y = np.stack([attr_map[i] for i in common_ids], axis=0)  # [N, A]
    x = np.stack([lm_map[i].reshape(-1) for i in common_ids], axis=0)  # [N, P]

    # Spearman: rank transform then Pearson
    y_rank = rankdata_2d(y)
    x_rank = rankdata_2d(x)

    # Z-score columns
    y_mean = y_rank.mean(axis=0, keepdims=True)
    x_mean = x_rank.mean(axis=0, keepdims=True)
    y_std = y_rank.std(axis=0, ddof=1, keepdims=True)
    x_std = x_rank.std(axis=0, ddof=1, keepdims=True)

    y_z = (y_rank - y_mean) / y_std
    x_z = (x_rank - x_mean) / x_std

    corr = (y_z.T @ x_z) / (y_z.shape[0] - 1)

    mean_landmarks = np.stack([lm_map[i] for i in common_ids], axis=0).mean(axis=0)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        corr=corr.astype(np.float32),
        attrs=np.array(attrs),
        feature_names=np.array(build_feature_names(n_landmarks)),
        mean_landmarks=mean_landmarks.astype(np.float32),
        used_ids=np.array(common_ids),
    )

    print(
        f"Saved {output} with corr shape {corr.shape} "
        f"(attrs={len(attrs)}, features={x.shape[1]}), samples={len(common_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
