#!/usr/bin/env python
"""
Pack landmark->attribute ridge model into browser-friendly meta/bin files.

Inputs:
- omi/corr/landmark_to_attr_ridge.npz

Outputs:
- omi_mesh_data_reverse.meta.json
- omi_mesh_data_reverse.bin

Binary layout (Float32Array):
  [0 : mean_len]                     y_mean (flattened mean landmarks)
  [mean_len : mean_len + std_len]    y_std
  [.. : .. + coef_count]             coef (attrs x features)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reverse model meta/bin.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("omi") / "corr" / "landmark_to_attr_ridge.npz",
        help="Path to landmark_to_attr_ridge.npz",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("omi_mesh_data_reverse"),
        help="Output prefix for .meta.json and .bin files.",
    )
    parser.add_argument(
        "--percentiles_json",
        type=Path,
        default=None,
        help="Optional JSON file with per-attribute percentiles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing npz file: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    coef = data["coef"].astype(np.float32)  # [A, P]
    attrs = [str(a) for a in data["attrs"]]
    y_mean = data["y_mean"].astype(np.float32).reshape(-1)
    y_std = data["y_std"].astype(np.float32).reshape(-1)
    x_mean = data["x_mean"].astype(np.float32).reshape(-1).tolist()
    x_std = data["x_std"].astype(np.float32).reshape(-1).tolist()
    ridge = float(data["ridge"][0]) if "ridge" in data else 0.0

    mean_len = int(y_mean.shape[0])
    std_len = int(y_std.shape[0])
    coef_shape = [int(coef.shape[0]), int(coef.shape[1])]

    floats = np.concatenate([y_mean, y_std, coef.reshape(-1)], axis=0)
    bin_path = args.output_prefix.with_suffix(".bin")
    bin_path.write_bytes(floats.tobytes())

    meta = {
        "attrs": attrs,
        "mean_len": mean_len,
        "std_len": std_len,
        "coef_shape": coef_shape,
        "x_mean": x_mean,
        "x_std": x_std,
        "ridge": ridge,
    }
    if args.percentiles_json and args.percentiles_json.exists():
        meta["percentiles"] = json.loads(args.percentiles_json.read_text(encoding="utf-8"))
    meta_path = args.output_prefix.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    print(f"Wrote {meta_path} and {bin_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
