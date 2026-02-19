#!/usr/bin/env python
"""
Visualize per-attribute correlation strength over the face mesh.

Usage:
  py -3.11 visualize_corr.py --attr trustworthy
  py -3.11 visualize_corr.py --attr 0 --output omi/corr/trustworthy.png
  py -3.11 visualize_corr.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Spearman correlations.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("omi") / "corr" / "spearman_corr.npz",
        help="Path to spearman_corr.npz",
    )
    parser.add_argument(
        "--attr",
        type=str,
        default=None,
        help="Attribute name or index (0-based).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available attributes and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing file: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    corr = data["corr"]  # [A, L*3]
    attrs = [str(a) for a in data["attrs"]]
    mean_landmarks = data["mean_landmarks"]  # [L, 3]

    if args.list:
        for i, name in enumerate(attrs):
            print(f"{i}\t{name}")
        return 0

    if args.attr is None:
        raise SystemExit("Provide --attr NAME or --attr INDEX. Use --list to see options.")

    if args.attr.isdigit():
        idx = int(args.attr)
        if idx < 0 or idx >= len(attrs):
            raise SystemExit(f"Index out of range: {idx}")
    else:
        if args.attr not in attrs:
            raise SystemExit(f"Attribute not found: {args.attr}")
        idx = attrs.index(args.attr)

    n_landmarks = mean_landmarks.shape[0]
    corr_vec = corr[idx].reshape(n_landmarks, 3)
    mag = np.linalg.norm(corr_vec, axis=1)  # strength per landmark

    fig = plt.figure(figsize=(6, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Spearman |corr| strength: {attrs[idx]}")

    xs = mean_landmarks[:, 0]
    ys = mean_landmarks[:, 1]
    zs = mean_landmarks[:, 2]
    sc = ax.scatter(xs, ys, zs, c=mag, cmap="viridis", s=6, alpha=0.9)
    fig.colorbar(sc, ax=ax, shrink=0.7)

    # Match image-like view: invert y for conventional face orientation
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.invert_yaxis()

    out = args.output
    if out is None:
        out = Path("omi") / "corr" / f"corr_{attrs[idx]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
