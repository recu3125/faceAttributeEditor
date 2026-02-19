#!/usr/bin/env python
"""
Update omi_mesh_viewer.html data payload using official MediaPipe face mesh triangles.

You must provide face_model_with_iris.obj (official canonical mesh) manually because
this environment cannot fetch it from the internet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update mesh viewer data payload.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("omi") / "corr" / "spearman_corr.npz",
        help="Path to spearman_corr.npz",
    )
    parser.add_argument(
        "--obj",
        type=Path,
        required=True,
        help="Path to official face_model_with_iris.obj",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("omi_mesh_viewer.html"),
        help="Path to HTML file to update.",
    )
    return parser.parse_args()


def parse_obj_faces(path: Path) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            parts = line.strip().split()[1:]
            # Convert "v", "v/t", "v/t/n" to vertex index
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) < 3:
                continue
            # Fan triangulation if polygon
            for i in range(1, len(idx) - 1):
                faces.append((idx[0], idx[i], idx[i + 1]))
    return faces


def update_html_payload(html_path: Path, payload: dict) -> None:
    html = html_path.read_text(encoding="utf-8")
    start = html.find('<script id="data" type="application/json">')
    if start == -1:
        raise SystemExit("Could not find data script tag in HTML.")
    start = html.find(">", start) + 1
    end = html.find("</script>", start)
    if end == -1:
        raise SystemExit("Could not find closing </script> for data tag.")
    new_html = html[:start] + json.dumps(payload) + html[end:]
    html_path.write_text(new_html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing npz file: {args.npz}")
    if not args.obj.exists():
        raise SystemExit(f"Missing obj file: {args.obj}")
    if not args.html.exists():
        raise SystemExit(f"Missing html file: {args.html}")

    data = np.load(args.npz, allow_pickle=True)
    attrs = [str(a) for a in data["attrs"]]
    mean_landmarks = data["mean_landmarks"].astype(np.float32)
    corr = data["corr"].astype(np.float32)

    triangles = parse_obj_faces(args.obj)
    if not triangles:
        raise SystemExit("No faces found in obj file.")

    edges = set()
    for a, b, c in triangles:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    payload = {
        "attrs": attrs,
        "mean": mean_landmarks.reshape(-1).tolist(),
        "corr": corr.tolist(),
        "edges": sorted(edges),
        "triangles": triangles,
        "landmark_count": int(mean_landmarks.shape[0]),
    }

    update_html_payload(args.html, payload)
    print(f"Updated {args.html} with {len(triangles)} official triangles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
