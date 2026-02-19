#!/usr/bin/env python
"""
Build omi_mesh_data.json from a regression/correlation NPZ and face mesh obj.

Includes:
  - attrs, mean landmarks, coef/corr
  - triangles and edges from the official mesh obj
  - attr_score_sorted for percentile readouts in the viewer
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build omi_mesh_data.json payload.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("omi") / "corr" / "partial_regression.npz",
        help="Path to NPZ with coef.",
    )
    parser.add_argument(
        "--corr_npz",
        type=Path,
        default=Path("omi") / "corr" / "spearman_corr.npz",
        help="Optional NPZ with corr (for mode toggle).",
    )
    parser.add_argument(
        "--uni_npz",
        type=Path,
        default=Path("omi") / "corr" / "univariate_regression.npz",
        help="Optional NPZ with univariate regression coef (for mode toggle).",
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
        "--obj",
        type=Path,
        default=Path("face_model_with_iris.obj"),
        help="Path to official face_model_with_iris.obj",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("omi_mesh_data.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="partial",
        help="Label for regression_meta.type (e.g. partial, linear, spearman).",
    )
    parser.add_argument(
        "--apply_symmetry",
        action="store_true",
        help="Apply symmetric/asymmetric split and keep asymmetry on the right side.",
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


def parse_obj_faces(path: Path) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            parts = line.strip().split()[1:]
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) < 3:
                continue
            for i in range(1, len(idx) - 1):
                faces.append((idx[0], idx[i], idx[i + 1]))
    return faces


LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_EYE_LOOP = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_LOOP = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]


def compute_center(points: np.ndarray, indices: List[int]) -> np.ndarray:
    return points[indices].mean(axis=0)


def compute_axes(mean_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left = compute_center(mean_points, LEFT_EYE_LOOP)
    right = compute_center(mean_points, RIGHT_EYE_LOOP)
    mouth = compute_center(mean_points, LIPS_OUTER)
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

    return np.stack([x_axis, y_axis, z_axis], axis=0), mid_eyes  # rows, origin


def apply_rotation(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return points @ rot.T


def invert_rotation(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return points @ rot


def build_symmetry_map(points: np.ndarray) -> List[int]:
    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    diff = mirrored[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist.argmin(axis=1).tolist()


def apply_symmetry_only(vectors: np.ndarray, mean_points: np.ndarray, center_threshold: float = 0.05) -> np.ndarray:
    rot, origin = compute_axes(mean_points)
    mean_centered = mean_points - origin
    mean_rot = apply_rotation(mean_centered, rot)
    sym_map = build_symmetry_map(mean_rot)

    out = vectors.copy()
    for a in range(out.shape[0]):
        vec = out[a].reshape(-1, 3)
        vec_rot = apply_rotation(vec, rot)
        vec_new = vec_rot.copy()
        for idx, x in enumerate(mean_rot[:, 0]):
            if abs(x) < center_threshold or sym_map[idx] == idx:
                vec_new[idx, 0] = 0.0
        visited = np.zeros(len(sym_map), dtype=bool)
        for i, j in enumerate(sym_map):
            if i == j:
                continue
            xi = mean_rot[i, 0]
            xj = mean_rot[j, 0]
            # Skip if both are on the same side or near center
            if xi == 0 or xj == 0 or np.sign(xi) == np.sign(xj):
                continue
            left = i if xi < 0 else j
            right = j if left == i else i
            if visited[left] or visited[right]:
                continue
            visited[left] = True
            visited[right] = True

            lvec = vec_rot[left]
            rvec = vec_rot[right]
            r_mirror = np.array([-rvec[0], rvec[1], rvec[2]])
            sym = 0.5 * (lvec + r_mirror)
            vec_new[left] = sym
            vec_new[right] = np.array([-sym[0], sym[1], sym[2]])

        out[a] = invert_rotation(vec_new, rot).reshape(-1)
    return out


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing npz file: {args.npz}")
    if not args.attr_csv.exists():
        raise SystemExit(f"Missing attributes file: {args.attr_csv}")
    if not args.landmarks_jsonl.exists():
        raise SystemExit(f"Missing landmarks file: {args.landmarks_jsonl}")
    if not args.obj.exists():
        raise SystemExit(f"Missing obj file: {args.obj}")

    data = np.load(args.npz, allow_pickle=True)
    coef = data["coef"].astype(np.float32)
    attrs = [str(a) for a in data["attrs"]]
    mean_landmarks = data["mean_landmarks"].astype(np.float32)
    used_ids = [str(i) for i in data["used_ids"]]

    corr = None
    if args.corr_npz.exists():
        corr_data = np.load(args.corr_npz, allow_pickle=True)
        if "corr" in corr_data:
            corr = corr_data["corr"].astype(np.float32)

    uni = None
    if args.uni_npz.exists():
        uni_data = np.load(args.uni_npz, allow_pickle=True)
        if "coef" in uni_data:
            uni = uni_data["coef"].astype(np.float32)

    _, attr_map = load_attributes(args.attr_csv)
    lm_map, _ = load_landmarks(args.landmarks_jsonl)

    # Build landmark matrix for used_ids
    Y = np.stack([lm_map[i].reshape(-1) for i in used_ids], axis=0)
    mean_flat = mean_landmarks.reshape(-1)

    if args.apply_symmetry:
        coef = apply_symmetry_only(coef, mean_landmarks)
        if corr is not None:
            corr = apply_symmetry_only(corr, mean_landmarks)
        if uni is not None:
            uni = apply_symmetry_only(uni, mean_landmarks)

    def vector_norm(vec: np.ndarray) -> float:
        return float(np.sqrt((vec * vec).sum()))

    def average_norm(vectors: np.ndarray | None) -> float:
        if vectors is None or len(vectors) == 0:
            return 1.0
        total = 0.0
        for v in vectors:
            total += vector_norm(v)
        return total / len(vectors) if total > 0 else 1.0

    def scale_vectors(vectors: np.ndarray | None, target: float) -> np.ndarray | None:
        if vectors is None:
            return None
        current = average_norm(vectors)
        scale = target / (current or 1.0)
        if abs(scale - 1.0) < 1e-6:
            return vectors
        return (vectors * scale).astype(np.float32)

    def build_recommended(partial: np.ndarray | None, univariate: np.ndarray | None) -> np.ndarray | None:
        if partial is None or univariate is None:
            return None
        out = np.zeros_like(partial, dtype=np.float32)
        for i in range(partial.shape[0]):
            p = partial[i]
            u = univariate[i]
            p_norm = vector_norm(p) or 1.0
            u_norm = vector_norm(u) or 1.0
            out[i] = (p / p_norm) + (u / u_norm)
        return out

    recommended = build_recommended(coef, uni)
    base_norm = average_norm(coef if coef is not None else (corr if corr is not None else (uni if uni is not None else recommended)))

    coef = scale_vectors(coef, base_norm)
    corr = scale_vectors(corr, base_norm)
    uni = scale_vectors(uni, base_norm)
    recommended = scale_vectors(recommended, base_norm)

    def build_sorted_scores(vectors: np.ndarray) -> List[List[float]]:
        scores_local = (Y - mean_flat[None, :]) @ vectors.T
        return [sorted(scores_local[:, i].tolist()) for i in range(vectors.shape[0])]

    attr_score_sorted = build_sorted_scores(coef)
    attr_score_sorted_corr = build_sorted_scores(corr) if corr is not None else None
    attr_score_sorted_uni = build_sorted_scores(uni) if uni is not None else None
    attr_score_sorted_rec = build_sorted_scores(recommended) if recommended is not None else None

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
        "coef": coef.tolist(),
        "corr": corr.tolist() if corr is not None else None,
        "uni": uni.tolist() if uni is not None else None,
        "rec": recommended.tolist() if recommended is not None else None,
        "edges": sorted(edges),
        "triangles": triangles,
        "landmark_count": int(mean_landmarks.shape[0]),
        "attr_score_sorted": attr_score_sorted,
        "attr_score_sorted_corr": attr_score_sorted_corr,
        "attr_score_sorted_uni": attr_score_sorted_uni,
        "attr_score_sorted_rec": attr_score_sorted_rec,
        "regression_meta": {
            "type": args.mode,
            "residualize": "attributes+landmarks",
            "standardized": True,
            "asymmetry": "symmetric" if args.apply_symmetry else "none",
        },
    }

    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Updated {args.output} with {len(triangles)} triangles and coef shape {coef.shape}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
