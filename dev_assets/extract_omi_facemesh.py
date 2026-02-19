#!/usr/bin/env python
"""
Extract MediaPipe Face Landmarker landmarks from OMI images.

Default:
- input:  omi/images
- output: omi/landmarks/face_landmarks.jsonl

Requires:
- mediapipe
- pillow
- numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import mediapipe tasks. Install with: pip install mediapipe"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Face Landmarker landmarks for OMI images."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("omi") / "images",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("omi") / "landmarks" / "face_landmarks.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to face_landmarker.task model file.",
    )
    parser.add_argument(
        "--max_faces",
        type=int,
        default=1,
        help="Maximum number of faces to detect per image.",
    )
    parser.add_argument(
        "--output_blendshapes",
        action="store_true",
        help="Include face blendshapes in output.",
    )
    parser.add_argument(
        "--per_image",
        action="store_true",
        help="Write one JSON file per image instead of JSONL.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already have output (per-image mode only).",
    )
    return parser.parse_args()


def build_landmarker(
    model_path: Path, max_faces: int, output_blendshapes: bool
) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        output_face_blendshapes=output_blendshapes,
    )
    return vision.FaceLandmarker.create_from_options(options)


def load_image(path: Path) -> mp.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        data = np.asarray(img)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=data)


def landmarks_to_list(landmarks: List[Any]) -> List[List[float]]:
    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks]


def blendshapes_to_list(blendshapes: List[Any]) -> List[Dict[str, float]]:
    return [
        {"category": bs.category_name, "score": float(bs.score)}
        for bs in blendshapes
    ]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    if not args.model.exists():
        raise SystemExit(
            f"Model file not found: {args.model}\n"
            "Download face_landmarker.task from the official MediaPipe models and pass --model."
        )

    image_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    landmarker = build_landmarker(
        args.model, max_faces=args.max_faces, output_blendshapes=args.output_blendshapes
    )

    if args.per_image:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    processed = 0
    for path in image_paths:
        out_path = args.output / f"{path.stem}.json" if args.per_image else None
        if args.per_image and args.skip_existing and out_path.exists():
            continue

        mp_image = load_image(path)
        result = landmarker.detect(mp_image)

        faces = []
        if result.face_landmarks:
            for idx, face_landmarks in enumerate(result.face_landmarks):
                face = {
                    "index": idx,
                    "landmarks": landmarks_to_list(face_landmarks),
                }
                if args.output_blendshapes and result.face_blendshapes:
                    face["blendshapes"] = blendshapes_to_list(
                        result.face_blendshapes[idx]
                    )
                faces.append(face)

        record = {
            "image": path.name,
            "width": mp_image.width,
            "height": mp_image.height,
            "faces": faces,
        }

        if args.per_image:
            write_json(out_path, record)
        else:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{len(image_paths)} images...")

    if not args.per_image:
        out_f.close()

    print(f"Done. Processed {processed} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
