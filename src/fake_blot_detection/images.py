"""Put real and synthetic blot images into one common format.

Every image goes through the same steps (grayscale, same longest side, same JPEG quality),
so a detector can't tell real from synthetic by file format, size or compression alone.
"""

import csv
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
MANIFEST_FIELDS = ["image_id", "path", "label", "source", "original"]


@dataclass(frozen=True)
class StandardizeConfig:
    long_side: int = 512
    jpeg_quality: int = 90


DEFAULT_CONFIG = StandardizeConfig()


def standardize(img: Image.Image, config: StandardizeConfig = DEFAULT_CONFIG) -> bytes:
    """Return the image as grayscale JPEG bytes with its longest side resized to `long_side`."""
    img = img.convert("L")
    scale = config.long_side / max(img.size)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.jpeg_quality)
    return buf.getvalue()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def build_dataset(
    sources: dict[str, tuple[Path, str]],
    out_dir: Path,
    config: StandardizeConfig = DEFAULT_CONFIG,
    max_per_source: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Standardize images from several source folders into `out_dir` and write manifest.csv.

    `sources` maps a source name (e.g. "biofors", "claude_procedural") to
    (folder, label), where label is "real" or "ai". Output files get opaque hash names
    so nothing about their origin leaks through the filename.
    """
    rng = random.Random(seed)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, (folder, label) in sources.items():
        if label not in ("real", "ai"):
            raise ValueError(f"label for {source!r} must be 'real' or 'ai', got {label!r}")
        paths = list_images(folder)
        if max_per_source is not None and len(paths) > max_per_source:
            paths = sorted(rng.sample(paths, max_per_source))
        for path in paths:
            with Image.open(path) as img:
                data = standardize(img, config)
            image_id = hashlib.sha256(data).hexdigest()[:16]
            (img_dir / f"{image_id}.jpg").write_bytes(data)
            rows.append(
                {
                    "image_id": image_id,
                    "path": f"images/{image_id}.jpg",
                    "label": label,
                    "source": source,
                    "original": str(path),
                }
            )
    rng.shuffle(rows)
    write_manifest(out_dir / "manifest.csv", rows)
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
