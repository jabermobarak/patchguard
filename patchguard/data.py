"""Synthetic texture fixture and strict folder-based dataset loading."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .features import SIZE, decode_image

KINDS = ("normal", "scratch", "stain", "hole")


def sample(seed: int, kind: str = "normal", shift: str = "clean") -> tuple[Image.Image, np.ndarray]:
    if kind not in KINDS:
        raise ValueError("Unknown defect type")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:SIZE, :SIZE]
    phase = rng.uniform(-0.2, 0.2)
    wave = 0.035 * np.sin(x * 0.48 + phase) + 0.025 * np.cos(y * 0.6 + phase)
    base = 0.48 + wave + rng.normal(0, 0.014, (SIZE, SIZE))
    colors = np.array([1.07, 0.98, 0.88]) + rng.normal(0, 0.008, 3)
    arr = np.clip(base[..., None] * colors + rng.normal(0, 0.003, (SIZE, SIZE, 3)), 0, 1)
    image = Image.fromarray((arr * 255).astype(np.uint8))
    mask_image = Image.new("L", (SIZE, SIZE))
    painter, mask = ImageDraw.Draw(image), ImageDraw.Draw(mask_image)
    px, py = rng.integers(18, 90, 2).tolist()
    if kind == "scratch":
        end = (min(px + int(rng.integers(18, 38)), 119), min(py + int(rng.integers(15, 30)), 119))
        width = int(rng.integers(2, 5))
        painter.line([(px, py), end], fill=(190, 182, 165), width=width)
        mask.line([(px, py), end], fill=255, width=width)
    elif kind == "stain":
        box = [px, py, px + int(rng.integers(10, 22)), py + int(rng.integers(8, 20))]
        painter.ellipse(box, fill=(94, 76, 58))
        mask.ellipse(box, fill=255)
    elif kind == "hole":
        box = [px, py, px + int(rng.integers(5, 13)), py + int(rng.integers(5, 13))]
        painter.ellipse(box, fill=(38, 34, 29))
        mask.ellipse(box, fill=255)
    if shift == "bright":
        image = ImageEnhance.Brightness(image).enhance(1.22)
    elif shift == "blur":
        image = image.filter(ImageFilter.GaussianBlur(1.2))
    elif shift != "clean":
        raise ValueError("Unknown shift")
    return image, np.asarray(mask_image) > 0


def demo_splits():
    train = [sample(1000 + i)[0] for i in range(80)]
    calibration = [sample(2000 + i)[0] for i in range(60)]
    test = [(3000 + i, KINDS[i % 4]) for i in range(160)]
    return train, calibration, test


@dataclass
class Record:
    path: Path
    image: Image.Image
    digest: str


def load_folder(folder: Path) -> list[Record]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    records = []
    for path in paths:
        raw = path.read_bytes()
        image = decode_image(raw)
        # Hash decoded pixels too, catching duplicate images with different metadata/encodings.
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        records.append(Record(path, image, digest))
    if not records:
        raise ValueError(f"No supported images in {folder}")
    return records


def disjoint(*groups: list[Record]) -> None:
    seen = set()
    for group in groups:
        current = {item.digest for item in group}
        if current & seen:
            raise ValueError("Duplicate image content crosses dataset splits")
        seen.update(current)
