"""Spatial shrinkage Gaussian model with a held-out normal calibration split."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label

from .features import GRID, SIZE, extract


@dataclass
class Detector:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float
    pixel_threshold: float
    rgb_only: bool = False
    model_id: str = "unversioned"
    brightness_bounds: tuple[float, float] = (0, 1)
    texture_bounds: tuple[float, float] = (0, 1)
    training_domain: str = "custom surface dataset"

    @classmethod
    def fit(
        cls,
        images: list[Image.Image],
        calibration: list[Image.Image],
        rgb_only: bool = False,
        quantile: float = 0.975,
    ) -> "Detector":
        if len(images) < 20 or len(calibration) < 20:
            raise ValueError("Need at least 20 training and 20 normal calibration images")
        if not 0.5 < quantile < 1:
            raise ValueError("Calibration quantile must lie between 0.5 and 1")
        x = np.stack([extract(im, rgb_only) for im in images]).astype(np.float64)
        mean = x.mean(axis=0)
        delta = x - mean
        cov = np.einsum("nhwf,nhwg->hwfg", delta, delta) / (len(x) - 1)
        f = x.shape[-1]
        # Shrink toward a diagonal target to stabilize correlated small-sample features.
        diagonal = np.eye(f) * np.diagonal(cov, axis1=-2, axis2=-1)[..., None, :]
        cov = 0.8 * cov + 0.2 * diagonal + np.eye(f) * 1e-5
        detector = cls(
            mean.astype(np.float32), np.linalg.inv(cov).astype(np.float32), 1, 1, rgb_only=rgb_only
        )
        maps = np.stack([detector.patch_scores(im) for im in calibration])
        detector.threshold = float(np.quantile([detector.image_score(m) for m in maps], quantile))
        detector.pixel_threshold = float(np.quantile(maps, 0.995))
        levels = np.array([np.asarray(im.convert("RGB"), dtype=float).mean() / 255 for im in calibration])
        detector.brightness_bounds = (float(levels.min() - 0.06), float(levels.max() + 0.06))
        texture = [float(extract(im)[..., 10].mean()) for im in calibration]
        detector.texture_bounds = (float(min(texture) * 0.7), float(max(texture) * 1.3))
        identity = (
            mean.tobytes()
            + cov.tobytes()
            + np.array(
                [
                    detector.threshold,
                    detector.pixel_threshold,
                    *detector.brightness_bounds,
                    *detector.texture_bounds,
                ],
                dtype=np.float64,
            ).tobytes()
        )
        detector.model_id = hashlib.sha256(identity).hexdigest()[:12]
        return detector

    def patch_scores(self, image: Image.Image) -> np.ndarray:
        d = extract(image, self.rgb_only) - self.mean
        return np.sqrt(np.maximum(np.einsum("hwf,hwfg,hwg->hw", d, self.precision, d), 0))

    @staticmethod
    def image_score(scores: np.ndarray) -> float:
        # Mean of four most anomalous patches reduces single-patch noise.
        return float(np.sort(scores.ravel())[-4:].mean())

    def infer(self, image: Image.Image) -> dict:
        scores = self.patch_scores(image)
        score = self.image_score(scores)
        ratio = score / max(self.threshold, 1e-8)
        brightness = float(np.asarray(image.convert("RGB")).mean() / 255)
        shift = not self.brightness_bounds[0] <= brightness <= self.brightness_bounds[1]
        texture = float(extract(image)[..., 10].mean())
        texture_shift = not self.texture_bounds[0] <= texture <= self.texture_bounds[1]
        status = (
            "review"
            if shift or texture_shift or 0.9 <= ratio <= 1.1
            else ("anomaly" if ratio > 1 else "normal")
        )
        labels, _ = label(scores > self.pixel_threshold)
        boxes = []
        for slices in find_objects(labels):
            if slices:
                y, x = slices
                boxes.append(
                    {
                        "x": x.start / GRID,
                        "y": y.start / GRID,
                        "width": (x.stop - x.start) / GRID,
                        "height": (y.stop - y.start) / GRID,
                    }
                )
        return {
            "status": status,
            "score": round(score, 5),
            "threshold": round(self.threshold, 5),
            "score_ratio": round(ratio, 4),
            "model_id": self.model_id,
            "brightness_shift": shift,
            "texture_shift": texture_shift,
            "regions": boxes,
            "patch_scores": scores.round(4).tolist(),
            "pixel_threshold": self.pixel_threshold,
            "note": "Scores measure deviation from normal appearance; they are not probabilities.",
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
            "rgb_only": self.rgb_only,
            "model_id": self.model_id,
            "brightness_bounds": self.brightness_bounds,
            "texture_bounds": self.texture_bounds,
            "training_domain": self.training_domain,
            "schema_version": 1,
        }
        np.savez_compressed(path, mean=self.mean, precision=self.precision, metadata=json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.pop("schema_version") != 1:
                raise ValueError("Unsupported model schema")
            model = cls(data["mean"].copy(), data["precision"].copy(), **meta)
        f = 3 if model.rgb_only else 11
        if model.mean.shape != (GRID, GRID, f) or model.precision.shape != (GRID, GRID, f, f):
            raise ValueError("Invalid model dimensions")
        if not np.isfinite(model.mean).all() or not np.isfinite(model.precision).all():
            raise ValueError("Non-finite model parameters")
        if model.threshold <= 0 or model.pixel_threshold <= 0:
            raise ValueError("Invalid model thresholds")
        return model


def heatmap(image: Image.Image, scores: np.ndarray, threshold: float) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    strength = (
        np.asarray(
            Image.fromarray((np.clip(scores / max(threshold, 1e-8), 0, 2) / 2 * 255).astype(np.uint8)).resize(
                (SIZE, SIZE), Image.Resampling.BILINEAR
            )
        )
        / 255
    )
    tint = np.zeros_like(base)
    tint[..., 0] = 255
    tint[..., 1] = 80
    alpha = np.clip(strength * 0.65, 0, 0.65)[..., None]
    return Image.fromarray((base * (1 - alpha) + tint * alpha).astype(np.uint8))
