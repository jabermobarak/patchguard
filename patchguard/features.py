"""Fixed, interpretable patch features. No pretrained weights or downloads."""

import warnings
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.ndimage import gaussian_filter, sobel

SIZE = 128
PATCH = 8
GRID = SIZE // PATCH
FEATURES = 11
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 8 MiB limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Use a PNG, JPEG, or WebP image")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 12 megapixel limit")
                if min(source.size) < 32:
                    raise ValueError("Image must be at least 32 by 32 pixels")
                return ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Cannot decode a supported image") from exc


def normalized(image: Image.Image) -> np.ndarray:
    return (
        np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )


def _pool(x: np.ndarray) -> np.ndarray:
    return x.reshape(GRID, PATCH, GRID, PATCH, -1).mean(axis=(1, 3))


def extract(image: Image.Image, rgb_only: bool = False) -> np.ndarray:
    x = normalized(image)
    mean = _pool(x)
    if rgb_only:
        return mean.astype(np.float32)
    std = np.sqrt(np.maximum(_pool(x * x) - mean * mean, 0))
    gray = x.mean(axis=2)
    dx = sobel(gray, axis=1, mode="reflect") / 8
    dy = sobel(gray, axis=0, mode="reflect") / 8
    detail = gray - gaussian_filter(gray, 1.5)
    # Signs preserve orientation; absolute values capture edge strength.
    extras = np.stack([dx, dy, np.abs(dx), np.abs(dy), np.abs(detail)], axis=-1)
    return np.concatenate([mean, std, _pool(extras)], axis=-1).astype(np.float32)
