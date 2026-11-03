from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from patchguard.api import create_app
from patchguard.features import MAX_BYTES, decode_image


@pytest.fixture
def client(trained, tmp_path):
    path = tmp_path / "model.npz"
    trained.save(path)
    with TestClient(create_app(path)) as client:
        yield client


def test_full_inspection_and_metrics(client):
    raw = client.get("/api/examples/scratch").content
    response = client.post("/api/inspect", files={"file": ("surface.png", raw, "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "anomaly"
    assert result["overlay"].startswith("data:image/png;base64,")
    assert len(result["patch_scores"]) == 16
    assert result["regions"]
    assert "patchguard_requests_total 1" in client.get("/metrics").text
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_invalid_image_is_rejected_without_counting(client):
    response = client.post("/api/inspect", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert "patchguard_requests_total 0" in client.get("/metrics").text


def test_oversized_upload_and_unknown_example(client):
    assert client.post("/api/inspect", files={"file": ("big.png", b"x" * (MAX_BYTES + 1))}).status_code == 413
    assert client.get("/api/examples/unknown").status_code == 404


def test_grayscale_and_exif_decoding():
    buffer = BytesIO()
    Image.new("L", (64, 64), 120).save(buffer, format="PNG")
    assert decode_image(buffer.getvalue()).mode == "RGB"


def test_small_and_unsupported_images_rejected():
    for size, fmt in [((16, 16), "PNG"), ((64, 64), "GIF")]:
        buffer = BytesIO()
        Image.new("RGB", size).save(buffer, format=fmt)
        with pytest.raises(ValueError):
            decode_image(buffer.getvalue())


def test_missing_model_fails_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError), TestClient(create_app(tmp_path / "missing.npz")):
        pass
