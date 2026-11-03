import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from patchguard.data import Record, disjoint, sample
from patchguard.export import export_onnx
from patchguard.features import extract
from patchguard.model import Detector


def test_unseen_defect_is_ranked_above_normal(trained):
    normal = trained.infer(sample(9000)[0])
    defect = trained.infer(sample(9000, "stain")[0])
    assert defect["score"] > normal["score"] * 2
    assert defect["regions"]
    assert defect["status"] == "anomaly"


def test_capture_shift_routes_to_review(trained):
    assert trained.infer(sample(9000, shift="bright")[0])["status"] == "review"
    assert trained.infer(sample(9000, shift="blur")[0])["status"] == "review"


def test_model_roundtrip_preserves_predictions(trained, tmp_path):
    path = tmp_path / "model.npz"
    trained.save(path)
    loaded = Detector.load(path)
    image = sample(9001, "scratch")[0]
    assert loaded.infer(image) == trained.infer(image)


def test_portable_batch_scoring_matches_python(trained, tmp_path):
    path = tmp_path / "scorer.onnx"
    export_onnx(trained, path)
    images = [sample(9000 + i, kind)[0] for i, kind in enumerate(["normal", "scratch", "stain", "hole"])]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": np.stack([extract(image) for image in images])})[0]
    expected = np.stack([trained.patch_scores(image) for image in images])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_leakage_rejected_even_if_paths_differ(tmp_path):
    image = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="Duplicate"):
        disjoint([Record(tmp_path / "a.png", image, "same")], [Record(tmp_path / "b.png", image, "same")])


def test_fit_requires_separate_sufficient_calibration():
    with pytest.raises(ValueError, match="at least 20"):
        Detector.fit([sample(5)[0]], [sample(6)[0]])


def test_test_data_does_not_change_frozen_threshold(trained):
    threshold = trained.threshold
    for kind in ("normal", "hole", "stain"):
        trained.infer(sample(9100, kind)[0])
    assert trained.threshold == threshold
