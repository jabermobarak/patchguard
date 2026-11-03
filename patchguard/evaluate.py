"""Frozen-threshold evaluation, ablation, stress tests, and scoring-head parity."""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .data import demo_splits, sample
from .export import export_onnx
from .features import SIZE, extract
from .model import Detector, heatmap


def metrics(model: Detector, items: list) -> dict:
    truth, scores, pixels, masks, times, predictions, ious = [], [], [], [], [], [], []
    per_kind = {}
    route_counts = {"normal": 0, "anomaly": 0, "review": 0}
    accepted_errors = 0
    for name, image, mask in items:
        start = time.perf_counter()
        patch = model.patch_scores(image)
        score = model.image_score(patch)
        times.append((time.perf_counter() - start) * 1000)
        is_bad = bool(mask.any())
        decision = model.infer(image)["status"]
        route_counts[decision] += 1
        accepted_errors += int(decision != "review" and (decision == "anomaly") != is_bad)
        truth.append(is_bad)
        scores.append(score)
        predictions.append(score > model.threshold)
        # Bilinear interpolation is also what the UI uses for display.
        dense = np.asarray(Image.fromarray(patch).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        pixels.extend(dense.ravel())
        masks.extend(mask.ravel())
        pred_mask = dense > model.pixel_threshold
        if is_bad:
            ious.append(float((pred_mask & mask).sum() / max((pred_mask | mask).sum(), 1)))
        bucket = per_kind.setdefault(name, {"count": 0, "flagged": 0})
        bucket["count"] += 1
        bucket["flagged"] += int(score > model.threshold)
    truth, scores, predictions = np.array(truth), np.array(scores), np.array(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0
    )
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(300):
        indexes = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[indexes])) == 2:
            bootstrap.append(roc_auc_score(truth[indexes], scores[indexes]))
    return {
        "n": len(truth),
        "route_counts": route_counts,
        "automatic_coverage": 1 - route_counts["review"] / len(truth),
        "accepted_errors": accepted_errors,
        "image_auroc": float(roc_auc_score(truth, scores)),
        "image_auroc_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "image_average_precision": float(average_precision_score(truth, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "normal_false_positive_rate": float(predictions[~truth].mean()),
        "pixel_auroc": float(roc_auc_score(masks, pixels)),
        "defect_mean_iou": float(np.mean(ious)),
        "latency_ms_p50": float(np.median(times)),
        "latency_ms_p95": float(np.quantile(times, 0.95)),
        "confusion": {
            "tn": int((~truth & ~predictions).sum()),
            "fp": int((~truth & predictions).sum()),
            "fn": int((truth & ~predictions).sum()),
            "tp": int((truth & predictions).sum()),
        },
        "by_defect": per_kind,
    }


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    train, cal, test = demo_splits()
    model = Detector.fit(train, cal)
    baseline = Detector.fit(train, cal, rgb_only=True)
    model.training_domain = baseline.training_domain = "synthetic woven texture"
    model.save(output / "model.npz")
    baseline.save(output / "rgb_baseline.npz")
    export_onnx(model, output / "scorer.onnx")
    session = ort.InferenceSession(str(output / "scorer.onnx"), providers=["CPUExecutionProvider"])
    features = np.stack([extract(sample(seed, kind)[0]) for seed, kind in test[:12]])
    actual = session.run(None, {"features": features})[0]
    expected = np.stack([model.patch_scores(sample(seed, kind)[0]) for seed, kind in test[:12]])
    parity = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"ONNX scoring parity failed: max error {parity}")
    # Check encoded image identity as well as disjoint seeds.
    groups = [train, cal, [sample(seed, kind)[0] for seed, kind in test]]
    hashes = [[hashlib.sha256(im.tobytes()).hexdigest() for im in group] for group in groups]
    if any(set(hashes[i]) & set(hashes[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Dataset content leakage")
    result = {
        "dataset": "Procedural woven texture v1; synthetic demonstration only",
        "seed_ranges": {"train": [1000, 1079], "calibration": [2000, 2059], "test": [3000, 3159]},
        "split_counts": {"train_normal": len(train), "calibration_normal": len(cal), "test": len(test)},
        "threshold_policy": "97.5th percentile of normal calibration image scores; frozen for all tests",
        "pixel_policy": "99.5th percentile of normal calibration patch scores",
        "model_id": model.model_id,
        "onnx_max_abs_error": parity,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": "runtime default",
        },
        "results": {},
    }
    for shift in ("clean", "bright", "blur"):
        items = [(kind, *sample(seed, kind, shift)) for seed, kind in test]
        result["results"][shift] = metrics(model, items)
    result["results"]["rgb_only_clean"] = metrics(
        baseline, [(kind, *sample(seed, kind)) for seed, kind in test]
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "split_manifest.json").write_text(
        json.dumps(dict(zip(("train", "calibration", "test"), hashes)), indent=2) + "\n"
    )
    # A compact evidence image generated from actual predictions, not a design mockup.
    board = Image.new("RGB", (SIZE * 4, SIZE * 2 + 56), "#101e2c")
    draw = ImageDraw.Draw(board)
    for idx, kind in enumerate(("normal", "scratch", "stain", "hole")):
        image, _ = sample(3000 + idx, kind)
        p = model.patch_scores(image)
        board.paste(image, (SIZE * idx, 22))
        board.paste(heatmap(image, p, model.pixel_threshold), (SIZE * idx, SIZE + 42))
        draw.text((SIZE * idx + 8, 5), kind, fill="white")
        draw.text((SIZE * idx + 8, SIZE * 2 + 45), f"score {model.image_score(p):.2f}", fill="white")
    board.save(output / "inspection_examples.png")
    lines = [
        "# Measured results",
        "",
        result["dataset"],
        "",
        "Thresholds were fitted once on 60 normal calibration images. Test data were never used for fitting.",
        "",
        "| Condition | Image AUROC | F1 | Normal FPR | Pixel AUROC | Defect IoU | p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["results"].items():
        lines.append(
            f"| {name} | {row['image_auroc']:.4f} | {row['f1']:.4f} | "
            f"{row['normal_false_positive_rate']:.4f} | {row['pixel_auroc']:.4f} | "
            f"{row['defect_mean_iou']:.4f} | {row['latency_ms_p50']:.2f} |"
        )
    lines += [
        "",
        f"ONNX scoring-head maximum absolute difference: {parity:.8f}.",
        "",
        "Binary metrics use score > threshold. The UI adds a separate review band and capture-quality warnings.",
        "Latency covers Python preprocessing and scoring, excludes network/encoding, and is environment-specific.",
        "Stress tests reuse test image identities to isolate degradation effects; they are not independent datasets.",
        "Bootstrap intervals in metrics.json resample images and do not measure real-world domain uncertainty.",
        "No real industrial benchmark or production validation has been performed.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result
