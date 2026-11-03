"""CLI entry points for repeatable demo and external-image experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import disjoint, load_folder
from .evaluate import metrics, run_demo
from .export import export_onnx
from .features import SIZE
from .model import Detector


def main():
    parser = argparse.ArgumentParser(prog="patchguard")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Fit, evaluate, export and save the synthetic demonstration")
    demo.add_argument("--output", type=Path, default=Path("artifacts"))
    fit = sub.add_parser("train", help="Train on your normal images")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, default=Path("artifacts/custom.npz"))
    evaluate = sub.add_parser("evaluate", help="Evaluate a frozen model on folder data")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/custom_metrics.json"))
    args = parser.parse_args()
    if args.command == "demo":
        result = run_demo(args.output)
        print(json.dumps({"model_id": result["model_id"], "results": result["results"]}, indent=2))
        return
    train = load_folder(args.data / "train" / "normal")
    calibration = load_folder(args.data / "calibration" / "normal")
    if args.command == "train":
        disjoint(train, calibration)
        model = Detector.fit([r.image for r in train], [r.image for r in calibration])
        model.save(args.output)
        export_onnx(model, args.output.with_suffix(".onnx"))
        manifest = {
            "model_id": model.model_id,
            "train": sorted({r.digest for r in train}),
            "calibration": sorted({r.digest for r in calibration}),
        }
        args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Saved model {model.model_id} to {args.output}")
        return
    model = Detector.load(args.model)
    manifest = json.loads(args.model.with_suffix(".manifest.json").read_text())
    if (
        manifest["model_id"] != model.model_id
        or set(manifest["train"]) != {r.digest for r in train}
        or set(manifest["calibration"]) != {r.digest for r in calibration}
    ):
        raise ValueError("Dataset or model differs from the training manifest")
    test = load_folder(args.data / "test")
    disjoint(train, calibration, test)
    items = []
    for record in test:
        kind = record.path.relative_to(args.data / "test").parts[0]
        if kind == "normal":
            mask = np.zeros((SIZE, SIZE), dtype=bool)
        else:
            mask_path = args.data / "masks" / kind / (record.path.stem + ".png")
            with Image.open(mask_path) as mask_image:
                if mask_image.size != record.image.size:
                    raise ValueError(f"Mask dimensions do not match {record.path}")
                mask = np.asarray(mask_image.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
                if not mask.any():
                    raise ValueError(f"Defect mask is empty: {mask_path}")
        items.append((kind, record.image, mask))
    if len({bool(mask.any()) for _, _, mask in items}) < 2:
        raise ValueError("Evaluation requires both normal and defective images")
    result = metrics(model, items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
