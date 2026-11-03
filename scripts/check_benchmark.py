"""Regression gates for the synthetic fixture, not claims of industrial accuracy."""

import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
clean = report["results"]["clean"]
assert report["onnx_max_abs_error"] < 0.001
assert clean["image_auroc"] > 0.95
assert clean["normal_false_positive_rate"] <= 0.1
assert report["results"]["bright"]["route_counts"]["review"] >= 150
assert report["results"]["blur"]["route_counts"]["review"] >= 150
print("Synthetic benchmark regression gates passed")
