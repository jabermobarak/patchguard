import json
import sys

import pytest

from patchguard.cli import main
from patchguard.data import sample


def test_real_folder_workflow_and_changed_data_rejection(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    for split, start in [("train", 6000), ("calibration", 7000)]:
        folder = data / split / "normal"
        folder.mkdir(parents=True)
        for i in range(20):
            sample(start + i)[0].save(folder / f"{i}.png")
    for kind in ("normal", "stain"):
        folder = data / "test" / kind
        folder.mkdir(parents=True)
        (data / "masks" / kind).mkdir(parents=True)
        for i in range(3):
            image, mask = sample(8000 + i, kind)
            image.save(folder / f"{i}.png")
            from PIL import Image

            Image.fromarray(mask.astype("uint8") * 255).save(data / "masks" / kind / f"{i}.png")
    model_path = tmp_path / "custom.npz"
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys, "argv", ["patchguard", "train", "--data", str(data), "--output", str(model_path)]
    )
    main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["patchguard", "evaluate", "--data", str(data), "--model", str(model_path), "--output", str(output)],
    )
    main()
    assert json.loads(output.read_text())["n"] == 6
    sample(7777)[0].save(data / "calibration" / "normal" / "0.png")
    with pytest.raises(ValueError, match="differs"):
        main()
