# Data contract

## Included demonstration

The repository contains original procedural textures and their fitted statistics, not images from an employer or private dataset. The generator is in `patchguard/data.py`.

| Split | Seeds | Count | Contents |
|---|---|---:|---|
| Train | 1000–1079 | 80 | Normal only |
| Calibration | 2000–2059 | 60 | Normal only |
| Test | 3000–3159 | 160 | 40 normal, 40 scratches, 40 stains, 40 holes |

Independent seeds and decoded pixel hashes are checked. Brightness and blur stress conditions reuse the test images, with changed capture appearance and unchanged ground truth. They are paired diagnostic conditions, not additional independent samples.

The distribution is narrow and visibly synthetic. Defects are generated with simple drawing operations. The benchmark is intended to exercise the complete pipeline and reveal known weaknesses, not estimate performance on a new factory or product.

## External folder layout

| Path | Contents |
|---|---|
| `train/normal/*.png` | At least 20 normal fitting images |
| `calibration/normal/*.png` | At least 20 distinct normal threshold-calibration images |
| `test/normal/*.png` | Untouched normal test images |
| `test/<defect-name>/*.png` | Untouched defect test images |
| `masks/<defect-name>/<image-stem>.png` | Binary mask matching each defect image's original dimensions |

All paths are relative to the directory passed as `--data`. Supported image extensions are `.png`, `.jpg`, `.jpeg`, and `.webp`. Masks are PNG, with zero for background and nonzero for defects. Each defect mask must contain at least one defect pixel after resizing. Image stems must be unique within a defect group.

Split by physical item and capture session before putting images into these folders. Do not split crops from the same source image between train and test. Keep defect labels and masks out of training and calibration. Test must contain both normal and defective images to compute AUROC.

Decoded-pixel hashes detect exact copies across splits. The training command saves these hashes beside the model; evaluation rejects changed training/calibration content or a mismatched model manifest. Near-duplicates and physical-item leakage remain the dataset owner's responsibility.

External images are not downloaded, bundled, or licensed by this project. Use a dataset you have permission to use and retain its provenance outside the code license.
