# Measured results

Procedural woven texture v1; synthetic demonstration only

Thresholds were fitted once on 60 normal calibration images. Test data were never used for fitting.

| Condition | Image AUROC | F1 | Normal FPR | Pixel AUROC | Defect IoU | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.0000 | 0.0000 | 0.9984 | 0.1704 | 2.05 |
| bright | 0.9487 | 0.8571 | 1.0000 | 0.9362 | 0.0077 | 1.97 |
| blur | 1.0000 | 0.8571 | 1.0000 | 0.9983 | 0.0088 | 1.93 |
| rgb_only_clean | 1.0000 | 1.0000 | 0.0000 | 0.9987 | 0.2129 | 0.41 |

ONNX scoring-head maximum absolute difference: 0.00001907.

Binary metrics use score > threshold. The UI adds a separate review band and capture-quality warnings.
Latency covers Python preprocessing and scoring, excludes network/encoding, and is environment-specific.
Stress tests reuse test image identities to isolate degradation effects; they are not independent datasets.
Bootstrap intervals in metrics.json resample images and do not measure real-world domain uncertainty.
No real industrial benchmark or production validation has been performed.
