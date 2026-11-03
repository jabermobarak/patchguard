# Model card

- **Model:** spatial shrinkage Gaussian over 11 fixed patch descriptors.
- **Use:** research and local visual inspection experiments with stable camera views.
- **Training:** 80 normal synthetic texture images; no gradient-based training or pretrained network.
- **Calibration:** 60 independent normal synthetic images.
- **Test:** 160 independent synthetic images plus paired brightness and blur diagnostic conditions.
- **Outputs:** anomaly distance, heatmap, coarse boxes, capture warnings, and review decision. No defect subtype prediction.
- **Artifacts:** `model.npz`, color-only ablation model, and an ONNX scoring head. No Python pickle deserialization.
- **Measurements:** raw machine-readable results, fixed-threshold confusion matrices, per-defect detection counts, image and pixel AUROC, IoU, latency, coverage, accepted errors, and bootstrap intervals.
- **Limits:** narrow synthetic data, alignment assumptions, lighting/blur sensitivity, coarse masks, no industrial validation. The synthetic ablation does not justify choosing the full feature model over RGB alone.
- **Human review:** the system routes cases; no reviewer accuracy or defect correction has been measured. Zero accepted errors with zero coverage is not an effective automated detector.
- **Privacy:** the service processes uploaded images in memory. This does not cover logs or telemetry added by deployment infrastructure.
- **Deployment:** local demo by default. Production integration requires environment-specific data validation, security, load testing, and operating-point acceptance.

## Evidence provenance

Committed metrics are actual runs of `python -m patchguard.cli demo`. The generator is deterministic; timing and final floating-point digits depend on hardware and dependencies. The artifact image shows actual detector scores. The UI screenshot, when present, shows the running local application.
