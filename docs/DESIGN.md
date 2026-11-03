# Design

## Data flow

An image is EXIF-oriented, converted to RGB, and resized to 128 × 128. Eleven fixed features are computed on non-overlapping 8 × 8 patches: three RGB means, three RGB standard deviations, signed horizontal and vertical gradients, their absolute magnitudes, and an absolute high-frequency residual. The output is a 16 × 16 × 11 feature tensor.

For each patch location, fit a mean vector and covariance using only normal training images. Covariance is regularized as:

`C = 0.8 * empirical_covariance + 0.2 * diagonal(empirical_covariance) + 1e-5 * identity`

The score at position p is the Mahalanobis distance:

`d_p = sqrt(max((x_p - mean_p)^T inverse(C_p) (x_p - mean_p), 0))`

The image score is the mean of the four highest patch scores. The 97.5th percentile of held-out normal image scores becomes the image threshold. The 99.5th percentile of calibration patch scores becomes the region threshold. These are empirical operating points; there is no finite-sample false-positive guarantee.

## Review routing

The binary detector flags score > image threshold. UI routing is an additional policy:

1. Review if image score / threshold is in [0.9, 1.1].
2. Review if global brightness lies outside the calibration min/max with a fixed 0.06 margin.
3. Review if mean high-frequency residual lies outside [0.7 × calibration minimum, 1.3 × calibration maximum].
4. Otherwise output normal or anomaly using the frozen binary threshold.

These fixed quality-check margins are development heuristics. A change in texture can be a defect or a capture change; review does not identify the cause. Whole-image checks can miss local degradation or route genuine defects to review. Coverage and accepted errors are reported so review-all is not presented as successful automation.

## Region evidence

Patch scores exceeding the pixel threshold form a 4-connected mask. Each connected component produces a normalized bounding box. A bilinear upsample generates the heatmap and dense metric scores. Boxes describe coarse evidence, not exact segmentation. No defect-type classifier is fitted: “scratch”, “stain”, and “hole” name synthetic examples, not predicted categories.

## Export boundary

The ONNX graph takes a float32 batch of feature tensors and returns patch distances. The graph includes mean subtraction, per-location precision weighting, quadratic-form reduction, clamping, and square root. It does not include image decoding, feature extraction, image score aggregation, thresholds, or review routing. The NPZ model remains the service's source of truth; the ONNX file is a separately verified deployment option.

## Service

The FastAPI lifespan loads an explicit NPZ file with pickle disabled and fails if missing. Image decoding and inference run in a worker thread. Successful requests update aggregate counters under a lock. Images are not retained. The UI is served from the same origin without third-party fonts, scripts, analytics, or remote APIs.

Uploads are read up to 8 MiB + 1 byte after multipart parsing, and decoded images are limited to 12 megapixels. This application-level check does not replace a reverse proxy body limit, because multipart parsing precedes endpoint execution.

Model IDs hash fitted statistics, decision thresholds, and quality bounds. External fitting writes a manifest of decoded image hashes; evaluation verifies the model ID, training/calibration hashes, and cross-split duplicates before scoring. This detects exact pixel duplicates, not resized copies or shared physical objects.

## Deliberate tradeoffs

Fixed descriptors make the model small and CPU-friendly, but restrict semantic understanding and robustness. Spatial statistics assume aligned views. Resizing distorts aspect ratio and can erase tiny defects. Color-only ablation performs as well or better on this simple synthetic fixture; the full model is useful for studying sensitivity rather than demonstrating superiority. A next external experiment would compare it with learned patch features on a session-separated, real surface dataset.
