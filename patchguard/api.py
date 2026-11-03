"""Local inspection service. Uploads are processed in memory and never retained."""

import base64
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .data import KINDS, sample
from .features import MAX_BYTES, decode_image
from .model import Detector, heatmap

STATIC = Path(__file__).parent / "static"


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("PATCHGUARD_MODEL", "artifacts/model.npz"))
    counters = {"requests": 0, "anomaly": 0, "review": 0, "normal": 0, "latency_sum_ms": 0.0}
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        # Fail explicitly rather than silently train a different model during startup.
        app.state.detector = Detector.load(path)
        yield

    app = FastAPI(title="PatchGuard", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": app.state.detector.model_id}

    @app.get("/api/model")
    def model_info():
        detector = app.state.detector
        return {
            "model_id": detector.model_id,
            "threshold": detector.threshold,
            "pixel_threshold": detector.pixel_threshold,
            "features": "RGB means only" if detector.rgb_only else "RGB moments and texture gradients",
            "training_domain": detector.training_domain,
            "scoring": "spatial shrinkage Gaussian",
            "image_size": 128,
            "review_band": [0.9, 1.1],
        }

    @app.get("/api/examples/{kind}")
    def example(kind: str):
        if kind not in KINDS:
            raise HTTPException(404, "Unknown example")
        image, _ = sample(3000 + KINDS.index(kind), kind)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    def inspect_bytes(raw: bytes):
        start = time.perf_counter()
        try:
            image = decode_image(raw)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        detector = app.state.detector
        result = detector.infer(image)
        overlay = heatmap(image, detector.patch_scores(image), detector.pixel_threshold)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        result["overlay"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        with lock:
            counters["requests"] += 1
            counters[result["status"]] += 1
            counters["latency_sum_ms"] += result["latency_ms"]
        return result

    @app.post("/api/inspect")
    async def inspect(file: UploadFile):
        try:
            raw = await file.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise HTTPException(413, "Image exceeds the 8 MiB limit")
            return await run_in_threadpool(inspect_bytes, raw)
        finally:
            await file.close()

    @app.get("/metrics", include_in_schema=False)
    def prometheus():
        with lock:
            snapshot = counters.copy()
        lines = [
            "# TYPE patchguard_requests_total counter",
            f"patchguard_requests_total {snapshot['requests']}",
            "# TYPE patchguard_decisions_total counter",
        ]
        lines += [
            f'patchguard_decisions_total{{status="{status}"}} {snapshot[status]}'
            for status in ("normal", "anomaly", "review")
        ]
        lines += [
            "# TYPE patchguard_latency_ms_sum counter",
            f"patchguard_latency_ms_sum {snapshot['latency_sum_ms']:.4f}",
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


app = create_app()
