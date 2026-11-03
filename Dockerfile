FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY patchguard ./patchguard
RUN pip install --no-cache-dir . && useradd --create-home inspector
COPY artifacts/model.npz ./artifacts/model.npz
USER inspector
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "patchguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
