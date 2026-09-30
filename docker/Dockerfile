# NOT BUILT OR RUN in the authoring environment (no Docker daemon available there).
# Build from the project root:   docker build -f docker/Dockerfile -t churn-api:1.0.0 .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHURN_MODEL_PATH=/models/model.joblib

# 1) Dependencies first: this layer is rebuilt only when requirements-serve.txt changes.
WORKDIR /app
COPY docker/requirements-serve.txt /app/requirements-serve.txt
RUN pip install -r requirements-serve.txt

# 2) Our code changes often, so it comes last. --no-deps: everything is already pinned above.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
RUN pip install --no-deps .

# 3) Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# The model is NOT baked into the image: mount it (see docker-compose.yml) so a new model
# does not need a new image.
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
