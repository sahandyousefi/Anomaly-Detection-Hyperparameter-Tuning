# syntax=docker/dockerfile:1
# Scoring service image. The model artifact is not baked in; mount the
# directory written by `fraud-detection train` (artifacts/model/artifact) at
# /models/artifact, as docker-compose.yml does.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[serve]"

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FRAUD_DETECTION_ARTIFACT=/models/artifact \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba
# XGBoost links against the system OpenMP runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links /wheels "fraud-detection[serve]" \
    && rm -rf /wheels
RUN useradd --create-home --uid 10001 scoring
USER scoring
WORKDIR /home/scoring
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "fraud_detection.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
