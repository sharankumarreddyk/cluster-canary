# Multi-stage build for the leaky-flask synthetic-data workload.
# Target final image size <80MB.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
RUN pip install --prefix=/install --no-cache-dir \
    "flask>=3.0,<4.0" \
    "gunicorn>=22.0" \
    "prometheus-client>=0.21"

FROM base AS runtime
COPY --from=deps /install /usr/local
WORKDIR /app
COPY app.py /app/app.py
RUN useradd -m -u 1000 leaky && chown -R leaky:leaky /app
USER leaky
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD curl -fs http://localhost:5000/healthz || exit 1
# 1 worker keeps the in-process leak observable; we use Deployment replicas for scale.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", \
     "--access-logfile", "-", "--error-logfile", "-", "app:app"]
