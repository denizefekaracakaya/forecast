# ─────────────────────────────────────────────
# Builder — compiles deps that need build-essential (prophet/cmdstanpy)
# into a venv; this stage's toolchain never reaches the runtime images.
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Runtime base — slim, no compiler, non-root user
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime-base

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=appuser:appuser . .
USER appuser

# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
FROM runtime-base AS app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ─────────────────────────────────────────────
# FastAPI Gateway
# ─────────────────────────────────────────────
FROM runtime-base AS gateway
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "forecast.gateway.app:app", "--host=0.0.0.0", "--port=8000"]

# ─────────────────────────────────────────────
# Ingestion Worker
# ─────────────────────────────────────────────
FROM runtime-base AS ingestion
CMD ["python", "-m", "forecast.ingestion.worker"]
