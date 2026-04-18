# ============================================================================
# Dockerfile – AI-Powered HTTP Request Analyzer (BlueAgent WAF)
#
# Multi-stage build:
#   Stage 1 (builder) – install Python dependencies into a virtual-env.
#   Stage 2 (runtime) – slim image with only the venv + application code.
#
# Build:
#   docker build -t blueagent .
#
# Run (pass .env at runtime – never bake secrets into the image):
#   docker run --env-file .env -p 5000:5000 blueagent
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 – Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time deps (hiredis C extension needs a compiler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

LABEL maintainer="BlueAgent"
LABEL description="AI-Powered HTTP Request Analyzer – Defence-in-Depth WAF"

# Non-root user for security
RUN groupadd -r waf && useradd -r -g waf -d /app -s /sbin/nologin waf

WORKDIR /app

# Copy virtual-env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY app.py orchestrator.py ./
COPY config/ config/
COPY nodes/ nodes/
COPY schema/ schema/
COPY utils/ utils/
COPY workers/ workers/
COPY static/ static/

# Create writable directories for runtime data (SQLite DB + JSONL logs)
RUN mkdir -p data logs && chown -R waf:waf /app

# Switch to non-root
USER waf

EXPOSE 5000

# Health check against the Swagger docs endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/docs')" || exit 1

# Default: start the Flask API server
CMD ["python", "app.py"]
