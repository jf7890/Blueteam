# ============================================================================
# Dockerfile – http.ingest (Go)
#
# Lightweight HTTP mirror receiver that captures raw traffic and forwards
# it as JSON to the BlueAgent WAF /api/analyze endpoint.
#
# Build:
#   docker build -t http-ingest ./http.ingest
#
# Run:
#   docker run -e AI_URL=http://blueagent:5000/api/analyze -p 9002:9002 http-ingest
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 – Build
# ---------------------------------------------------------------------------
FROM golang:1.23-alpine AS builder

WORKDIR /src
COPY go.mod ./
# No external deps – only stdlib
RUN go mod download

COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /http-ingest .

# ---------------------------------------------------------------------------
# Stage 2 – Runtime (scratch for minimal image)
# ---------------------------------------------------------------------------
FROM alpine:3.20

RUN apk add --no-cache ca-certificates

COPY --from=builder /http-ingest /usr/local/bin/http-ingest

# Non-root user
RUN addgroup -S ingest && adduser -S -G ingest ingest
USER ingest

EXPOSE 9002

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:9002/healthz || exit 1

ENTRYPOINT ["http-ingest"]
