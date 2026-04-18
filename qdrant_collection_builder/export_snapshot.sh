#!/usr/bin/env bash
set -euo pipefail

# Create a Qdrant snapshot and copy it out of the container.
#
# Usage:
#   bash qdrant_collection_builder/export_snapshot.sh [collection] [container] [output_dir] [qdrant_url]
#
# Defaults:
#   collection = waf_payloads
#   container  = waf-qdrant
#   output_dir = ./qdrant_snapshots
#   qdrant_url = http://localhost:6333

COLLECTION="${1:-waf_payloads}"
CONTAINER="${2:-waf-qdrant}"
OUT_DIR="${3:-./qdrant_snapshots}"
QDRANT_URL="${4:-http://localhost:6333}"

mkdir -p "$OUT_DIR"

echo "Creating snapshot for collection '$COLLECTION' via '$QDRANT_URL'..."
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  RAW_JSON="$(curl -sS -X POST -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections/$COLLECTION/snapshots")"
else
  RAW_JSON="$(curl -sS -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots")"
fi

SNAPSHOT_NAME="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("result",{}).get("name",""))' <<< "$RAW_JSON")"

if [[ -z "$SNAPSHOT_NAME" ]]; then
  echo "Failed to parse snapshot name from response:" >&2
  echo "$RAW_JSON" >&2
  exit 1
fi

SRC_A="/qdrant/storage/snapshots/$COLLECTION/$SNAPSHOT_NAME"
SRC_B="/qdrant/snapshots/$COLLECTION/$SNAPSHOT_NAME"

TARGET="$OUT_DIR/$SNAPSHOT_NAME"

if docker exec "$CONTAINER" test -f "$SRC_A"; then
  docker cp "$CONTAINER:$SRC_A" "$TARGET"
elif docker exec "$CONTAINER" test -f "$SRC_B"; then
  docker cp "$CONTAINER:$SRC_B" "$TARGET"
else
  echo "Snapshot file not found in known paths:" >&2
  echo "  $SRC_A" >&2
  echo "  $SRC_B" >&2
  exit 1
fi

echo "Snapshot exported to: $TARGET"
