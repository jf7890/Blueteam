#!/usr/bin/env bash
set -euo pipefail

REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

redis_cmd() {
  redis-cli -u "$REDIS_URL" "$@"
}

delete_pattern() {
  local pattern="$1"
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    redis_cmd DEL "$key" >/dev/null
  done < <(redis_cmd --scan --pattern "$pattern")
}

delete_pattern 'waf:verdict:*'
delete_pattern 'waf:result:*'
redis_cmd DEL 'waf:queue:llm_analysis' >/dev/null
redis_cmd DEL 'waf:queue:llm_analysis_dlq' >/dev/null

echo "Benchmark cache cleared for $REDIS_URL"
