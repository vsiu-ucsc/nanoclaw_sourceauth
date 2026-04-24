#!/usr/bin/env bash
# benchmark/run_cross_model.sh
#
# Run the source-authorization benchmark across a set of models.
#   - Claude models use the native Anthropic SDK path in eval.py (no proxy).
#   - Non-Claude models are routed through a LiteLLM proxy we start on demand,
#     using benchmark/litellm_config.yaml.
#
# Usage:
#   benchmark/run_cross_model.sh                         # default model list
#   benchmark/run_cross_model.sh model1 model2 ...       # explicit list
#   MODELS="model1 model2" benchmark/run_cross_model.sh  # env-var list
#
# Environment variables consumed:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY,
#   MOONSHOT_API_KEY, FIREWORKS_API_KEY   (each populated on demand per model)
#   LITELLM_PORT   (default 4000)
#   WORKERS        (default 16 for Claude, 8 for proxied — see per-call below)

set -euo pipefail

# Repo root relative to this script
cd "$(dirname "$0")/.."

# ───────────────────────── model set ─────────────────────────
# Keep this in sync with benchmark/litellm_config.yaml for proxied models.
DEFAULT_MODELS=(
    claude-sonnet-4-6
    claude-opus-4-7
    claude-haiku-4-5-20251001
    gpt-5.4
    gemini-3.1-pro
    grok-4.20
    kimi-k2.5
    qwen-3.5-397b
)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: $0 [model1 model2 ...]

Runs benchmark/eval.py + benchmark/report.py across a set of models.
Claude models use the native Anthropic SDK path; others route through a
LiteLLM proxy we start on demand (benchmark/litellm_config.yaml).

Default model list:
$(printf '  - %s\n' "${DEFAULT_MODELS[@]}")

Environment variables:
  LITELLM_PORT  (default 4000)
  WORKERS       (default 16 for Claude, 8 for proxied providers)
  MODELS="a b"  alternative to positional args

Examples:
  $0                              # run the default set
  $0 claude-opus-4-7              # just one model
  $0 gpt-5.4 gemini-3.1-pro       # a couple of non-Claude
  MODELS="kimi-k2.5" $0           # via env
EOF
    exit 0
fi

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<< "$MODELS"
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

LITELLM_PORT="${LITELLM_PORT:-4000}"
PROXY_URL="http://localhost:${LITELLM_PORT}"
CONFIG_FILE="benchmark/litellm_config.yaml"
PROXY_LOG=".litellm-proxy.log"
PROXY_PID=""

# Load .env if present (Anthropic and provider keys)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
fi

# ───────────────────────── proxy helpers ─────────────────────────
needs_proxy() {
    for m in "${MODELS[@]}"; do
        [[ "$m" != claude* ]] && return 0
    done
    return 1
}

start_proxy() {
    if curl -sf --max-time 2 "${PROXY_URL}/health/readiness" > /dev/null 2>&1; then
        echo "LiteLLM proxy already running on ${PROXY_URL} — reusing."
        return 0
    fi

    if ! command -v litellm > /dev/null; then
        echo "ERROR: litellm not installed. Run: pip install 'litellm[proxy]'" >&2
        exit 1
    fi

    echo "Starting LiteLLM proxy on ${PROXY_URL} (log: ${PROXY_LOG})..."
    litellm --config "$CONFIG_FILE" --port "$LITELLM_PORT" > "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!

    # Wait up to 30s for health
    for _ in {1..30}; do
        if curl -sf --max-time 2 "${PROXY_URL}/health/readiness" > /dev/null 2>&1; then
            echo "LiteLLM proxy ready."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: LiteLLM proxy failed to come up in 30s. See ${PROXY_LOG}." >&2
    exit 1
}

stop_proxy() {
    if [[ -n "$PROXY_PID" ]]; then
        echo "Stopping LiteLLM proxy (pid=${PROXY_PID})..."
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
}
trap stop_proxy EXIT

# ───────────────────────── main loop ─────────────────────────
if needs_proxy; then
    start_proxy
fi

FAILED=()
for model in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running: ${model}"
    echo "════════════════════════════════════════════════════════════"

    if [[ "$model" == claude* ]]; then
        python3 benchmark/eval.py \
            --model "$model" \
            --workers "${WORKERS:-16}" \
        || { FAILED+=("$model"); continue; }
    else
        python3 benchmark/eval.py \
            --model "$model" \
            --workers "${WORKERS:-8}" \
            --base-url "$PROXY_URL" \
            --api-key "sk-proxy" \
        || { FAILED+=("$model"); continue; }
    fi

    python3 benchmark/report.py --model "$model" \
        || echo "WARN: report generation failed for $model (eval data is still in results/)"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done. Models run: ${#MODELS[@]}"
if (( ${#FAILED[@]} > 0 )); then
    echo "  FAILED: ${FAILED[*]}"
    echo "  (retry with: $0 ${FAILED[*]})"
fi
echo "  Reports: benchmark/reports/"
echo "════════════════════════════════════════════════════════════"
