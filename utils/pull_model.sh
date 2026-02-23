#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
SERVICE_NAME="ollama"

usage() {
  echo "Usage: $(basename "$0") <model-name>"
  echo "Example: $(basename "$0") llama3.2"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_NAME="${1:-}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Error: model name is required."
  usage
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: docker-compose.yml not found at $COMPOSE_FILE"
  exit 1
fi

if [[ -z "$(docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME")" ]]; then
  echo "Error: service '$SERVICE_NAME' is not running."
  echo "Start it first with: docker compose -f "$COMPOSE_FILE" up -d"
  exit 1
fi

echo "Pulling model '$MODEL_NAME' in container service '$SERVICE_NAME'..."
docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" sh -lc "OLLAMA_HOST=http://127.0.0.1:11434 ollama pull '$MODEL_NAME'"
echo "Model '$MODEL_NAME' pulled successfully."
