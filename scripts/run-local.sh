#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
runtime_python="${ALTIMAP_PYTHON:-$repo_dir/.venv-da3/bin/python}"
if [[ ! -x "$runtime_python" ]]; then
  echo "Missing runtime Python: $runtime_python. See docs/surface-workspace.md."
  exit 1
fi
npm --prefix frontend run build
export ALTIMAP_WEB_DIR="$repo_dir/frontend/dist"
exec "$runtime_python" -m viewer.server --host "${ALTIMAP_HOST:-127.0.0.1}" --port "${ALTIMAP_PORT:-8080}"
