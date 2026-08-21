#!/usr/bin/env bash
# Build and live-serve the docs at http://localhost:8000.
# MkDocs rebuilds changed pages and reloads connected browsers automatically.
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
cd "$DOCS_DIR"

VENV="$REPO_ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtualenv at $VENV"
    if command -v python3.10 >/dev/null 2>&1; then
        python3.10 -m venv "$VENV"
    else
        python3 -m venv "$VENV"
    fi
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

python -m pip install \
    --quiet \
    --disable-pip-version-check \
    -r "$REPO_ROOT/requirements.txt"

exec python -m mkdocs serve \
    --dev-addr 127.0.0.1:8000 \
    --watch "$DOCS_DIR/src" \
    --watch "$DOCS_DIR/mkdocs.yml"
