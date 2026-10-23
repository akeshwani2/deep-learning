#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/parkinsons-voice-uv-cache}"

uv sync --frozen --all-groups
uv run --frozen ruff check .
uv run --frozen pytest
uv run --frozen parkinsons-voice --help >/dev/null
