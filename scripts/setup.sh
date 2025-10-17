#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
uv pip install -r requirements.txt

# Configure repository-managed git hooks so pre-commit formatting runs automatically.
git config core.hooksPath .githooks
