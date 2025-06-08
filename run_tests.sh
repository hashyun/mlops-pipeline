#!/bin/bash
set -e
python -m py_compile $(git ls-files '*.py')
if command -v docker >/dev/null; then
    docker compose config >/dev/null
else
    echo "Skipping docker compose config check - docker not found." >&2
fi
