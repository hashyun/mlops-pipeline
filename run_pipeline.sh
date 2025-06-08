#!/bin/bash
# Simple helper to set up environment and run the MLOps pipeline via docker compose.

set -e

ENV_FILE="src/config/.env"
EXAMPLE_FILE="src/config/.env.example"

if [ ! -f "$ENV_FILE" ]; then
    echo "No $ENV_FILE found. Creating one from the example template..."
    if [ ! -f "$EXAMPLE_FILE" ]; then
        echo "Example env file $EXAMPLE_FILE not found. Please create $ENV_FILE manually." >&2
        exit 1
    fi
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    echo "Copied $EXAMPLE_FILE to $ENV_FILE. Please edit it to add your API keys before continuing."
fi

# Launch the containers
if ! command -v docker >/dev/null; then
    echo "Docker is not installed. Please install Docker before running this script." >&2
    exit 1
fi

docker compose up -d
