#!/bin/bash
set -e
set -o pipefail

REPO_NAME="smartlegal-app-optimised"

echo "=== SmartLegal: Docker Build Only ==="
echo "Image name: ${REPO_NAME}:latest"
echo

# --- Check Docker daemon ---
echo "[1/2] Checking Docker daemon..."
DOCKER_CMD="docker"

if ! ${DOCKER_CMD} info >/dev/null 2>&1; then
    echo "Docker not accessible as current user. Trying sudo..."
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        echo "Using sudo for Docker commands."
    else
        echo "ERROR: Docker daemon not running or insufficient permissions."
        echo "Fix: ensure docker is running or add your user to the docker group:"
        echo "  sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi
else
    echo "Docker daemon is accessible."
fi

# --- Build Image ---
echo "[2/2] Building Docker image..."
${DOCKER_CMD} build -t ${REPO_NAME}:latest .

echo
echo "=== Docker Image Built Successfully ==="
echo "Run locally using:"
echo "  ${DOCKER_CMD} run -p 8501:8501 ${REPO_NAME}:latest"

