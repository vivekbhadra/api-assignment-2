#!/bin/bash
set -e  # stop if any command fails
set -o pipefail

# ===== CONFIG =====
AWS_REGION="eu-west-2"
ACCOUNT_ID="402691950139"
REPO_NAME="smartlegal-app"
CLUSTER_NAME="smartlegal-cluster-v2"
DEPLOYMENT_NAME="smartlegal-deployment"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "=== SmartLegal Automated Build + Deploy ==="
echo "Region: ${AWS_REGION}"
echo "Cluster: ${CLUSTER_NAME}"
echo "ECR Repo: ${ECR_URI}"
echo

# ===== Pre-flight check: Docker daemon accessibility =====
echo "[0/6] Checking Docker daemon..."

DOCKER_CMD="docker"

if ! ${DOCKER_CMD} info >/dev/null 2>&1; then
    echo "Docker daemon not accessible as current user. Trying with sudo..."
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        echo "Using sudo for Docker commands."
    else
        echo "ERROR: Docker daemon not running or permission denied."
        echo "Fix: run 'sudo snap start docker' or ensure your user is in the 'docker' group:"
        echo "  sudo usermod -aG docker \$USER && newgrp docker"
        echo "Then re-run this script."
        exit 1
    fi
else
    echo "Docker daemon is accessible."
fi

# ===== Authenticate to AWS ECR =====
echo "[1/6] Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | ${DOCKER_CMD} login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# ===== Build Docker image =====
echo "[2/6] Building Docker image..."
${DOCKER_CMD} build -t ${REPO_NAME}-optimised .

# ===== Tag and push image =====
echo "[3/6] Pushing image to ECR..."
${DOCKER_CMD} tag ${REPO_NAME}-optimised:latest ${ECR_URI}
${DOCKER_CMD} push ${ECR_URI}

# ===== Update kubeconfig (ensure kubectl connected to correct cluster) =====
echo "[4/6] Updating kubeconfig for EKS cluster..."
aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

# ===== Restart deployment =====
echo "[5/6] Restarting Kubernetes deployment..."
kubectl rollout restart deployment ${DEPLOYMENT_NAME}
echo "Waiting for pods to be ready..."
kubectl rollout status deployment/${DEPLOYMENT_NAME} --timeout=600s

# ===== Verify pods and service =====
echo "[6/6] Verifying pods and service..."
kubectl get pods -o wide
kubectl get svc smartlegal-service

echo
echo "=== Deployment Completed Successfully ==="

