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

# ===== Authenticate to AWS ECR =====
echo "[1/6] Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# ===== Build Docker image =====
echo "[2/6] Building Docker image..."
docker build -t ${REPO_NAME}-optimised .

# ===== Tag and push image =====
echo "[3/6] Pushing image to ECR..."
docker tag ${REPO_NAME}-optimised:latest ${ECR_URI}
docker push ${ECR_URI}

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

