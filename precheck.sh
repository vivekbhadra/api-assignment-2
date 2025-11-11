#!/bin/bash
set -e
set -o pipefail

AWS_REGION="eu-west-2"
ACCOUNT_ID="402691950139"
REPO_NAME="smartlegal-app"
CLUSTER_NAME="smartlegal-cluster-v2"
DEPLOYMENT_NAME="smartlegal-deployment"
NAMESPACE="smartlegal"

echo "=============================================="
echo " SMARTLEGAL PRE-DEPLOYMENT VALIDATION SCRIPT"
echo "=============================================="
echo "Region: ${AWS_REGION}"
echo "Cluster: ${CLUSTER_NAME}"
echo "Repo: ${REPO_NAME}"
echo

pass() { echo -e "$1"; }
fail() { echo -e "$1"; exit 1; }

# Docker
echo "[1/8] Checking Docker..."
if command -v docker &>/dev/null && docker ps &>/dev/null; then
  pass "Docker is installed and running."
else
  fail "Docker not running or not installed."
fi

# AWS CLI
echo "[2/8] Checking AWS CLI..."
if command -v aws &>/dev/null; then
  pass "AWS CLI found: $(aws --version | head -n1)"
else
  fail "AWS CLI not installed."
fi

# kubectl
echo "[3/8] Checking kubectl..."
if command -v kubectl &>/dev/null; then
  pass "kubectl found: $(kubectl version --client --short)"
else
  fail "kubectl not installed."
fi

# AWS Authentication
echo "[4/8] Verifying AWS credentials..."
if aws sts get-caller-identity --region ${AWS_REGION} &>/dev/null; then
  ACCOUNT=$(aws sts get-caller-identity --query "Account" --output text)
  pass "Authenticated with AWS account: ${ACCOUNT}"
else
  fail "AWS authentication failed. Run 'aws configure'."
fi

# ECR Repository
echo "[5/8] Checking ECR repository..."
if aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${AWS_REGION} &>/dev/null; then
  pass "ECR repository '${REPO_NAME}' exists."
else
  echo "Repository not found. Creating..."
  aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} >/dev/null
  pass "Created new ECR repository '${REPO_NAME}'."
fi

# EKS Cluster
echo "[6/8] Checking EKS cluster connectivity..."
if aws eks describe-cluster --name ${CLUSTER_NAME} --region ${AWS_REGION} &>/dev/null; then
  pass "EKS cluster '${CLUSTER_NAME}' exists."
  aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME} >/dev/null
  kubectl get nodes >/dev/null && pass "Cluster nodes reachable."
else
  fail "EKS cluster '${CLUSTER_NAME}' not found."
fi

# Kubernetes Namespace and Secret
echo "[7/8] Checking Kubernetes namespace and secret..."
if kubectl get ns ${NAMESPACE} &>/dev/null; then
  pass "Namespace '${NAMESPACE}' exists."
else
  echo "Creating namespace '${NAMESPACE}'..."
  kubectl create namespace ${NAMESPACE}
  pass "Namespace created."
fi

if kubectl get secret smartlegal-env -n ${NAMESPACE} &>/dev/null; then
  pass "Secret 'smartlegal-env' found in '${NAMESPACE}'."
else
  echo "Secret missing. Checking local .env..."
  if [ -f .env ]; then
    source .env
    kubectl create secret generic smartlegal-env \
      --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
      --from-literal=GEMINI_API_KEY="${GEMINI_API_KEY}" \
      -n ${NAMESPACE}
    pass "Secret created from local .env file."
  else
    fail "No .env file found and no existing secret."
  fi
fi

# Deployment Check
echo "[8/8] Checking deployment status..."
if kubectl get deployment ${DEPLOYMENT_NAME} -n ${NAMESPACE} &>/dev/null; then
  pass "Deployment '${DEPLOYMENT_NAME}' exists."
else
  echo "Deployment not found yet — will be created on first deploy."
fi

echo
echo "=============================================="
echo "ALL PRECHECKS PASSED — YOU'RE GOOD TO DEPLOY"
echo "=============================================="

