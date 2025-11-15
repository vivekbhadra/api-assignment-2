#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIGURATION (modify if needed)
########################################
REGION="eu-west-2"
CLUSTER_NAME="smartlegal-cluster"
NAMESPACE="smartlegal"
ECR_REPO="smartlegal"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "====================================================="
echo " SmartLegal Full AWS Teardown Script"
echo " Region      : ${REGION}"
echo " Cluster     : ${CLUSTER_NAME}"
echo " Namespace   : ${NAMESPACE}"
echo " ECR Repo    : ${ECR_URI}"
echo "====================================================="
echo
read -p "Are you sure you want to remove EVERYTHING? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

########################################
# STEP 1: Delete Kubernetes resources
########################################
echo "Deleting Kubernetes resources..."

kubectl delete deployment smartlegal-deployment --namespace ${NAMESPACE} --ignore-not-found
kubectl delete service smartlegal-service --namespace ${NAMESPACE} --ignore-not-found
kubectl delete secret smartlegal-env --namespace ${NAMESPACE} --ignore-not-found
kubectl delete namespace ${NAMESPACE} --ignore-not-found

echo "Kubernetes resources removed."

########################################
# STEP 2: Delete Load Balancer created by the Service
########################################
echo "Checking for orphaned Load Balancers..."

LB_ARNS=$(aws elbv2 describe-load-balancers --region ${REGION} \
    --query "LoadBalancers[?contains(DNSName, 'elb')].LoadBalancerArn" --output text || true)

for LB_ARN in $LB_ARNS; do
    echo "Deleting Load Balancer: $LB_ARN"
    TG_ARNS=$(aws elbv2 describe-target-groups --region ${REGION} \
        --query "TargetGroups[?LoadBalancerArns[0]==\`${LB_ARN}\`].TargetGroupArn" --output text || true)

    for TG in $TG_ARNS; do
        echo "Deleting Target Group: $TG"
        aws elbv2 delete-target-group --target-group-arn "$TG" --region ${REGION} || true
    done

    aws elbv2 delete-load-balancer --load-balancer-arn "$LB_ARN" --region ${REGION} || true
done

echo "Load balancers removed."

########################################
# STEP 3: Delete CloudWatch log groups
########################################
echo "Deleting CloudWatch log groups..."

LOG_GROUPS=$(aws logs describe-log-groups \
    --log-group-name-prefix "/aws/eks/${CLUSTER_NAME}" \
    --query "logGroups[].logGroupName" --output text || true)

for LG in $LOG_GROUPS; do
    echo "Deleting Log Group: $LG"
    aws logs delete-log-group --log-group-name "$LG" || true
done

echo "CloudWatch log groups removed."

########################################
# STEP 4: Delete ECR images + repository
########################################
echo "Deleting ECR images and repository..."

aws ecr batch-delete-image \
    --repository-name ${ECR_REPO} \
    --image-ids $(aws ecr list-images --repository-name ${ECR_REPO} --query 'imageIds[*]' --output json) \
    --region ${REGION} || true

aws ecr delete-repository --repository-name ${ECR_REPO} --force --region ${REGION} || true

echo "ECR repository removed."

########################################
# STEP 5: Delete EKS cluster + nodegroups
########################################
echo
echo "Deleting EKS cluster and managed node groups..."

NODEGROUPS=$(aws eks list-nodegroups --cluster-name ${CLUSTER_NAME} --region ${REGION} \
    --query "nodegroups[]" --output text || true)

for NG in $NODEGROUPS; do
    echo "Removing Node Group: $NG"
    aws eks delete-nodegroup \
        --cluster-name ${CLUSTER_NAME} \
        --nodegroup-name $NG \
        --region ${REGION} || true
done

aws eks delete-cluster --name ${CLUSTER_NAME} --region ${REGION} || true

echo "EKS cluster removed."

########################################
# STEP 6: Delete the EKS VPC (Optional but recommended)
########################################
echo
read -p "Do you also want to delete the EKS-created VPC and subnets? (yes/no): " CONFIRM_VPC

if [[ "$CONFIRM_VPC" == "yes" ]]; then
    echo "Searching for EKS-owned VPC..."

    VPC_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} --region ${REGION} \
        --query "cluster.resourcesVpcConfig.vpcId" --output text || true)

    if [[ "$VPC_ID" != "None" && "$VPC_ID" != "" ]]; then
        echo "Detected VPC: $VPC_ID"

        echo "Detaching and deleting internet gateways..."
        IGW=$(aws ec2 describe-internet-gateways \
            --filters "Name=attachment.vpc-id,Values=$VPC_ID" \
            --query "InternetGateways[].InternetGatewayId" --output text || true)

        for G in $IGW; do
            aws ec2 detach-internet-gateway --internet-gateway-id $G --vpc-id $VPC_ID || true
            aws ec2 delete-internet-gateway --internet-gateway-id $G || true
        done

        echo "Deleting subnets..."
        SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
            --query "Subnets[].SubnetId" --output text)

        for S in $SUBNETS; do
            aws ec2 delete-subnet --subnet-id $S || true
        done

        echo "Deleting route tables..."
        ROUTES=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" \
            --query "RouteTables[].RouteTableId" --output text)

        for R in $ROUTES; do
            aws ec2 delete-route-table --route-table-id $R || true
        done

        echo "Deleting security groups..."
        SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" \
            --query "SecurityGroups[].GroupId" --output text)

        for G in $SG; do
            if [[ "$G" != "sg-*" ]]; then continue; fi
            aws ec2 delete-security-group --group-id $G || true
        done

        echo "Deleting VPC..."
        aws ec2 delete-vpc --vpc-id $VPC_ID || true
    fi
fi

echo "All resources removed successfully. You are no longer incurring AWS charges."
echo "====================================================="
echo "   SmartLegal Teardown Completed"
echo "====================================================="

