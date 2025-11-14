#!/bin/bash

DEPLOYMENT="smartlegal-deployment"
SERVICE="smartlegal-service"

start_deployment() {
    echo "Starting SmartLegal deployment..."
    kubectl scale deployment ${DEPLOYMENT} --replicas=1
    echo "Deployment started. Checking status..."
    sleep 3
    kubectl get pods -o wide
}

stop_deployment() {
    echo "Stopping SmartLegal deployment..."
    kubectl scale deployment ${DEPLOYMENT} --replicas=0
    echo "Deployment stopped. Checking status..."
    sleep 2
    kubectl get pods -o wide
}

check_status() {
    echo "Checking service and deployment status..."
    echo

    echo "--- Deployment replicas ---"
    kubectl get deployment ${DEPLOYMENT} -o jsonpath='{.status.replicas}' 2>/dev/null || echo "Deployment not found"
    echo

    echo "--- Pods ---"
    kubectl get pods -o wide
    echo

    echo "--- Service ---"
    kubectl get svc ${SERVICE}
    echo
}

case "$1" in
  start)
    start_deployment
    ;;

  stop)
    stop_deployment
    ;;

  status)
    check_status
    ;;

  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac

