#!/bin/bash

cleanup() {
    echo -e "\n========================================[Stack Teardown] Catching termination signal...============================================="
    #echo "[Stack Teardown] Stopping Grafana Server..."
    #sudo systemctl stop grafana-server
    echo "=============================================[Stack Teardown] Stopping all services...==================================================="
    docker compose down --volumes --rmi all
    echo "=============================================[Stack Teardown] All services gracefully stopped.================================================="
    exit 0
}

trap cleanup SIGINT SIGTERM
# echo "Starting Observability Stack"
#echo "[1/2] Start Grafana"
#sudo systemctl start grafana-server
echo "==============================Starting Services============================"
docker compose up -d
cleanup
