#!/bin/bash

cleanup() {
    echo -e "\n[Stack Teardown] Catching termination signal..."
    echo "[Stack Teardown] Stopping Grafana Server..."
    sudo systemctl stop grafana-server
    echo "[Stack Teardown] All services gracefully stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM
cd ..
echo "Starting Observability Stack"
echo "[1/2] Start Grafana"
sudo systemctl start grafana-server
echo "[2/2] Starting Honcho"
honcho start
cleanup