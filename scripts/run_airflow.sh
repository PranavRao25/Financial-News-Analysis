#! /bin/bash

docker compose up airflow-init
docker compose run airflow-worker airflow info --remove-orphans
