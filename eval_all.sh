#!/bin/sh

source ~/.profile

./scripts/train/run_film_humans.sh 256
./scripts/train/run_film_humans.sh 512
./scripts/train/run_film_humans.sh 1024
./scripts/train/run_film_humans.sh 2048
./scripts/train/run_film_humans.sh 4096
