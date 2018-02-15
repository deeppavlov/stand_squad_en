#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 &&
source ./env/bin/activate &&
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/ &&
nohup python3 squad_en_api.py > ./squad_en.log &
