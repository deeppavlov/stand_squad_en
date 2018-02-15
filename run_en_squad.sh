#!/bin/bash
if [ ! -d "./log_char_300_pretrained_train_full_eng" ]; then
    echo downloading model files..
    wget -r http://lnsigo.mipt.ru/export/deepreply_data/stand_squad_en/log_char_300_pretrained_train_full_eng.tar.gz &&
    tar -zxvf log_char_300_pretrained_train_full_eng.tar.gz &&
    rm log_char_300_pretrained_train_full_eng.tar.gz &&
    echo download successful
fi &&
#export CUDA_VISIBLE_DEVICES=2 &&
source ../../virtualenv/en_squad/bin/activate &&
#source ../virtualenv/en_squad/bin/activate &&
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/ &&
nohup python3 squad_en_api.py > ./squad_en.log &
