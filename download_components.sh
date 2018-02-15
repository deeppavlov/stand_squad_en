#!/bin/bash
echo downloading model files..
wget http://lnsigo.mipt.ru/export/deepreply_data/stand_squad_en/log_char_300_pretrained_train_full_eng.tar.gz &&
tar -zxvf log_char_300_pretrained_train_full_eng.tar.gz &&
rm log_char_300_pretrained_train_full_eng.tar.gz &&
echo download successful