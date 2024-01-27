#!/bin/bash

PART="2"
DATASET="reddit"
PART_METHOD="metis"
PART_OBJ="vol"
MISSION="induc"
PART_CONFIG="partitions/${DATASET}-${PART}-${PART_METHOD}-${PART_OBJ}-${MISSION}/${DATASET}-${PART}-${PART_METHOD}-${PART_OBJ}-${MISSION}.json"


WORKSPACE="/nfsroot/GraphES"
IP_CONFIG="ip_config.txt"

LOGDIR="${WORKSPACE}/log_result/${DATASET}.${PART_METHOD}.${PART}.part"

cd ${WORKSPACE} && /home/yp/.conda/envs/dgs/bin/python launch.py \
    --workspace ${WORKSPACE} \
    --num_trainers 1 \
    --num_servers 1 \
    --part_config ${PART_CONFIG} \
    --ip_config ${IP_CONFIG} \
    --log_dir ${LOGDIR} \
    " /home/yp/.conda/envs/dgs/bin/python train.py \
    --world_size ${PART} \
    --device ${PART} \
    --dataset ${DATASET} \
    --part_config ${PART_CONFIG} \
    --n-hidden 512 \
    --n-layers 4 \
    --lr 0.001 \
    --dropout 0.1 \
    --n-epoch 3000 \
    --inductive \
    --eval "
