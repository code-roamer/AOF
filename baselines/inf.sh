#!/bin/bash

gpuid=$1
data=$2
prefix=$3
victim_models=("pointnet" "pointnet2" "pointconv" "dgcnn")


for j in ${!victim_models[@]}; do
    CUDA_VISIBLE_DEVICES=$gpuid python inference.py --model=${victim_models[$j]} --batch_size=16 --data_root=$data --prefix=$prefix
done