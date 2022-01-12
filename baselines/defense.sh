#!/bin/bash

data_folder=$1
prefix=$2
victim=$3

data="${data_folder}/${prefix}-concat.npz"
# echo $data

defense_types=("srs" "dup" "sor")
for i in ${!defense_types[@]}; do
    tp=${defense_types[$i]}
    echo "$tp defense and inference"
    CUDA_VISIBLE_DEVICES=4 python defend_npz.py --data_root=$data --defense=$tp
    tp_folder="$data_folder/$tp"
    tp_prefix="${tp}_${prefix}"
    # echo $tp_folder
    # echo $tp_prefix
    ./inf.sh 4 $tp_folder $tp_prefix $victim
done

echo "ConvOnet Defense and inference"
CUDA_VISIBLE_DEVICES=4 python ../ConvONet/opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=$data
tp_folder="$data_folder/ConvONet-Opt"
tp_prefix="$convonet_opt-$prefix"
./inf.sh 4 $tp_folder $tp_prefix $victim

echo "Onet-Remesh Defense and inference"
CUDA_VISIBLE_DEVICES=4 python ../ONet/remesh_defense.py --sample_npoint=1024 --train=False --data_root=$data
tp_folder="$data_folder/ONet-Mesh"
tp_prefix="$onet_remesh-$prefix"
./inf.sh 4 $tp_folder $tp_prefix $victim

echo "Onet-Opt Defense and inference"
CUDA_VISIBLE_DEVICES=4 python ../ONet/opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=$data
tp_folder="$data_folder/ONet-Opt"
tp_prefix="$onet_opt-$prefix"
./inf.sh 4 $tp_folder $tp_prefix $victim