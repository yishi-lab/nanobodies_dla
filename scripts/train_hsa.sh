#! /bin/bash
python ../train.py \
        --input_path ../data/hsa/2019_12_19_16_41/ \
        --gpu False \
        --train_file train_high.npz \
        --test_file test_high.npz \
        --env high \
        --dataset hsa \
        --num_classes 2