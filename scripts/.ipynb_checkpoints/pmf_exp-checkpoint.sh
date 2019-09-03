#!/bin/sh
trap "exit" INT

for l2_reg in 0.1 0.01 0.001; do
    for pos_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
        for neg_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
            python3 pmf_implicit.py --dataset=$1 --l2_reg=$l2_reg --pos_ratio=$pos_ratio --neg_ratio=$neg_ratio --log=True
        done
    done
done
