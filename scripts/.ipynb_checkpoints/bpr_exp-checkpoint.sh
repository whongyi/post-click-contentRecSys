#!/bin/sh
trap "exit" INT

for l2_reg in 0.1 0.01 0.001; do
    for p_n_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
        python3 bpr_implicit.py --dataset=$1 --l2_reg=$l2_reg --p_n_ratio=$p_n_ratio --log=True
    done
done
