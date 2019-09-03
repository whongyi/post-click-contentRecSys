#!/bin/sh

# Experiments settings for WRMF-NR model with different hyperparameters
# Note that WRMF-BL is a special case of WRMF-NR if we fix `neg_ratio` to 0
for l2_reg in 0.1 0.01 0.001 0.0001; do
    for pos_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
        for neg_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
            python3 wrmf_postclick.py --dataset=$1 --l2_reg=$l2_reg --pos_ratio=$pos_ratio --neg_ratio=$neg_ratio --log --eval_explicit
        done
    done
done

# Experiemnt setting for the standard WRMF model 
# `neg_ratio` is set to `None` to use stratified pointwise sampleing during training
for l2_reg in 0.1 0.01 0.001 0.0001; do
    for pos_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
        python3 wrmf_postclick.py --dataset=$1 --l2_reg=$l2_reg --pos_ratio=$pos_ratio --neg_ratio=None --log --eval_explicit
    done
done

# To evaluate the performance on click-only data, i.e. the conventional implicit feedback evaluation setting, remove the `eval_explicit` flag. For example: 
# python3 wrmf_postclick.py --dataset=bytedance --l2_reg=0.01 --pos_ratio=0.4 --neg_ratio=0.2 --log 
