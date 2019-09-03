#!/bin/sh

# Experiments settings for BPR-NR models with different hyperparameters
# Note that BPR-BL is a special case of BPR-NR if we fix `p_n_ratio` to 0
for l2_reg in 0.1 0.01 0.001 0.0001; do
    for p_n_ratio in 0.0 0.2 0.4 0.6 0.8 1.0; do
        python3 bpr_postclick.py --dataset=$1 --l2_reg=$l2_reg --p_n_ratio=$p_n_ratio --log --eval_explicit
    done
done


# Experiment settings for BPR, note that here `p_n_ratio` is set to `None` so that the model will use standard pairwise sampling during training
for l2_reg in 0.1 0.01 0.001 0.0001; do
    python3 bpr_postclick.py --dataset=$1 --l2_reg=$l2_reg --p_n_ratio=None --log --eval_explicit
done

# To evaluate the performance on click-only data, i.e. the conventional implicit feedback evaluation setting, remove the `eval_explicit` flag. For example: 
# python3 bpr_postclick.py --dataset=bytedance --l2_reg=0.01 --p_n_ratio=0.4 --log 

