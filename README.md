# Post-click Feeback for Recommender Systems
This is the code repo for our RecSys 2019 paper: [Leveraging Post-click Feedback for Content Recommendations](https://cornell-nyc-sdl-postclick-recsys.s3.amazonaws.com/paper.pdf). In this paper, we leverage post-click feedback, e.g. skips and completions, to improve the training and evaluation of content recommenders. Check our paper for more details.

# Install
We used [OpenRec](https://github.com/ylongqi/openrec) to build our recommendation algorithms. It is built on the [TensorFlow](https://github.com/tensorflow/tensorflow) framework. 

To install the dependencies needed for this repo:
```
$ ./scripts/install.sh
```

# Data
We randomly sampled data from two publicly-available datasets for experiments. For pre-processing, please refer to our paper.
- [ByteDance](https://biendata.com/competition/icmechallenge2019/). Contains user interactions with short videos (average 10 seconds in length), including whether or not each video was completed.
- [Spotify](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge). Contains music listening sessions across Spotify users. A user may skip or complete listening to each song.

After processing data into train, validaion, test sets, put them under a "dataset" folder. The directory format is referred to `dataloader.py`.

You can also use `download_dataset.sh` to start with the data preprocessed by us:
```
$ ./scripts/download_dataset.sh
```


# Train and Evaluate
To train a BPR-NR model on the Bytedance dataset under the post-click-aware evaluation metric (click-complete as positive observation, click-skip as negative observation, non-click as missing observation):
```
$ python3 bpr_postclick.py --dataset=bytedance --l2_reg=0.01 --p_n_ratio=0.4 --eval_explicit
```
where `p_n_ratio` corresponds to the hyperparameter $\lambda_p,n$ in the paper. It controls the weights put on each types of signals, and can be any float between 0 to 1.

If you want to see the separated performance on click-skip and click-complete items, add the `eval_rank` flag:
```
$ python3 bpr_postclick.py --dataset=bytedance --l2_reg=0.01 --p_n_ratio=0.4 --eval_explicit --eval_rank
```

If you want to evaluate on the click-only metric (click as positive, non-click as negative), remove the `eval_explicit` flag:
```
$ python3 bpr_postclick.py --dataset=bytedance --l2_reg=0.01 --p_n_ratio=0.4
```

Similarly, to train a WRMF-NR model on the Spotify dataset, we use two hyperparameters to control the weights on positive and negative samples:
```
$ python3 wrmf_postclick.py --dataset=spotify --l2_reg=0.001 --pos_ratio=0.6 --neg_ratio=0.2 --eval_explicit
```

# Experiments
To replicate the experiments in the paper, refer to `./scripts/{bpr,wrmf}_exp.sh`. Assign dataset to be one of {bytedance, spotify} when you run the script. You can also add your own experiments following the instructions in those scripts:
```
$ ./scripts/wrmf_exp.sh bytedance 
```

# Citation
To cite our paper:
```
Hongyi Wen, Longqi Yang, and Deborah Estrin. 2019. Leveraging Post-click
Feedback for Content Recommendations. In Thirteenth ACM Conference on
Recommender Systems (RecSys ’19), September 16–20, 2019, Copenhagen, Denmark.
ACM, New York, NY, USA, 9 pages.
```

```
@inproceedings{wen2019leveraging,
    title={Leveraging Post-click Feedback for Content Recommendations},
    author={Wen, Hongyi and Yang, Longqi and Estrin, Deborah},
    booktitle={Proceedings of the 13th ACM Conference on Recommender Systems},
    year={2019},
    organization={ACM}
}
```

# Contact
If you have questions related to this repo, feel free to raise an issue, or contact us via:
- Email: hw557@cornell.edu
- Twitter: [@hongyi_wen](https://twitter.com/hongyi_wen)
