import numpy as np
import sys, os
import argparse
from openrec import ModelTrainer
from openrec.recommenders import PMF # this openrec implementation shall be referred to as WRMF to avoid confusion
from openrec.utils import Dataset
from openrec.utils.evaluators import AUC 
from openrec.utils.samplers import StratifiedPointwiseSampler, EvaluationSampler
from negative_pointwise_sampler import NegativePointwiseSampler
from dataloader import *


### training parameter ###
total_iter = 10000   # iterations for training 
batch_size = 1000      # training batch size
eval_iter = 1000        # iteration of evaluation
save_iter = eval_iter   # iteration of saving model

### embeding ### 
dim_user_embed = 100     # dimension of user embedding
dim_item_embed = 100     # dimension of item embedding


def exp(dataset, l2_reg, pos_ratio, neg_ratio, eval_explicit, save_log, eval_rank):
    
    if neg_ratio is not None:
        if pos_ratio + neg_ratio > 1.0 or pos_ratio + neg_ratio <= 0.0:
            print ("Invalid sampling ratios...")
            return
    
    if dataset == 'spotify':
        data = loadSpotify()
        
    elif dataset == 'bytedance':
        data = loadByteDance()
        
    else:
        print ("Unsupported dataset...")
        return 
    
    # save logging and model
    log_dir = "validation_logs/{}_{}_{}_{}_{}_{}/".format(dataset, l2_reg, pos_ratio, neg_ratio, eval_explicit, eval_rank)
    os.popen("mkdir -p %s" % log_dir).read()
    if save_log:
        log = open(log_dir + "validation.log", "w")
        sys.stdout = log
    
    
    # prepare train, val, test sets and samplers
    train_dataset = Dataset(data['train'], data['total_users'], data['total_items'], name='Train')    
    if neg_ratio is None:
        train_sampler = StratifiedPointwiseSampler(batch_size=batch_size, 
                                                   dataset=train_dataset, 
                                                   pos_ratio=pos_ratio, 
                                                   num_process=5)
    else:
        train_sampler = NegativePointwiseSampler(batch_size=batch_size, 
                                                 dataset=train_dataset, 
                                                 pos_ratio=pos_ratio, 
                                                 neg_ratio=neg_ratio, 
                                                 num_process=5)
        if neg_ratio > 0.0:
            print ("Re-weighting implicit negative feedback")
        else:
            print ("Corrected negative feedback labels but not re-weighting")
    
    
    eval_num_neg = None if eval_explicit else 500 # num of negative samples for evaluation
    if eval_rank:
        # show evaluation metrics for click-complete and click-skip items separately
        pos_dataset = Dataset(data['pos_test'],  data['total_users'], data['total_items'], 
                              implicit_negative=not eval_explicit, name='Pos_Test', num_negatives=eval_num_neg)
        neg_dataset = Dataset(data['neg_test'],  data['total_users'], data['total_items'], 
                              implicit_negative=not eval_explicit, name='Neg_Test', num_negatives=eval_num_neg)
        pos_sampler = EvaluationSampler(batch_size=batch_size, dataset=pos_dataset)
        neg_sampler = EvaluationSampler(batch_size=batch_size, dataset=neg_dataset)
        eval_samplers = [pos_sampler, neg_sampler]
    else:
        val_dataset = Dataset(data['val'],  data['total_users'], data['total_items'], 
                              implicit_negative=not eval_explicit, name='Val', num_negatives=eval_num_neg)
        test_dataset = Dataset(data['test'],  data['total_users'], data['total_items'], 
                               implicit_negative=not eval_explicit, name='Test', num_negatives=eval_num_neg)
        val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
        test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)
        eval_samplers = [val_sampler, test_sampler]
    
    # set evaluators
    auc_evaluator = AUC()
    evaluators = [auc_evaluator]
  
    
    # set model parameters
    model = PMF(l2_reg=l2_reg, 
                batch_size=batch_size, 
                total_users=train_dataset.total_users(), 
                total_items=train_dataset.total_items(), 
                dim_user_embed=dim_user_embed, 
                dim_item_embed=dim_item_embed, 
                save_model_dir=log_dir, 
                train=True, 
                serve=True)
    
    
    # set model trainer
    model_trainer = ModelTrainer(model=model)  
    model_trainer.train(total_iter=total_iter, 
                        eval_iter=eval_iter, 
                        save_iter=save_iter, 
                        train_sampler=train_sampler, 
                        eval_samplers=eval_samplers, 
                        evaluators=evaluators)

    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parse parameters')
    parser.add_argument('--dataset', type=str, default='bytedance', help='dataset to use')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='l2 regularization of latent factor')
    parser.add_argument('--pos_ratio', type=float, default=0.5, help='pos ratio of sampling')
    parser.add_argument('--neg_ratio', type=float, default=None, help='negative ratio of sampling')
    parser.add_argument('--eval_explicit', action='store_true', help='turn on to use labels to evaluate, by default treat click as positive and non-click as negative')
    parser.add_argument('--eval_rank', action='store_true', help='show ranking accuracy for pos and neg samples')
    parser.add_argument('--log', action='store_true', help='turn on for logging results to file, by default will print on screen')
    args = parser.parse_args()
    print (args)
    
    # run experiments
    exp(dataset=args.dataset, l2_reg=args.l2_reg, pos_ratio=args.pos_ratio, neg_ratio=args.neg_ratio, eval_explicit=args.eval_explicit, save_log=args.log, eval_rank=args.eval_rank)


    
