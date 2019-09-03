import numpy as np
import random
from openrec.utils.samplers import Sampler

def StratifiedPairwiseSampler(dataset, batch_size, p_n_ratio, num_process=5, seed=100):
    
    random.seed(seed)
    
    def batch(dataset, batch_size=batch_size, seed=seed):
        
        num_pos = int(p_n_ratio * batch_size)
        
        while True:
            input_npy = np.zeros(batch_size, dtype=[('user_id', np.int32),
                                                    ('p_item_id', np.int32),
                                                    ('n_item_id', np.int32)])
            # sample (positive, negative) pairs
            pos_ind = 0
            while pos_ind < num_pos:
                entry = dataset.next_random_record()
                user_id = entry['user_id']
                counter_entry = dataset.random_positive_record(user_id)
                if entry['neg_implicit'] != counter_entry['neg_implicit']:
                    if entry['neg_implicit'] == False:
                        p_item_id = entry['item_id']
                        n_item_id = counter_entry['item_id']
                    else:
                        p_item_id = counter_entry['item_id']
                        n_item_id = entry['item_id']
                    input_npy[pos_ind] = (user_id, p_item_id, n_item_id)
                    pos_ind += 1
                
                
            # sample (positive, unobserved) pairs
            ind = 0
            while ind < batch_size - num_pos:
                entry = dataset.next_random_record()
                if entry['neg_implicit'] == False:
                    user_id = entry['user_id']
                    p_item_id = entry['item_id']
                    n_item_id = dataset.sample_negative_items(user_id)[0]
                    input_npy[ind + num_pos] = (user_id, p_item_id, n_item_id)
                    ind += 1
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s