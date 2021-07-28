import numpy as np
import random
from openrec.utils.samplers import Sampler


def NegativePointwiseSampler(batch_size, dataset, pos_ratio=0.5, neg_ratio=0.3, num_process=5, seed=100):
    
    random.seed(seed)
    def batch(dataset=dataset, batch_size=batch_size, seed=seed):
        
        num_pos = int(batch_size * pos_ratio)
        num_neg = int(batch_size * neg_ratio)
        
        while True:
            
            input_npy = np.zeros(batch_size, dtype=[('user_id', np.int32),
                                                    ('item_id', np.int32),
                                                    ('label', np.float32)])
            
            pos_ind = 0
            neg_ind = 0
            ind = 0
            while pos_ind + neg_ind < num_pos + num_neg:
                entry = dataset.next_random_record()
                if entry['neg_implicit'] == False and pos_ind < num_pos:
                    input_npy[ind] = (entry['user_id'], entry['item_id'], 1.0)
                    pos_ind += 1
                    ind += 1
                    
                if entry['neg_implicit'] == True and neg_ind < num_neg:
                    input_npy[ind] = (entry['user_id'], entry['item_id'], 0.0)
                    neg_ind += 1
                    ind += 1

            for i in range(batch_size - num_pos - num_neg):
                user_id = random.randint(0, dataset.total_users()-1)
                item_id = random.randint(0, dataset.total_items()-1)
                while dataset.is_positive(user_id, item_id):
                    user_id = random.randint(0, dataset.total_users()-1)
                    item_id = random.randint(0, dataset.total_items()-1)
                input_npy[ind+i] = (user_id, item_id, 0.0)
            
            yield input_npy
        
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s
