import tensorflow as tf
import numpy as np
import random

def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def merge_dicts(*list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp